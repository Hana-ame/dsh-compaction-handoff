/**
 * compaction-handoff — handoff-brief compaction engine.
 *
 * Base DSH compaction (`dsh-compaction-basic`) condenses history with a
 * generic "checkpoint summary". This subclass replaces the summarizer
 * instruction with the *handoff brief* discipline (the end-of-dialog
 * compaction technique from the "you don't need long context" line of work):
 *
 *   - when the context crosses the threshold, the model hands in ONE
 *     structured work-state brief — a standardized handover document
 *     (objective / progress / verified facts / open problems / next action /
 *     standing constraints) written for a successor that has NO other memory;
 *   - every re-compaction re-merges the prior brief into a single updated
 *     brief instead of stacking summaries;
 *   - the base engine's replay and durability strategy stays untouched: the
 *     oldest balanced span is replaced by the framed brief, the recent tail
 *     stays verbatim.
 *
 * The base class documents `summarize()` as "the sole subclass customization
 * hook — override this for a template or remote summarizer", but the module
 * keeps `summarizeWithLlm` (envelope) and `resolveTargetPolicy` (per-model
 * merge) private, so the primary call is replicated here field-for-field from
 * `dsh-compaction-basic/lib/index.js` with only the instruction swapped.
 *
 * The summarizer fallback chain mirrors dsh-compaction-fallback: the primary
 * call follows the conversation's routed target to reuse the warm KV prefix
 * cache; on refusal / 429 / truncation / overflow it walks `fallbackTargets`.
 *
 * Mount in place of `@deepseek-ai/dsh-compaction-basic` (same config keys,
 * plus `fallbackTargets`):
 *
 *   - id: compaction-basic
 *     name: '../../plugins/dsh-compaction-handoff/index.js'
 *     config:
 *       fallbackTargets:
 *         - provider: sensenova-pool
 *           model: deepseek-flash
 *
 * @module dsh-compaction-handoff
 */

import z from '@deepseek-ai/schemastery'
import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'
import { isCompactCheckpointSource, toolPairingBalancedAfter, toolPairingBalancedBefore } from '@deepseek-ai/dsh-compaction'
import { BlockAssembler, LlmError, contentHasImage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { deepFreeze } from '@deepseek-ai/dsh-util-values'

/** Source kind of the per-round window reminder this engine injects. */
const REMINDER_SOURCE_KIND = 'lean-window-reminder'

/**
 * Foldable-spans floor. A framed handoff brief costs roughly 500–900 tokens;
 * a span smaller than this cannot shrink the context, and the base engine
 * rejects such a transaction ("summary is not smaller than the shadowed
 * content"). Skipping below the floor is the correct behavior — a `null`
 * compaction instead of a failed one.
 */
const MIN_FOLDABLE_TOKENS = 1024

/** Tags wrapping the structured summary inside the landed checkpoint node. */
const SUMMARY_OPEN_TAG = '<compacted-summary>'

/**
 * The handoff-brief summarizer instruction, delivered as the FINAL user
 * message after the replayed conversation prefix.
 */
const COMPACTION_INSTRUCTION = [
  'You are now acting as the outgoing engineer on this AI coding assistant\'s shift. The conversation ABOVE is your work log. Write the handoff brief a successor continues from, with no other memory of that log.',
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Objective',
  '- [the task exactly as it stands now: original goal plus any later changes; quote critical wording verbatim]',
  '',
  '## Progress so far',
  '- [what is already done and observable, newest last; one bullet per deliverable or decision]',
  '',
  '## Verified facts',
  '- [findings you actually confirmed with evidence: exact paths, line numbers, command outputs, identifiers, versions; no guesses]',
  '',
  '## Open problems',
  '- [what is broken, unresolved, or still to verify; include exact error strings and what has already been tried]',
  '',
  '## Next action',
  '- [the single next step the successor should take immediately; concrete and directly in line with the objective]',
  '',
  '## Standing constraints',
  '- [user preferences, conventions, do/don\'t rules, and decisions with their rationale that must not be lost]',
  '',
  'Rules:',
  '- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.',
  '- The brief replaces the whole log above: anything the successor needs MUST be in it, because the log disappears.',
  '- Capture user feedback and explicit instructions faithfully, especially corrections.',
  '- Do NOT mention this handoff request or that the context was compacted.',
  '- Output only the brief text: do not call any tool or take any other action.',
  `- If the conversation already contains a ${SUMMARY_OPEN_TAG} block, it is a PRIOR handoff brief. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated brief under the same structure.`,
].join('\n')

/**
 * Summary-only variant of the instruction. In that mode the model never sees
 * verbatim history: the replayed prefix is a prior handoff brief followed by
 * exactly one completed round, and its output becomes the next round's sole
 * memory. The output format is identical, so briefs stay interchangeable.
 */
const SUMMARY_ONLY_INSTRUCTION = [
  "You maintain the working memory of an AI coding assistant that operates on ONE rolling brief: it never sees raw conversation history, only the brief you write. The conversation ABOVE is the previous brief followed by the round that just completed. Merge them into the NEXT round's brief.",
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section - never drop a section.',
  '',
  '## Objective',
  '- [the task exactly as it stands now: original goal plus any later changes; quote critical wording verbatim]',
  '',
  '## Progress so far',
  '- [what is already done and observable, newest last; one bullet per deliverable or decision]',
  '',
  '## Verified facts',
  '- [findings actually confirmed with evidence: exact paths, line numbers, command outputs, identifiers, versions; no guesses]',
  '',
  '## Open problems',
  '- [what is broken, unresolved, or still to verify; include exact error strings and what has already been tried]',
  '',
  '## Next action',
  '- [the single next step the successor should take immediately; concrete and directly in line with the objective]',
  '',
  '## Standing constraints',
  '- [user preferences, conventions, do/don\'t rules, and decisions with their rationale that must not be lost]',
  '',
  'Rules:',
  '- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.',
  '- The brief you write REPLACES both the previous brief and the round above: anything the next round needs MUST be in it.',
  '- Carry forward every still-true fact from the previous brief; drop only what the new round proved stale or superseded.',
  '- Capture user feedback and explicit instructions faithfully, especially corrections.',
  '- Do NOT mention this handoff request or that the context was compacted.',
  '- Output only the brief text: do not call any tool or take any other action.',
].join('\n')

/* --- Config schema: base fields plus the fallback chain. --- */

const thresholdRatioSchema = z.number()
const headroomTokensSchema = z.number().step(1).min(0)
const retainRatioSchema = z.number()
const retainTokensSchema = z.number().step(1).min(0)
const summarizationProviderSchema = z.string()
const summarizationModelSchema = z.string()
const maxTokensSchema = z.number().step(1).min(1)
const compactionRetriesSchema = z.number().step(1).min(0)
const maxOverflowRetriesSchema = z.number().step(1).min(0)

const modelPolicy = z.object({
  provider: z.string().required(),
  model: z.string().required(),
  thresholdRatio: thresholdRatioSchema,
  headroomTokens: headroomTokensSchema,
  retainRatio: retainRatioSchema,
  retainTokens: retainTokensSchema,
  summarizationProvider: summarizationProviderSchema,
  summarizationModel: summarizationModelSchema,
  maxTokens: maxTokensSchema,
  compactionRetries: compactionRetriesSchema,
  maxOverflowRetries: maxOverflowRetriesSchema,
})

const fallbackTarget = z.object({
  provider: z.string().required(),
  model: z.string().required(),
  maxTokens: maxTokensSchema,
})

/** Map a terminal summarization finish to its fail-closed error. */
function finishError(finish) {
  switch (finish.kind) {
    case 'error':
    case 'aborted':
      return new LlmError(finish.failure.message, finish.failure.code, finish.failure)
    case 'max-tokens': {
      const error = new Error('summarization truncated at the token cap (incomplete handoff brief)')
      error.code = 'MAX_TOKENS'
      return error
    }
    default:
      return undefined
  }
}

/** Reject visual output and keep only text before synthesizing a user message. */
function summaryText(blocks) {
  if (contentHasImage(blocks)) {
    throw new LlmError('compaction summary cannot contain image output', 'UNSUPPORTED_CONTENT')
  }
  return blocks.filter((block) => block.type === 'text')
}

/** A resolved provider/model call target. */
function targetOf(provider, model) {
  return { provider, model }
}

/** The `system/message` event at a surface seq, when that node is one. */
function systemHead(session, headSeq) {
  const head = session.eventAt(headSeq)
  return head.type === 'system/message' ? head : undefined
}

/**
 * Select the oldest balanced span to fold, keeping `retainTokens` of the
 * newest surface text verbatim. The same selection the base engine performs
 * under pressure; replicated here (the base keeps it module-private) for the
 * always-compact path. Never splits an assistant tool-call/result pair.
 *
 * @param session - session supplying current surface positions.
 * @param measurement - unified pressure and surface measurement.
 * @param retainTokens - newest tail budget kept verbatim.
 * @returns the inclusive surface seq range, or null when nothing is foldable.
 */
function selectCompactableRange(session, measurement, retainTokens) {
  const pricedNodes = measurement.nodes
  if (pricedNodes.length === 0) return null
  const surfaceNodes = session.surface.nodes
  if (surfaceNodes.length !== pricedNodes.length || surfaceNodes.some((seq, index) => seq !== pricedNodes[index]?.seq)) {
    throw new Error('compaction: token-meter surface does not match the current session surface')
  }
  const firstIdx = systemHead(session, surfaceNodes[0]) === undefined ? 0 : 1
  let accumulated = 0
  let keepFromIdx = pricedNodes.length
  for (let index = pricedNodes.length - 1; index >= 0; index -= 1) {
    accumulated += pricedNodes[index].tokens
    keepFromIdx = index
    if (accumulated >= retainTokens) break
  }
  if (keepFromIdx <= firstIdx) return null
  while (keepFromIdx > firstIdx) {
    if (toolPairingBalancedBefore(session, surfaceNodes[keepFromIdx])) break
    keepFromIdx -= 1
  }
  if (keepFromIdx <= firstIdx) return null
  return {
    start: surfaceNodes[firstIdx],
    end: surfaceNodes[keepFromIdx - 1],
  }
}

/**
 * Select the fold range for a hard ROUND cap: everything before the last
 * `keepRounds` user rounds is folded; those rounds (plus everything after
 * them) stay verbatim. Synthetic context messages never count as rounds:
 * compaction checkpoints and this engine's own window reminders are skipped.
 *
 * @param session - session supplying the surface.
 * @param measurement - token-meter measurement for the same surface.
 * @param keepRounds - how many newest user rounds stay verbatim.
 * @returns the balanced inclusive surface range, or null when nothing folds.
 */
/**
 * Select everything BEFORE the current round, for summary-only operation:
 * the model works from the previous handoff brief plus only the round in
 * progress, so each round starts by folding the prior brief together with
 * the round that just ended into a fresh brief (that brief being the next
 * round's only memory).
 *
 * The current round's own messages are kept verbatim while it runs — a tool
 * loop cannot work from a summary of its own unfinished work.
 *
 * @param session - session supplying current surface positions.
 * @param measurement - unified pressure and surface measurement.
 * @returns the inclusive surface seq range to fold, or null when the surface
 *   holds nothing newer than the existing briefs.
 */
function selectSummaryOnlyRange(session, measurement) {
  const pricedNodes = measurement.nodes
  const surfaceNodes = session.surface.nodes
  if (surfaceNodes.length === 0 || pricedNodes.length !== surfaceNodes.length) return null
  const firstIdx = systemHead(session, surfaceNodes[0]) === undefined ? 0 : 1
  // the current round begins at the newest real user message (reminders and
  // briefs are engine artifacts and never start a round).
  let currentRoundIdx = -1
  for (let index = surfaceNodes.length - 1; index >= firstIdx; index -= 1) {
    const event = session.eventAt(surfaceNodes[index])
    if (event?.type !== 'user/message') continue
    const messageSource = event.data?.source ?? event.data?.message?.source
    if (messageSource !== undefined && isCompactCheckpointSource(messageSource)) continue
    if (messageSource?.kind === REMINDER_SOURCE_KIND) continue
    currentRoundIdx = index
    break
  }
  if (currentRoundIdx < 0) return null
  // The newest real user message is only "the round in progress" when nothing
  // has answered it yet. At the next round's first pre-step the inbox has not
  // been spliced, so the newest user message is still the round that just
  // FINISHED (it has assistant/tool activity after it) — that round is folded,
  // and relying on it as the boundary would leave only harness context blocks
  // in the span.
  let answered = false
  for (let index = currentRoundIdx + 1; index < surfaceNodes.length; index += 1) {
    const event = session.eventAt(surfaceNodes[index])
    if (event?.type === 'assistant/message') { answered = true; break }
  }
  let startIdx = firstIdx
  let endIdx = answered ? surfaceNodes.length - 1 : currentRoundIdx - 1
  if (endIdx < startIdx) return null
  while (endIdx >= startIdx && !toolPairingBalancedAfter(session, surfaceNodes[endIdx])) endIdx -= 1
  while (startIdx < endIdx && !toolPairingBalancedBefore(session, surfaceNodes[startIdx])) startIdx += 1
  if (startIdx > endIdx) return null
  // Nothing new since the last brief: folding a brief into itself only loses
  // information. Also require at least one COMPLETED round inside the span —
  // before the first round ends the span holds only harness-injected context
  // blocks, which must not be folded (and are far too small to fold anyway).
  let fresh = 0
  let completedRounds = 0
  for (let index = startIdx; index <= endIdx; index += 1) {
    const event = session.eventAt(surfaceNodes[index])
    const messageSource = event?.data?.source ?? event?.data?.message?.source
    if (messageSource !== undefined && isCompactCheckpointSource(messageSource)) continue
    if (messageSource?.kind === REMINDER_SOURCE_KIND) continue
    fresh += 1
    // Only a REAL user round counts: harness-injected blocks (agent-instructions,
    // runtime-context, time-context, system-reminder) also ride the user role
    // but they are context plumbing, never a completed round.
    if (event?.type === 'user/message' && messageSource?.kind === 'user') completedRounds += 1
    if (event?.type === 'user/message' && messageSource?.kind === undefined) completedRounds += 1
  }
  if (fresh === 0 || completedRounds === 0) return null
  return { start: surfaceNodes[startIdx], end: surfaceNodes[endIdx] }
}

function selectRoundsRange(session, measurement, keepRounds) {
  const pricedNodes = measurement.nodes
  const surfaceNodes = session.surface.nodes
  if (surfaceNodes.length === 0 || pricedNodes.length !== surfaceNodes.length) return null
  const firstIdx = systemHead(session, surfaceNodes[0]) === undefined ? 0 : 1
  const roundIndexes = []
  for (let index = firstIdx; index < surfaceNodes.length; index += 1) {
    const event = session.eventAt(surfaceNodes[index])
    if (event?.type !== 'user/message') continue
    // Surface user/message events carry the message fields directly on `data`
    // ({content, source, role, id}); some projections nest them under
    // `data.message`, so read both shapes.
    const messageSource = event.data?.source ?? event.data?.message?.source
    if (messageSource !== undefined && isCompactCheckpointSource(messageSource)) continue
    if (messageSource?.kind === REMINDER_SOURCE_KIND) continue
    roundIndexes.push(index)
  }
  if (roundIndexes.length <= keepRounds) return null
  // first user message of the retained window: fold everything before it.
  let startIdx = firstIdx
  let endIdx = roundIndexes[roundIndexes.length - keepRounds] - 1
  if (endIdx < startIdx) return null
  // never split an assistant tool-call/result pair at either edge.
  while (endIdx >= startIdx && !toolPairingBalancedAfter(session, surfaceNodes[endIdx])) endIdx -= 1
  while (startIdx < endIdx && !toolPairingBalancedBefore(session, surfaceNodes[startIdx])) startIdx += 1
  if (startIdx > endIdx) return null
  return {
    start: surfaceNodes[startIdx],
    end: surfaceNodes[endIdx],
  }
}

class HandoffCompactionEngine extends BasicCompactionEngine {
  static Config = z.object({
    thresholdRatio: thresholdRatioSchema,
    headroomTokens: headroomTokensSchema,
    retainRatio: retainRatioSchema,
    retainTokens: retainTokensSchema,
    summarizationProvider: summarizationProviderSchema,
    summarizationModel: summarizationModelSchema,
    maxTokens: maxTokensSchema,
    compactionRetries: compactionRetriesSchema,
    maxOverflowRetries: maxOverflowRetriesSchema,
    modelPolicies: z.array(modelPolicy),
    auto: z.boolean(),
    alwaysCompact: z.boolean(),
    alwaysMinSpanTokens: z.number().step(1).min(0),
    keepRounds: z.number().step(1).min(0),
    summaryOnly: z.boolean(),
    remindOnRound: z.number().step(1).min(0),
    reminder: z.string(),
    fallbackTargets: z.array(fallbackTarget),
  })

  constructor(ctx, config = {}) {
    const { fallbackTargets = [], alwaysCompact = false, alwaysMinSpanTokens = 1024, keepRounds = 0, summaryOnly = false, remindOnRound, reminder, ...baseConfig } = config ?? {}
    super(ctx, baseConfig)
    this.fallbackTargets = (fallbackTargets ?? []).map((target) => Object.freeze({ ...target }))
    /** Single-action compression: fold the history above the tail at EVERY step. */
    this.alwaysCompact = alwaysCompact === true
    /** Foldable spans below this token count are skipped (a brief could not shrink them). */
    this.alwaysMinSpanTokens = alwaysMinSpanTokens
    /**
     * Summary-only operation: the model never sees verbatim history. Every
     * round starts with the previous handoff brief folded together with the
     * round that just ended, so the brief is the sole memory; the round in
     * progress stays verbatim until it completes (a tool loop cannot work
     * from a summary of its own unfinished work).
     */
    this.summaryOnly = summaryOnly === true
    /** Hard cap: how many newest user rounds stay verbatim (0 = token-tail mode). */
    this.keepRounds = Number.isInteger(keepRounds) && keepRounds > 0 ? keepRounds : 0
    /**
     * Which turn number carries the window reminder. `undefined` defaults to
     * the last retained round (keepRounds); an explicit 0 disables it.
     */
    this.remindOnRound = this.summaryOnly
      ? 1 // summary-only: every round works from the brief, so every round is reminded
      : Number.isInteger(remindOnRound) && remindOnRound >= 0
        ? remindOnRound
        : this.keepRounds
    /** Model-facing reminder text; empty means no injection. */
    const defaultReminder = this.summaryOnly
      ? '【窗口提醒】本会话按「单轮记忆」运行：你不会看到任何历史原文，唯一的记忆是上面的交接小结。需要更早的细节时先读小结，或重新用工具取证；本轮结束后这份小结会被重写成下一轮的版本，务必把关键事实写进小结而不是留在对话里。'
      : `【窗口提醒】本会话只保留最近 ${this.keepRounds} 轮原文；更早的历史已折叠成上面的交接小结。需要更早的细节时，先读小结或重新用工具取证，不要凭记忆补细节。`
    this.reminder = (this.summaryOnly || this.keepRounds > 0) && this.remindOnRound > 0
      ? (reminder === undefined ? defaultReminder : reminder)
      : ''
    /** Agents already reminded for a specific turn. */
    this.remindedTurns = new WeakMap()
    if (this.reminder.length > 0) this.registerReminder()
  }

  /**
   * Inject the window reminder at the configured turn's first step only
   * (`remindOnRound`), after the loop's own pre-step decision settled. The
   * message is a normal user-role message on the logged channel (same shape
   * the model-switch notice uses), so the model sees it exactly on that round
   * and the log stays authoritative.
   */
  registerReminder() {
    this.ctx.on('agent/pre-step', async (payload, next) => {
      const decision = await next()
      if (decision.kind === 'reject' || payload.signal?.aborted) return decision
      // summary-only reminds on every round (the brief is the only memory);
      // otherwise only the configured round carries it.
      if (payload.step !== 1) return decision
      if (!this.summaryOnly && payload.turn !== this.remindOnRound) return decision
      if (this.remindedTurns.get(payload.agent) === payload.turn) return decision
      this.remindedTurns.set(payload.agent, payload.turn)
      const notice = createUserMessage({
        content: [{ type: 'text', text: this.reminder }],
        source: {
          kind: REMINDER_SOURCE_KIND,
          form: 'notice',
          summary: this.summaryOnly
            ? `窗口提醒：单轮记忆模式（第 ${payload.turn} 轮）`
            : `窗口提醒：保留最近 ${this.keepRounds} 轮（第 ${this.remindOnRound} 轮）`,
        },
      })
      return { ...decision, messages: [...decision.messages, notice] }
    }, { prepend: true })
  }

  /**
   * The conversation's routed target: the latest durable request header when
   * it carries both fields, else the agent options pair (the base class's
   * private `conversationTarget` / `routedTarget` pair, replicated).
   *
   * @param agent - agent owning the session.
   * @returns the conversation target, or undefined when nothing is routed.
   */
  conversationTarget(agent) {
    const latest = agent.session.requestHeader()?.config
    if (latest !== undefined && (latest.provider?.length ?? 0) > 0 && (latest.model?.length ?? 0) > 0) {
      return targetOf(latest.provider, latest.model)
    }
    const options = agent.options
    if ((options?.provider?.length ?? 0) > 0 && (options?.model?.length ?? 0) > 0) {
      return targetOf(options.provider, options.model)
    }
    return undefined
  }

  /**
   * Effective summarizer settings for the conversation's target, merging the
   * matching exact `modelPolicies` entry over the defaults the same way the
   * base class's private `resolveTargetPolicy` does for the fields the
   * envelope consumes (summarizer pair + output cap).
   *
   * @param agent - agent owning the session.
   * @returns resolved { summarizationProvider, summarizationModel, maxTokens }.
   */
  resolveCallConfig(agent) {
    const base = {
      summarizationProvider: this.config.summarizationProvider,
      summarizationModel: this.config.summarizationModel,
      maxTokens: this.config.maxTokens,
    }
    const conversation = this.conversationTarget(agent)
    if (conversation === undefined) return base
    const override = this.config.modelPolicies.find(
      (policy) => policy.provider === conversation.provider && policy.model === conversation.model,
    )
    if (override === undefined) return base
    return {
      summarizationProvider: override.summarizationProvider ?? base.summarizationProvider,
      summarizationModel: override.summarizationModel ?? base.summarizationModel,
      maxTokens: override.maxTokens ?? base.maxTokens,
    }
  }

  /**
   * Resolve the primary summarization call target exactly as the base
   * envelope does: configured summarizer pair, else the latest routed
   * request, else the agent options pair.
   *
   * @param agent - agent owning the session.
   * @returns the primary provider/model pair, or undefined when nothing is available.
   */
  primaryTarget(agent) {
    const config = this.resolveCallConfig(agent)
    if (config.summarizationProvider.length > 0) {
      return targetOf(config.summarizationProvider, config.summarizationModel)
    }
    return this.conversationTarget(agent)
  }

  /**
   * Automatic-compaction policy hook. The base engine registers the
   * `agent/pre-step` handler, but dispatch is performed on `this`, so this
   * override owns the decision.
   *
   * `alwaysCompact: true` implements the paper's single-action compression:
   * EVERY pre-step folds the balanced history above the retained tail into
   * one handoff brief — no pressure threshold is consulted, so the mounted
   * context stays ~[brief + recent tail + current input] after every exchange.
   * Context-overflow recovery keeps the base implementation.
   *
   * @param agent - agent whose next step may fold history.
   * @param trigger - pressure (every pre-step here) or context overflow.
   * @param signal - turn cancellation forwarded to summarization.
   * @returns the compaction result, or null when nothing foldable qualifies.
   */
  async compactIfNeeded(agent, trigger, signal) {
    if (trigger !== 'pressure' || this.alwaysCompact !== true) {
      return super.compactIfNeeded(agent, trigger, signal)
    }
    signal?.throwIfAborted?.()
    const meter = this.ctx.tokenMeter
    const pruner = this.ctx.get('toolResultPruner')
    if (pruner !== undefined) pruner.pruneSession(agent.session)
    const measurement = meter.measure(agent.session)
    const retainTokens = this.config.retainTokens ?? 2048
    const range = this.summaryOnly
      ? selectSummaryOnlyRange(agent.session, measurement)
      : this.keepRounds > 0
        ? selectRoundsRange(agent.session, measurement, this.keepRounds)
        : selectCompactableRange(agent.session, measurement, retainTokens)
    if (range === null) return null
    const nodes = agent.session.surface.nodes
    const startIdx = nodes.indexOf(range.start)
    const endIdx = nodes.indexOf(range.end)
    if (startIdx < 0 || endIdx < startIdx) return null
    const shadowedTokens = measurement.nodes
      .slice(startIdx, endIdx + 1)
      .reduce((total, node) => total + (node.tokens ?? 0), 0)
    // Skip below the configured floor AND the hard physical floor: a brief
    // cannot be smaller than the span it replaces under that size, so trying
    // would only produce a failed compaction transaction. Summary-only mode
    // folds at every round boundary, so it uses the physical floor only —
    // otherwise tiny rounds would accumulate verbatim and defeat the mode.
    const floor = this.summaryOnly ? 0 : Math.max(this.alwaysMinSpanTokens, MIN_FOLDABLE_TOKENS)
    if (shadowedTokens < floor) return null
    try {
      return await this.compactRegion(range.start, range.end, agent, signal)
    } catch (error) {
      if (signal?.aborted === true) throw error
      // Surface the reason instead of letting the generic pre-step handler
      // swallow it: a repeatedly failing fold silently pins the context.
      const message = String(error?.message ?? error)
      // A tiny round whose brief would be larger than the span is an expected
      // outcome of summary-only operation: it simply waits for more content.
      const expected = this.summaryOnly && message.includes('not smaller than the shadowed content')
      if (!expected) {
        this.ctx.logger.warn(
          `compaction-handoff: fold skipped (${shadowedTokens} tokens in range ${range.start}-${range.end}): ${message}`,
        )
      }
      return null
    }
  }

  /**
   * Summarize into a handoff brief: primary call first (same target and
   * envelope the base engine uses, warm KV prefix cache, only the
   * instruction swapped), then the configured fallback chain on failure.
   *
   * @param input - replayed conversation prefix to condense.
   * @param agent - agent owning the session.
   * @param signal - cancellation forwarded to every attempt.
   * @returns the summary result shape the region transaction expects.
   */
  async summarize(input, agent, signal) {
    const config = this.resolveCallConfig(agent)
    const primary = this.primaryTarget(agent)
    if (primary === undefined) {
      throw new Error(
        'no provider/model available for summarization: set both BasicCompactionConfig summarization fields, route one request, or set both AgentOptions fields',
      )
    }
    try {
      return await this.summarizeWithTarget(input, agent, primary, config.maxTokens, signal)
    } catch (error) {
      if (signal?.aborted === true || this.fallbackTargets.length === 0) throw error
      let lastError = error
      for (const target of this.fallbackTargets) {
        if (signal?.aborted === true) throw lastError
        if (primary.provider === target.provider && primary.model === target.model) continue
        const label = `${target.provider}/${target.model}`
        try {
          const result = await this.summarizeWithTarget(
            input,
            agent,
            targetOf(target.provider, target.model),
            target.maxTokens ?? config.maxTokens,
            signal,
          )
          this.ctx.logger.warn(
            `compaction-handoff: primary summarizer failed (${lastError?.message ?? lastError}); ${label} produced the brief`,
          )
          return result
        } catch (fallbackError) {
          if (signal?.aborted === true) throw fallbackError
          this.ctx.logger.warn(
            `compaction-handoff: fallback ${label} failed: ${fallbackError?.message ?? fallbackError}`,
          )
          lastError = fallbackError
        }
      }
      throw lastError
    }
  }

  /**
   * One handoff-brief summarization call: the base engine's envelope
   * (`purpose: 'compaction'`, tool history, tools, session id, output cap)
   * with the handoff instruction as the final user message.
   *
   * @param input - replayed conversation prefix to condense.
   * @param agent - agent owning the session.
   * @param target - exact provider/model for this attempt.
   * @param maxTokens - output cap for this attempt.
   * @param signal - optional cancellation forwarded to the adapter.
   * @returns safe text-only summary blocks and the call envelope/output.
   */
  async summarizeWithTarget(input, agent, target, maxTokens, signal) {
    const messages = [
      ...input.messages,
      deepFreeze({
        role: 'user',
        content: [{ type: 'text', text: this.summaryOnly ? SUMMARY_ONLY_INSTRUCTION : COMPACTION_INSTRUCTION }],
      }),
    ]
    const options = {
      provider: target.provider,
      model: target.model,
      messages,
      toolHistory: agent.session.toolHistory(),
      ...(input.tools === undefined ? {} : { tools: [...input.tools] }),
      maxTokens,
      sessionId: agent.session.id,
      purpose: 'compaction',
      ...(signal === undefined ? {} : { signal }),
    }
    const assembler = new BlockAssembler()
    for await (const chunk of this.ctx.llm.stream(options)) assembler.push(chunk)
    const error = finishError(assembler.finish)
    if (error !== undefined) throw error
    const rawOutput = assembler.blocks()
    const summary = summaryText(rawOutput)
    if (!summary.some((block) => block.text.trim().length > 0)) {
      throw new Error('summarization produced no text summary content')
    }
    return {
      summary,
      rawOutput,
      llmStreamCall: true,
      provider: target.provider,
      model: target.model,
      maxTokens,
      ...(assembler.usage === undefined ? {} : { usage: assembler.usage }),
    }
  }
}

export { HandoffCompactionEngine, COMPACTION_INSTRUCTION, SUMMARY_ONLY_INSTRUCTION, selectCompactableRange, selectRoundsRange, selectSummaryOnlyRange }
export default HandoffCompactionEngine
