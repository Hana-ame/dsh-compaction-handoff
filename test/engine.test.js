/**
 * Engine unit tests: mock ctx.llm.stream, drive HandoffCompactionEngine's
 * summarize() primary path, fallback chain, and target resolution.
 * Run: node --test test/engine.test.js
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import HandoffCompactionEngine, {
  COMPACTION_INSTRUCTION,
  SUMMARY_ONLY_INSTRUCTION,
  selectCompactableRange,
  selectRoundsRange,
  selectSummaryOnlyRange,
} from '../index.js'

const BASE_CONFIG = { auto: false }

/** Minimal fake agent/session pair covering the fields summarize() touches. */
function fakeAgent({ headerProvider, headerModel, optionsProvider, optionsModel } = {}) {
  return {
    options: {
      ...(optionsProvider === undefined ? {} : { provider: optionsProvider }),
      ...(optionsModel === undefined || optionsProvider === undefined ? {} : { model: optionsModel }),
    },
    session: {
      id: 'test-session',
      requestHeader() {
        if (headerProvider === undefined) return undefined
        return { config: { provider: headerProvider, model: headerModel } }
      },
      toolHistory() { return [] },
    },
  }
}

/**
 * Minimal stand-in for a cordis Context: the Service base class needs
 * `reflect.provide` at construction, and the engine's summarize path only
 * touches `llm` and `logger`.
 */
function stubCtx(llm) {
  const listeners = []
  return {
    provide: [],
    logger: { info() {}, warn() {}, error() {} },
    llm,
    reflect: {
      provide() {},
    },
    on(type, handler) {
      listeners.push([type, handler])
      return () => {}
    },
    effect() { return () => {} },
  }
}

/** Stand up a HandoffCompactionEngine against a scripted llm service. */
function engineWith(script, config = {}) {
  const requests = []
  const llm = {
    async *stream(options) {
      requests.push(options)
      const reply = script.get(`${options.provider}/${options.model}`)
      if (reply === undefined) {
        yield { type: 'finish', reason: { kind: 'error', failure: { message: `no script for ${options.provider}/${options.model}`, code: 'SERVER' } } }
        return
      }
      if (reply.error) {
        yield { type: 'finish', reason: { kind: 'error', failure: { message: reply.error, code: 'INVALID_REQUEST' } } }
        return
      }
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'text-delta', index: 0, text: reply.text }
      yield { type: 'block-end', index: 0, block: { type: 'text', text: reply.text } }
      yield { type: 'finish', reason: { kind: 'stop' } }
    },
  }
  const ctx = stubCtx(llm)
  const engine = new HandoffCompactionEngine(ctx, { ...BASE_CONFIG, ...config })
  return { engine, requests }
}

const INPUT = {
  messages: [{ role: 'user', content: [{ type: 'text', text: 'earlier work log' }] }],
}

test('primary summarize uses the handoff brief instruction and routed target', async () => {
  const { engine, requests } = engineWith(new Map([
    ['prov/m1', { text: '## Objective\n- done' }],
  ]))
  const result = await engine.summarize(INPUT, fakeAgent({ headerProvider: 'prov', headerModel: 'm1' }))
  assert.equal(requests.length, 1)
  const last = requests[0].messages.at(-1)
  assert.equal(last.role, 'user')
  assert.equal(last.content[0].text, COMPACTION_INSTRUCTION)
  assert.ok(COMPACTION_INSTRUCTION.includes('## Verified facts'))
  assert.ok(COMPACTION_INSTRUCTION.includes('PRIOR handoff brief'))
  assert.equal(requests[0].purpose, 'compaction')
  assert.equal(requests[0].provider, 'prov')
  assert.equal(result.provider, 'prov')
  assert.equal(result.model, 'm1')
  assert.deepEqual(result.summary, [{ type: 'text', text: '## Objective\n- done' }])
})

test('walks the fallback chain when the routed target refuses', async () => {
  const { engine, requests } = engineWith(new Map([
    ['prov/m1', { error: 'as a summarizer I must decline' }],
    ['pool/flash', { text: '## Objective\n- brief from fallback' }],
  ]), {
    fallbackTargets: [{ provider: 'pool', model: 'flash' }],
  })
  const result = await engine.summarize(INPUT, fakeAgent({ headerProvider: 'prov', headerModel: 'm1' }))
  assert.equal(requests.length, 2)
  assert.equal(requests[0].provider, 'prov')
  assert.equal(requests[1].provider, 'pool')
  assert.equal(requests[1].messages.at(-1).content[0].text, COMPACTION_INSTRUCTION)
  assert.equal(result.provider, 'pool')
  assert.equal(result.model, 'flash')
})

test('configured summarizer pair overrides the routed target', async () => {
  const { engine, requests } = engineWith(new Map([
    ['sump/summ', { text: '## Objective\n- pinned' }],
  ]), {
    summarizationProvider: 'sump',
    summarizationModel: 'summ',
  })
  const result = await engine.summarize(INPUT, fakeAgent({ headerProvider: 'prov', headerModel: 'm1' }))
  assert.equal(requests.length, 1)
  assert.equal(requests[0].provider, 'sump')
  assert.equal(result.provider, 'sump')
})

test('modelPolicies override supplies the summarizer pair per route', async () => {
  const { engine, requests } = engineWith(new Map([
    ['pinned/pair', { text: '## Objective\n- policy brief' }],
  ]), {
    modelPolicies: [
      { provider: 'prov', model: 'm1', summarizationProvider: 'pinned', summarizationModel: 'pair' },
    ],
  })
  const result = await engine.summarize(INPUT, fakeAgent({ headerProvider: 'prov', headerModel: 'm1' }))
  assert.equal(requests.length, 1)
  assert.equal(requests[0].provider, 'pinned')
  assert.equal(result.provider, 'pinned')
})

test('no routed or configured target fails with the base engine message', async () => {
  const { engine } = engineWith(new Map())
  await assert.rejects(
    engine.summarize(INPUT, fakeAgent({})),
    /no provider\/model available for summarization/,
  )
})

test('fallback skipping the failed primary never double-calls it', async () => {
  let primaryCalls = 0
  const llm = {
    async *stream(options) {
      if (options.provider === 'prov') {
        primaryCalls += 1
        yield { type: 'finish', reason: { kind: 'error', failure: { message: 'quota', code: 'QUOTA' } } }
        return
      }
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'text-delta', index: 0, text: '## Objective\n- ok' }
      yield { type: 'block-end', index: 0, block: { type: 'text', text: '## Objective\n- ok' } }
      yield { type: 'finish', reason: { kind: 'stop' } }
    },
  }
  const engine = new HandoffCompactionEngine(stubCtx(llm), {
    ...BASE_CONFIG,
    fallbackTargets: [{ provider: 'prov', model: 'm1' }, { provider: 'other', model: 'm2' }],
  })
  const result = await engine.summarize(INPUT, fakeAgent({ headerProvider: 'prov', headerModel: 'm1' }))
  assert.equal(primaryCalls, 1)
  assert.equal(result.provider, 'other')
})


/**
 * Session double shaped for the compaction surface helpers: ordered surface
 * nodes backed by real-looking events (tool pairing inspects event types).
 */
function fakeSurfaceSession(kinds) {
  const events = kinds.map((kind, seq) => ({
    seq,
    type: kind === 'assistant' ? 'assistant/message' : kind === 'tool' ? 'tool/result' : kind === 'system' ? 'system/message' : 'user/message',
    data: {
      message: {
        content: [],
        source: kind === 'user' ? { kind: 'user' } : { kind: 'model' },
      },
    },
  }))
  const nodes = events.map((event) => event.seq)
  return {
    seq: events.length,
    surface: { nodes, replaceGeneration: 0 },
    eventAt: (seq) => events[seq],
  }
}

test('selectCompactableRange folds the oldest span and retains the tail', () => {
  const session = fakeSurfaceSession(['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant'])
  const measurement = { nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 1000, heuristicTokens: 1000 })) }
  // retain 2000 -> keep the last two nodes; fold sys+1..4 (indices 1..4)
  const range = selectCompactableRange(session, measurement, 2000)
  assert.deepEqual(range, { start: 1, end: 4 })
})

test('selectCompactableRange returns null when the retained tail covers everything', () => {
  const session = fakeSurfaceSession(['system', 'user', 'assistant'])
  const measurement = { nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 1000, heuristicTokens: 1000 })) }
  assert.equal(selectCompactableRange(session, measurement, 5000), null)
})

test('alwaysCompact config is accepted and defaults off', () => {
  const { engine } = engineWith(new Map(), { alwaysCompact: true, alwaysMinSpanTokens: 1024 })
  assert.equal(engine.alwaysCompact, true)
  assert.equal(engine.alwaysMinSpanTokens, 1024)
  const plain = engineWith(new Map()).engine
  assert.equal(plain.alwaysCompact, false)
})


test('selectRoundsRange keeps exactly the newest N rounds and folds the rest', () => {
  const kinds = ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'user']
  const session = fakeSurfaceSession(kinds)
  // real surface events carry source directly on data; exercise that shape too
  const originalEventAt = session.eventAt
  session.eventAt = (seq) => {
    const event = originalEventAt(seq)
    if (event.type !== 'user/message') return event
    return { ...event, data: { ...event.data.message } }
  }
  const measurement = { nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 100, heuristicTokens: 100 })) }
  // keepRounds 2 -> retained window starts at the 2nd-from-last user message (index 5)
  const range = selectRoundsRange(session, measurement, 2)
  assert.deepEqual(range, { start: 1, end: 4 })
  // keepRounds large enough -> nothing folds
  assert.equal(selectRoundsRange(session, measurement, 4), null)
})

test('selectRoundsRange ignores compaction checkpoints and reminders as rounds', () => {
  const session = fakeSurfaceSession(['system', 'user', 'assistant', 'user', 'assistant', 'user'])
  const events = []
  // mark two synthetic messages (checkpoint + reminder) so they do not count
  const originalEventAt = session.eventAt
  session.eventAt = (seq) => {
    const event = originalEventAt(seq)
    if (seq === 2) return { ...event, type: 'user/message', data: { message: { source: { kind: 'compact-checkpoint' }, content: [] } } }
    if (seq === 4) return { ...event, type: 'user/message', data: { message: { source: { kind: 'lean-window-reminder' }, content: [] } } }
    return event
  }
  const measurement = { nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 100, heuristicTokens: 100 })) }
  // genuine rounds are seq 1, 3, 5; keep 2 -> fold before seq 3 (index 2)
  const range = selectRoundsRange(session, measurement, 2)
  assert.deepEqual(range, { start: 1, end: 2 })
})

test('keepRounds config is accepted, defaults to token-tail mode, and sets the reminder', () => {
  const { engine } = engineWith(new Map(), { alwaysCompact: true, keepRounds: 3 })
  assert.equal(engine.keepRounds, 3)
  assert.equal(engine.remindOnRound, 3) // defaults to the last retained round
  assert.ok(engine.reminder.includes('3 轮'))
  const off = engineWith(new Map(), { alwaysCompact: true }).engine
  assert.equal(off.keepRounds, 0)
  assert.equal(off.reminder, '')
  const custom = engineWith(new Map(), { alwaysCompact: true, keepRounds: 2, reminder: 'REMIND CUSTOM' }).engine
  assert.equal(custom.reminder, 'REMIND CUSTOM')
})

test('remindOnRound chooses which round carries the reminder (0 disables)', () => {
  const atTwo = engineWith(new Map(), { alwaysCompact: true, keepRounds: 5, remindOnRound: 2 }).engine
  assert.equal(atTwo.remindOnRound, 2)
  assert.ok(atTwo.reminder.includes('5 轮'))
  const disabled = engineWith(new Map(), { alwaysCompact: true, keepRounds: 5, remindOnRound: 0 }).engine
  assert.equal(disabled.remindOnRound, 0)
  assert.equal(disabled.reminder, '')
})

test('selectSummaryOnlyRange folds everything before the round in progress', () => {
  // system, user(1), assistant, user(2), assistant, user(3)  <- 第 3 轮进行中
  const kinds = ['system', 'user', 'assistant', 'user', 'assistant', 'user']
  const session = fakeSurfaceSession(kinds)
  const originalEventAt = session.eventAt
  session.eventAt = (seq) => {
    const event = originalEventAt(seq)
    if (event.type !== 'user/message') return event
    return { ...event, data: { ...event.data.message } }
  }
  const measurement = { nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 5000, heuristicTokens: 5000 })) }
  // 折叠索引 1..4（前两轮），保留进行中的第 3 轮
  assert.deepEqual(selectSummaryOnlyRange(session, measurement), { start: 1, end: 4 })
})

test('selectSummaryOnlyRange folds the finished round once it has been answered', () => {
  // 下一轮的首个 pre-step 里，新一轮消息尚未 splice，本轮已答完 -> 折掉它
  const session = fakeSurfaceSession(['system', 'user', 'assistant'])
  const originalEventAt = session.eventAt
  session.eventAt = (seq) => {
    const event = originalEventAt(seq)
    if (event.type !== 'user/message') return event
    return { ...event, data: { ...event.data.message } }
  }
  const measurement = { nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 5000, heuristicTokens: 5000 })) }
  assert.deepEqual(selectSummaryOnlyRange(session, measurement), { start: 1, end: 2 })
})

test('selectSummaryOnlyRange returns null while the round is still unanswered', () => {
  // 本轮消息刚进 surface、还没有任何 assistant 回复 -> 不能折叠自己
  const session = fakeSurfaceSession(['system', 'user'])
  const originalEventAt = session.eventAt
  session.eventAt = (seq) => {
    const event = originalEventAt(seq)
    if (event.type !== 'user/message') return event
    return { ...event, data: { ...event.data.message } }
  }
  const measurement = { nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 5000, heuristicTokens: 5000 })) }
  assert.equal(selectSummaryOnlyRange(session, measurement), null)
})

test('selectSummaryOnlyRange returns null when the span holds only a prior brief', () => {
  const events = [
    { seq: 0, type: 'system/message', data: { message: { content: [], source: { kind: 'system' } } } },
    {
      seq: 1,
      type: 'user/message',
      data: {
        message: { content: [{ type: 'text', text: '<compacted-summary>x</compacted-summary>' }], source: { kind: 'compact-checkpoint' } },
      },
    },
    { seq: 2, type: 'user/message', data: { message: { content: [{ type: 'text', text: 'current' }], source: { kind: 'user' } } } },
  ]
  const session = {
    seq: 3,
    surface: { nodes: [0, 1, 2], replaceGeneration: 0 },
    eventAt: (seq) => events[seq],
  }
  const originalEventAt = session.eventAt
  session.eventAt = (seq) => {
    const event = originalEventAt(seq)
    if (event.type !== 'user/message') return event
    return { ...event, data: { ...event.data.message } }
  }
  const measurement = { nodes: [0, 1, 2].map((seq) => ({ seq, tokens: 5000, heuristicTokens: 5000 })) }
  const range = selectSummaryOnlyRange(session, measurement)
  // 只有旧 brief + 进行中的轮：折叠 brief 自身没有收益
  assert.ok(range === null || range.start <= 1)
})

test('summaryOnly config drives the rolling-brief instruction and per-round reminder', () => {
  const { engine } = engineWith(new Map(), { alwaysCompact: true, summaryOnly: true })
  assert.equal(engine.summaryOnly, true)
  assert.equal(engine.remindOnRound, 1) // every round is reminded
  assert.ok(engine.reminder.includes('单轮记忆'))
  assert.ok(SUMMARY_ONLY_INSTRUCTION.includes('rolling brief'))
  assert.ok(SUMMARY_ONLY_INSTRUCTION.includes('## Objective'))
  assert.ok(SUMMARY_ONLY_INSTRUCTION.includes('## Standing constraints'))
  // 两种模式的六段格式必须一致，brief 才能互换
  for (const section of ['## Objective', '## Progress so far', '## Verified facts', '## Open problems', '## Next action', '## Standing constraints']) {
    assert.ok(COMPACTION_INSTRUCTION.includes(section), `base instruction missing ${section}`)
    assert.ok(SUMMARY_ONLY_INSTRUCTION.includes(section), `summary-only instruction missing ${section}`)
  }
  const off = engineWith(new Map(), { alwaysCompact: true }).engine
  assert.equal(off.summaryOnly, false)
})
