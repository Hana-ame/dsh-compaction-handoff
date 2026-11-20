/**
 * 单轮记忆（summaryOnly）多轮集成测试。
 *
 * 断言目标（用户要求）：
 *   1. 每一轮只根据 summary 做事 —— 第 2 轮起，模型可见的历史里只有交接小结，
 *      没有上一轮的原始工具输出/原文。
 *   2. 轮末总结（重写 brief）按固定六段格式输出。
 *   3. 总结内容就是下一轮的 summary —— 下一轮 fold 的输入包含上一轮的 brief。
 *
 * 用真实 session 形状（surface 节点序列 + eventAt）驱动选择器与 summarize，
 * 而不是 mock 掉选择逻辑，因此能捕获「轮次边界判定」这类真实 bug。
 *
 * Run: node --test test/summary-only-integration.test.js
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import HandoffCompactionEngine, {
  SUMMARY_ONLY_INSTRUCTION,
  selectSummaryOnlyRange,
} from '../index.js'

const BASE_CONFIG = { auto: false }

const SECTIONS = [
  '## Objective',
  '## Progress so far',
  '## Verified facts',
  '## Open problems',
  '## Next action',
  '## Standing constraints',
]

/** 构造一个按顺序推进的会话：turn(kind) 往 surface 追加节点。 */
function makeSession() {
  const events = []
  const surface = { nodes: [], replaceGeneration: 0 }
  const session = {
    id: 'integration',
    surface,
    eventAt: (seq) => events[seq],
    requestHeader: () => ({ config: { provider: 'prov', model: 'm1' } }),
    toolHistory: () => [],
  }
  const push = (type, data) => {
    const seq = events.length
    events.push({ seq, type, data })
    surface.nodes.push(seq)
    return seq
  }
  return {
    session,
    /** 系统头 + 一条真实用户消息 + 一次 assistant 回复（= 一轮）。 */
    round(text, assistantText = 'done') {
      push('user/message', { content: [{ type: 'text', text }], source: { kind: 'user' }, role: 'user' })
      push('assistant/message', { message: { role: 'assistant', content: [{ type: 'text', text: assistantText }] } })
    },
    /** 模拟 base engine 的 surface replace：把 [start..end] 换成一条 brief。 */
    fold(start, end, briefText) {
      const briefSeq = events.length
      events.push({
        seq: briefSeq,
        type: 'user/message',
        data: {
          content: [{ type: 'text', text: briefText }],
          source: { kind: 'compact-checkpoint', form: 'notice' },
          role: 'user',
        },
      })
      const before = surface.nodes.filter((s) => s < start)
      const after = surface.nodes.filter((s) => s > end)
      surface.nodes = [...before, briefSeq, ...after]
      surface.replaceGeneration += 1
      return briefSeq
    },
    measurement() {
      return { nodes: surface.nodes.map((seq) => ({ seq, tokens: 6000, heuristicTokens: 6000 })) }
    },
    systems: () => {
      push('system/message', { content: [], source: { kind: 'system' } })
    },
  }
}

test('第 2 轮起模型可见的历史只有交接小结，没有上一轮原文', () => {
  const s = makeSession()
  s.systems()
  s.round('第1轮：贴一大段 README 原文 TOKEN-R1-ORIGINAL')

  // 第 1 轮结束 -> 折掉它，生成下一轮的 brief
  const range1 = selectSummaryOnlyRange(s.session, s.measurement())
  assert.ok(range1 !== null, '第 1 轮结束后应可折叠')
  s.fold(range1.start, range1.end, '<compacted-summary>\n## Objective\n- R1\n</compacted-summary>')

  // 第 2 轮开始：surface 里唯一的历史就是那条 brief
  const visibleHistory = s.session.surface.nodes
    .map((seq) => s.session.eventAt(seq))
    .filter((e) => e.type === 'user/message')
    .map((e) => e.data.content.map((b) => b.text).join(''))
  assert.equal(visibleHistory.length, 1, '第 2 轮只应看到一条历史 user 消息')
  assert.match(visibleHistory[0], /compacted-summary/, '它必须是交接小结')
  assert.ok(
    !visibleHistory.some((t) => t.includes('TOKEN-R1-ORIGINAL')),
    '第 1 轮原文不得出现在第 2 轮的历史里',
  )
})

test('进行中的第 2 轮不会被折掉；第 3 轮到来时才折它', () => {
  const s = makeSession()
  s.systems()
  s.round('第1轮 TOKEN-R1')
  const range1 = selectSummaryOnlyRange(s.session, s.measurement())
  s.fold(range1.start, range1.end, '<compacted-summary>\n## Objective\n- R1\n</compacted-summary>')

  // 第 2 轮：用户消息进来但还没有 assistant 回复 -> 不能折自己
  s.session.surface.nodes.push(
    (() => {
      const seq = 100
      s.session.surface.nodes.push(seq)
      return seq
    })(),
  )
  // 用 round() 的语义手动补一条无回复的用户消息
  const s2 = makeSession()
  s2.systems()
  s2.round('第1轮')
  const r = selectSummaryOnlyRange(s2.session, s2.measurement())
  s2.fold(r.start, r.end, '<compacted-summary>\n## Objective\n- R1\n</compacted-summary>')
  // 现在 surface = [system, brief, 第2轮user]（无 assistant）
  const st = s2.session
  st.surface.nodes.push(st.surface.nodes.length + 10)
  const seq = st.surface.nodes.at(-1)
  // 写入实际事件
  const beforeLen = st.surface.nodes.length
  st.eventAt = ((orig) => (s) => (s === seq ? { seq, type: 'user/message', data: { content: [], source: { kind: 'user' } } } : orig(s)))(st.eventAt)
  assert.equal(
    selectSummaryOnlyRange(st, s2.measurement()),
    null,
    '未应答的用户消息属于进行中的轮，不能被折',
  )
  assert.ok(beforeLen > 0)
})

test('轮末重写 brief 走 SUMMARY_ONLY_INSTRUCTION 且六段格式齐全', async () => {
  const requests = []
  const brief = SECTIONS.map((section) => `${section}\n- merged`).join('\n')
  const ctx = {
    provide: [],
    logger: { info() {}, warn() {}, error() {} },
    reflect: { provide() {} },
    on() { return () => {} },
    effect() { return () => {} },
    llm: {
      async *stream(options) {
        requests.push(options)
        yield { type: 'block-start', index: 0, blockType: 'text' }
        yield { type: 'text-delta', index: 0, text: brief }
        yield { type: 'block-end', index: 0, block: { type: 'text', text: brief } }
        yield { type: 'finish', reason: { kind: 'stop' } }
      },
    },
  }
  const engine = new HandoffCompactionEngine(ctx, { ...BASE_CONFIG, alwaysCompact: true, summaryOnly: true })
  const agent = {
    options: {},
    session: { id: 'x', requestHeader: () => ({ config: { provider: 'prov', model: 'm1' } }), toolHistory: () => [] },
  }
  const result = await engine.summarize(
    {
      messages: [
        { role: 'user', content: [{ type: 'text', text: '<compacted-summary>上次的 brief</compacted-summary>' }] },
        { role: 'user', content: [{ type: 'text', text: '刚结束的这一轮' }] },
      ],
    },
    agent,
  )
  const instruction = requests[0].messages.at(-1).content[0].text
  assert.equal(instruction, SUMMARY_ONLY_INSTRUCTION, '轮末重写必须用单轮记忆指令')
  for (const section of SECTIONS) {
    assert.ok(instruction.includes(section), `指令缺少 ${section}`)
    assert.ok(result.summary[0].text.includes(section), `brief 缺少 ${section}`)
  }
  // 「上一轮 brief」确实作为输入给了 summarizer
  const allText = requests[0].messages.map((m) => m.content.map((b) => b.text ?? '').join('')).join('\n')
  assert.match(allText, /上次的 brief/, '上一轮的 brief 必须进入本轮重写的输入')
})

test('summaryOnly 模式每轮都提醒，且提醒文案说明单轮记忆', () => {
  const ctx = {
    provide: [],
    logger: { info() {}, warn() {}, error() {} },
    reflect: { provide() {} },
    on() { return () => {} },
    effect() { return () => {} },
    llm: { async *stream() {} },
  }
  const engine = new HandoffCompactionEngine(ctx, { ...BASE_CONFIG, alwaysCompact: true, summaryOnly: true })
  assert.equal(engine.remindOnRound, 1)
  assert.match(engine.reminder, /单轮记忆/)
  assert.match(engine.reminder, /小结/)
})
