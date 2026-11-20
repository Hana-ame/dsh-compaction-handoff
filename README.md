# dsh-compaction-handoff

BasicCompactionEngine 子类：把默认的通用 checkpoint 摘要换成**交接式工作状态小结**（handoff brief），并支持论文式**单动作压缩**（每一条消息 / 每一步都直接 compact）。

## 两种模式

| 模式 | 触发 | 用途 |
|---|---|---|
| 默认（阈值） | 上下文越过 `thresholdRatio/headroom` 折线才压 | 普通会话 |
| `alwaysCompact: true` | **每个 `agent/pre-step`**（每次模型调用前）都折叠保留尾以上的平衡历史，不看任何阈值 | 论文「单动作压缩」：上下文长期 ≈ brief + 保留尾 + 当前输入 |

`alwaysCompact` 的核心语义：**每一条消息之后都尝试 compact**——每次模型调用前把历史折成一份交接 brief。它不做阈值判断；能否真正折叠只受两条物理约束：

1. **可折叠跨度必须大于 brief 本身**（保留尾之外的内容 < `alwaysMinSpanTokens` 或小于摘要体积时跳过，否则折完反而更大）；
2. **工具调用/结果对不可拆**，一个很大的 tool/result 节点若落在保留尾内，本轮就没有可折叠的旧跨度。

所以典型对话是：第 1 轮无可折叠历史 → 第 2 轮起**每一轮都产生一次 `compaction/summary`**。实测 3 轮 = 2 次 summary（第 2、3 轮各一次）。

## 配置键

基类键全部保留（`thresholdRatio / headroomTokens / retainRatio / retainTokens / summarizationProvider / summarizationModel / maxTokens / compactionRetries / maxOverflowRetries / modelPolicies / auto`），另加：

| 键 | 默认 | 含义 |
|---|---|---|
| `alwaysCompact` | `false` | 每步强制折叠（单动作压缩） |
| `summaryOnly` | `false` | **单轮记忆**：每轮只带交接小结做事；轮末把「上轮小结 + 本轮」重写成下一轮小结（六段固定格式）。开启后 `keepRounds`/`retainTokens` 不生效 |
| `keepRounds` | `0` | **极限 N 轮**：只保留最近 N 轮用户消息的原文，更早历史在每步折叠；`0`=改用 token 尾（`retainTokens`） |
| `remindOnRound` | = `keepRounds` | **第 X 轮带提醒**：在该轮首步向模型注入一条窗口提醒；`0`=关闭 |
| `alwaysMinSpanTokens` | `1024` | 可折叠跨度低于该 token 数时跳过（摘要无法缩小它） |
| `retainTokens` | `4096`（建议） | `keepRounds=0` 时的逐字尾 token 预算 |
| `reminder` | 引擎默认 | 提醒文案；显式 `''` = 关闭提醒 |
| `fallbackTargets` | `[]` | 摘要器失败后的回退链（主摘要跟随对话路由保留 KV cache 复用） |

### 单轮记忆模式（summaryOnly，leancat 现行）

```yaml
config:
  alwaysCompact: true
  summaryOnly: true
```

行为：
- 模型**永远看不到历史原文**，唯一记忆是上一轮生成的交接小结。
- 每轮开始时把「上一轮小结 + 刚结束的那一轮」重写成新小结，作为下一轮的记忆。
- 格式固定六段：`## Objective` / `## Progress so far` / `## Verified facts` / `## Open problems` / `## Next action` / `## Standing constraints`。
- **进行中的那一轮逐字可见**，否则工具循环无法基于对自己的摘要工作；轮末才被折进小结。
- 每轮都注入窗口提醒（不按 `remindOnRound`），提醒模型「本轮结束小结会被重写」。
- 只折"已完成且有 assistant 回复的轮"：尚无人应答的用户消息属于进行中，不折。

### 极限 N 轮 + 第 X 轮提醒

```yaml
config:
  alwaysCompact: true
  keepRounds: 3        # 只留最近 3 轮原文
  remindOnRound: 3     # 第 3 轮注入窗口提醒（0=关；不填=keepRounds）
```

- `keepRounds` 用「用户轮次」计数，只统计真实用户消息；交接 checkpoint 与引擎自己注入的提醒不计为轮次。
- 轮次边界不拆 tool-call/result 对：必要时少折一点，不会产生不平衡历史。
- 提醒以普通 user 消息写入会话日志（source `lean-window-reminder`），模型在该轮首步看到，轨迹里可审阅。

单动作模式的 `retainTokens` 是取舍旋钮：保留尾越大，近期细节保留越多，但要等更长的旧历史才够折叠；leancat 用 `4096`（512 实测太容易丢近期上下文），brief 承载更早的跨轮状态。

```yaml
- id: compaction-basic
  name: '../../plugins/dsh-compaction-handoff/index.js'
  config:
    alwaysCompact: true
    alwaysMinSpanTokens: 1024
    retainTokens: 4096
    fallbackTargets:
      - provider: sensenova-pool
        model: deepseek-flash
```

## 测试

```
node --test test/engine.test.js
```

9 个用例：指令模板、路由目标、fallback 链、覆盖优先级、无目标报错、fallback 去重、`selectCompactableRange` 折叠/保留/无可折、`alwaysCompact` 配置解析。

## 边界

- `alwaysCompact` 每步一次摘要调用 → 成本换上下文；极短对话首轮不折（见上）。
- 交接 brief 的模板与 `<compacted-summary>` 合并规则不变：每次再压缩把先前的 brief 合并成单一更新版本。
