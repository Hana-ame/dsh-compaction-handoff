/**
 * Activation smoke test: load the REAL web profile through dsh-app-boot's
 * own loader, find the composed leancat preset's compaction group, and
 * activate the engine plugin against a minimal cordis-style context carrying
 * the services the engine injects.
 *
 * Verifies, end to end against the real composed tree:
 *   1. the profile parses with the loader's own YAML dialect (!!js included);
 *   2. the leancat preset row composes (id/name/plugins present);
 *   3. the plugin file imports cleanly;
 *   4. its config validates the composed leancat config (thresholdRatio /
 *      modelPolicies / fallbackTargets);
 *   5. a compactNow on an empty session returns null (base-engine behavior),
 *      not an import/type error.
 *
 * Run: node test/activation.smoke.mjs
 */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

const require = createRequire('/home/lumin/.dsh/profiles/web/package.json')
const appBoot = require('@deepseek-ai/dsh-app-boot')

// 1. Load the real profile (same entry points dsh itself uses): bundle
//    layers plus the user patch layer, composed exactly as boot does.
//    installAnchor must be a file URL (the running launcher passes its own).
const profile = appBoot.loadProfile('dsh', 'web', require.resolve('@deepseek-ai/dsh-app-boot/package.json'))
assert.equal(profile.skippedBundles.length, 0, `no bundle skipped: ${JSON.stringify(profile.skippedBundles)}`)
const allPatches = [...profile.layers.map((layer) => layer.patches).flat(), ...profile.patches]
const entries = appBoot.composeEntries([allPatches])
assert.ok(Array.isArray(entries), 'profile entries compose to a list')

// 2. Locate the composed leancat preset row.
const leancat = entries.find((row) => row?.id === 'preset-leancat')
assert.ok(leancat, 'preset-leancat row present in the composed tree')
assert.equal(leancat.name, '@deepseek-ai/dsh-agent-preset')
assert.equal(leancat.config.id, 'leancat')
const plugins = leancat.config.plugins
assert.ok(Array.isArray(plugins) && plugins.length >= 10, `leancat plugins list (${plugins.length} rows)`)
const compactionGroup = plugins.find((row) => row?.id === 'compaction')
assert.ok(compactionGroup?.group === true, 'compaction group present')
const engineRow = compactionGroup.config.find((row) => row.id === 'compaction-basic')
assert.ok(engineRow, 'compaction-basic row present')
assert.equal(engineRow.name, '../../plugins/dsh-compaction-handoff/index.js')
console.log(`leancat composed: ${plugins.length} plugin rows; compaction-basic -> ${engineRow.name}`)
console.log('composed compaction config:', JSON.stringify(engineRow.config))

// 3. Minimal cordis-style context with the three injected services.
const services = new Map()
const ctx = {
  provide: [],
  reflect: { provide() {}, get: (name) => services.get(name) },
  get: (name) => services.get(name),
  on() { return () => {} },
  off() {},
  effect(fn) { const dispose = fn?.(); return () => dispose?.() },
  logger: { info() {}, warn() {}, error() {} },
  waterfall(event, payload, next) { return next() },
}
const tokenMeter = {
  measure() { return { totalTokens: 0, nodes: [] } },
  estimateMessage() { return 1 },
}
services.set('llm', {
  async *stream() {
    yield { type: 'finish', reason: { kind: 'error', failure: { message: 'smoke: no llm behind', code: 'SERVER' } } }
  },
  async resolveModelInfo() {
    return { context: { contextWindow: 131072 }, defaultMaxTokens: 8192 }
  },
})
services.set('tokenMeter', tokenMeter)
services.set('sessions', { async flush() {} })
// The engine reads `this.ctx.tokenMeter` as a property (cordis exposes
// injected services as properties); mirror that on the stub context.
Object.defineProperty(ctx, 'tokenMeter', { get: () => tokenMeter })
Object.defineProperty(ctx, 'llm', { get: () => services.get('llm') })
Object.defineProperty(ctx, 'sessions', { get: () => services.get('sessions') })

// 4. Import and activate with the composed config.
const { default: HandoffCompactionEngine } = await import('/home/lumin/.dsh/plugins/dsh-compaction-handoff/index.js')
const engine = new HandoffCompactionEngine(ctx, engineRow.config)
services.set('compaction', engine)

// Config resolution mirrors load-time resolveConfig: every composed value
// passes validation and the modelPolicies resolve with exact overrides.
assert.equal(engine.config.thresholdRatio, 0.125)
assert.equal(engine.config.retainTokens, 2048)
assert.equal(engine.config.modelPolicies.length, 3)
const glmPolicy = engine.config.modelPolicies.find((p) => p.model === 'glm-5.3@轮换')
assert.ok(glmPolicy, 'glm-5.3@轮换 policy resolved')
assert.equal(glmPolicy.thresholdRatio, 0.125)
assert.equal(glmPolicy.retainTokens, 2048)
assert.equal(engine.fallbackTargets.length, 2)
assert.equal(engine.fallbackTargets[0].provider, 'sensenova-pool')

// 5. compactNow on an empty surface: base-engine "no compactable range" -> null.
const emptySession = {
  id: 'smoke',
  surface: { nodes: [], replaceGeneration: 0 },
  seq: 0,
  eventAt() { throw new Error('no events') },
  requestHeader() { return undefined },
  toolHistory() { return [] },
}
const emptyAgent = { session: emptySession, options: {}, runMaintenance: async (fn) => fn(new AbortController().signal) }
const result = await engine.compactNow(emptyAgent, new AbortController().signal)
assert.equal(result, null)

console.log('ACTIVATION SMOKE OK: real profile composes; engine mounts as ctx.compaction with the composed leancat config')
