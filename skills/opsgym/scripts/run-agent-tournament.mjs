#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { adapterRefFromArena, loadAdapter } from "./lib/adapter-loader.mjs";
import {
  AGENT_SCHEMA,
  PARAMETER_DESCRIPTIONS,
  clampInteger,
  environmentBrief,
  assertValidAgentPlans,
  normalizePlans,
  readAgentPlansFile,
  toPolicyMap
} from "./lib/agents.mjs";
import { createRunContract } from "./lib/contracts.mjs";
import { appendProgress, configPathFromArgs, readProjectConfigMaybe } from "./lib/config.mjs";
import { loadEnvironment, parseArgs } from "./lib/workspace.mjs";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(metric, value) {
  if (value === null || value === undefined) return "-";
  if (metric.unit === "inr") return `₹${Math.round(value).toLocaleString("en-IN")}`;
  if (metric.unit === "ratio" || metric.unit === "percent") return `${Math.round(Number(value) * 100)}%`;
  return Number(value).toLocaleString("en-IN");
}

function metricList(env, scoreboard) {
  const catalog = new Map((env.metrics?.catalog || []).map((m) => [m.id, m]));
  const ids = [env.metrics?.primary, ...(env.metrics?.raw || [])].filter(Boolean);
  const seen = new Set();
  const out = [];
  for (const id of ids) {
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(catalog.get(id) ?? { id, label: id, unit: "count" });
  }
  for (const id of Object.keys(scoreboard[0]?.averages || {})) {
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(catalog.get(id) ?? { id, label: id, unit: "count" });
  }
  return out;
}

function responseText(payload) {
  if (payload.output_text) return payload.output_text;
  const parts = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (content.type === "output_text" && content.text) parts.push(content.text);
      if (content.type === "text" && content.text) parts.push(content.text);
    }
  }
  return parts.join("\n");
}

// ---------------------------------------------------------------------------
// Offline fallback agents (generic archetypes for any domain)
// ---------------------------------------------------------------------------

function offlinePlans(count, environment) {
  // Build agent archetypes at runtime from the environment.
  // Names, theses, and parameter spreads are derived from
  // whatever the adapter put into the environment — no hardcoded domain words.
  const domain = environment?.arena || environment?.adapter?.domain || "operations";
  const shocks = environment?.shocks || [];
  const entities = environment?.simulationEntities || environment?.entities || [];
  const fragile = entities.filter((e) => e.fragility > 0.5).map((e) => e.name);
  const reliable = entities.filter((e) => e.reliability > 0.8).map((e) => e.name);
  const shockLabel = shocks.length ? shocks.map((s) => s.label || s.type).join(", ") : "demand volatility";
  const fragileLabel = fragile.length ? fragile.join(", ") : "high-fragility entities";
  const reliableLabel = reliable.length ? reliable.join(", ") : "reliable entities";

  // Four archetypal strategies — spread across the parameter space
  const archetypes = [
    {
      id: "agent-protect",
      name: "Protect",
      thesis: `Shield ${fragileLabel} from ${shockLabel}. Deploy strong fallback recovery and prioritize the most vulnerable entities. Accept lower peak throughput to keep risk and backlog near zero.`,
      parameters: { capacityAggression: 0.38, riskTolerance: 0.25, executionAggression: 0.78, fallbackRecovery: 0.94, priorityFocus: 0.92 }
    },
    {
      id: "agent-push",
      name: "Push",
      thesis: `Maximize throughput by pushing capacity and execution hard across all entities, especially ${reliableLabel}. Accept elevated risk and cost as the price of higher output.`,
      parameters: { capacityAggression: 0.90, riskTolerance: 0.84, executionAggression: 1.12, fallbackRecovery: 0.50, priorityFocus: 0.44 }
    },
    {
      id: "agent-adapt",
      name: "Adapt",
      thesis: `Balance capacity between high-priority and fragile entities. Absorb ${shockLabel} with moderate fallback without overspending on recovery.`,
      parameters: { capacityAggression: 0.56, riskTolerance: 0.50, executionAggression: 0.94, fallbackRecovery: 0.76, priorityFocus: 0.74 }
    },
    {
      id: "agent-conserve",
      name: "Conserve",
      thesis: `Preserve capacity and minimize exposure. Accept lower throughput to keep operating cost and downside risk as low as possible, even if ${fragileLabel} occasionally miss demand.`,
      parameters: { capacityAggression: 0.30, riskTolerance: 0.16, executionAggression: 0.70, fallbackRecovery: 0.86, priorityFocus: 0.62 }
    }
  ];

  return archetypes.slice(0, count);
}

// ---------------------------------------------------------------------------
// OpenAI agent generation
// ---------------------------------------------------------------------------

async function callOpenAI({ model, environment, count }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("OPENAI_API_KEY is required for LLM agents. Use --offline for smoke tests.");

  const prompt = [
    "Create LLM decision agents for this OpsGym arena.",
    "Each agent should be meaningfully different and encoded as simulator parameters.",
    "Parameter meanings:",
    ...PARAMETER_DESCRIPTIONS.map(([name, desc]) => `- ${name}: ${desc}`),
    `Return exactly ${count} agents.`,
    "",
    JSON.stringify(environmentBrief(environment), null, 2)
  ].join("\n");

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      input: [
        { role: "system", content: "You design operational decision agents for simulation. Return only schema-valid JSON." },
        { role: "user", content: prompt }
      ],
      text: { format: { type: "json_schema", name: "opsgym_agent_plans", strict: true, schema: AGENT_SCHEMA } }
    })
  });

  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || `OpenAI API failed: ${response.status}`);
  const text = responseText(payload);
  if (!text) throw new Error("OpenAI response did not contain output_text.");
  return JSON.parse(text).agents;
}

// ---------------------------------------------------------------------------
// Report rendering (fully driven by environment metric catalog)
// ---------------------------------------------------------------------------

function renderMemo({ runId, model, source, agentPlans, scoreboard, wins, reportPath, metrics }) {
  const winner = scoreboard[0];
  return [
    `# OpsGym Agent Tournament: ${runId}`,
    "",
    `Agent source: ${source}`,
    `Model: ${model}`,
    "",
    `Winner: ${winner.policyName}`,
    ...metrics.map((m) => `- ${m.label}: ${fmt(m, winner.averages[m.id])}`),
    `- Rollout wins: ${wins[winner.policy] || 0}`,
    "",
    "Agents:",
    ...agentPlans.map((a) => `- ${a.name}: ${a.description}`),
    "",
    "Leaderboard:",
    ...scoreboard.map((row, i) => `${i + 1}. ${row.policyName}: ${fmt(metrics[0], row.averages[metrics[0]?.id])}`),
    "",
    `Report: ${reportPath}`
  ].join("\n");
}

function renderReport({ runId, model, source, agentPlans, scoreboard, wins, rollouts, environment, metrics }) {
  const winner = scoreboard[0];
  const runnerUp = scoreboard[1];
  const total = rollouts || "?";
  const question = environment?.question || runId;
  const primary = metrics[0];

  // Leaderboard rows — columns come from the metric catalog
  const headerCells = metrics.map((m) => `<th>${esc(m.label)}</th>`).join("");
  const rows = scoreboard.map((row, i) => {
    const winPct = total !== "?" ? `${Math.round(((wins[row.policy] || 0) / total) * 100)}%` : `${wins[row.policy] || 0}`;
    const cells = metrics.map((m) => `<td>${fmt(m, row.averages[m.id])}</td>`).join("");
    return `<tr${i === 0 ? ' class="wr"' : ''}><td>${i + 1}</td><td><strong>${esc(row.policyName)}</strong><br><span>${esc(row.description)}</span></td>${cells}<td>${winPct}</td></tr>`;
  }).join("");

  // Agent strategy cards
  const cards = agentPlans.map((agent) => {
    const rank = scoreboard.findIndex((r) => r.policy === agent.id);
    const badge = rank === 0 ? '<span class="b w">Winner</span>' : rank >= 0 ? `<span class="b">#${rank + 1}</span>` : "";
    const bars = Object.entries(agent.parameters || {}).map(([k, v]) => {
      const pct = Math.min(Math.round(Number(v) * 100), 100);
      return `<div class="pr"><span class="pl">${k.replace(/([A-Z])/g, " $1").trim()}</span><div class="pb"><div class="pf" style="width:${pct}%"></div></div><span class="pv">${Number(v).toFixed(2)}</span></div>`;
    }).join("");
    return `<article${rank === 0 ? ' class="wc"' : ""}><div class="ch"><h3>${esc(agent.name)}</h3>${badge}</div><p class="th">${esc(agent.description)}</p><div class="ps">${bars}</div></article>`;
  }).join("");

  // Winner explanation — generic, driven by primary metric and runner-up delta
  const margin = winner && runnerUp ? fmt(primary, winner.averages[primary.id] - runnerUp.averages[primary.id]) : "N/A";
  const explanation = runnerUp
    ? `${esc(winner.policyName)} scored ${fmt(primary, winner.averages[primary.id])} on ${esc(primary.label)}, winning ${wins[winner.policy] || 0} of ${total} rollouts — ${margin} ahead of runner-up ${esc(runnerUp.policyName)}.`
    : `${esc(winner.policyName)} scored ${fmt(primary, winner.averages[primary.id])} on ${esc(primary.label)}, winning ${wins[winner.policy] || 0} of ${total} rollouts.`;

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpsGym Agent Tournament — ${esc(runId)}</title>
<style>
:root{color-scheme:light;--ink:#17211f;--mu:#63706d;--ln:#dbe4df;--bg:#f7faf8;--w:#fff;--ac:#0f766e}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--ink)}
main{width:min(1140px,calc(100% - 32px));margin:0 auto;padding:32px 0 48px}
header{display:grid;gap:16px;grid-template-columns:1.3fr .7fr;align-items:end;border-bottom:1px solid var(--ln);padding-bottom:24px}
h1{font-size:clamp(26px,4.5vw,48px);line-height:1.05;margin:0}
h2{font-size:20px;margin:0 0 12px}
p{color:var(--mu);line-height:1.55;margin:0}
.hero{background:var(--ink);color:#fff;padding:18px;border-radius:8px}
.hero p{color:#d7e3df} .hero strong{color:#8ee4d4;display:block;font-size:20px;margin-top:4px}
.hero .sub{color:#a8d8cf;font-size:13px;margin-top:6px}
.tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:4px;background:#edf5f2;color:var(--ac);font-weight:600;margin-top:8px}
.memo{border-left:5px solid var(--ac);background:var(--w);padding:18px;border-radius:8px;margin-top:24px}
.memo p{color:var(--ink)}
section{margin-top:28px}
table{width:100%;border-collapse:collapse;background:var(--w);border:1px solid var(--ln);border-radius:8px;overflow:hidden}
th,td{padding:12px 10px;text-align:left;border-bottom:1px solid var(--ln);vertical-align:top}
th{font-size:11px;color:var(--mu);text-transform:uppercase;background:#edf5f2}
td span{color:var(--mu);font-size:13px}
.wr{background:#f0fdf9}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
article{background:var(--w);border:1px solid var(--ln);border-radius:8px;padding:16px}
.wc{border-color:var(--ac);border-width:2px}
.ch{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
h3{margin:0;font-size:16px}
.b{font-size:11px;padding:3px 8px;border-radius:4px;background:#edf5f2;color:var(--mu);font-weight:600}
.b.w{background:var(--ac);color:#fff}
.th{font-size:13px;line-height:1.5;margin:0 0 12px;color:var(--ink)}
.ps{display:grid;gap:5px}
.pr{display:grid;grid-template-columns:130px 1fr 40px;align-items:center;gap:8px;font-size:12px}
.pl{color:var(--mu);text-transform:capitalize}
.pb{height:6px;background:#edf5f2;border-radius:3px;overflow:hidden}
.pf{height:100%;background:var(--ac);border-radius:3px}
.pv{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
@media(max-width:860px){header{grid-template-columns:1fr}.grid{grid-template-columns:1fr}table{display:block;overflow-x:auto}}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <p>OpsGym Agent Tournament</p>
      <h1>${esc(question)}</h1>
      <span class="tag">${esc(source)} / ${esc(model)}</span>
    </div>
    <aside class="hero">
      <p>Winning agent</p>
      <strong>${esc(winner.policyName)}</strong>
      <p>${esc(winner.description)}</p>
      <p class="sub">${fmt(primary, winner.averages[primary.id])} ${esc(primary.label)} | Won ${wins[winner.policy] || 0}/${total} rollouts</p>
    </aside>
  </header>

  <section class="memo">
    <h2>Result</h2>
    <p>${explanation}</p>
  </section>

  <section>
    <h2>Leaderboard</h2>
    <table>
      <thead><tr><th>#</th><th>Agent</th>${headerCells}<th>Win Rate</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Agent Strategies</h2>
    <div class="grid">${cards}</div>
  </section>
</main>
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const configPath = configPathFromArgs(args);
  const config = await readProjectConfigMaybe(configPath);
  const { resolveArenaId } = await import("./lib/adapter-loader.mjs");
  const arena = await resolveArenaId(args.arena || config?.arenaId);
  const workspace = args.workspace || config?.workspace || ".ops-gym";
  const runId = args.run || `${config?.project || "agent"}-agents`;
  const rollouts = clampInteger(args.rollouts || config?.rollouts || 50, 1, 10000);
  const seed = args.seed || `${runId}-seed`;
  const model = args.model || process.env.OPSGYM_AGENT_MODEL || (args["agents-file"] ? "codex" : "gpt-4.1-mini");
  const agentCount = clampInteger(args.agents || 3, 1, 6);
  const source = args["agents-file"] ? "codex-file" : (args.offline ? "offline-fixture" : "openai");

  const envPath = resolve(workspace, "environments", arena, "environment.json");
  const environment = await loadEnvironment(workspace, arena);
  const adapter = await loadAdapter(adapterRefFromArena({ arenaId: arena, adapter: environment.adapter }));

  const rawPlans = args["agents-file"]
    ? await readAgentPlansFile(args["agents-file"], agentCount)
    : args.offline
      ? offlinePlans(agentCount, environment)
      : await callOpenAI({ model, environment, count: agentCount });
  assertValidAgentPlans(rawPlans, { strict: Boolean(args["agents-file"]) });
  const agentPlans = normalizePlans(rawPlans, source, model);
  const policies = toPolicyMap(agentPlans);

  const allRollouts = [];
  for (let i = 0; i < rollouts; i += 1) {
    for (const [policyKey, policyConfig] of Object.entries(policies)) {
      allRollouts.push(adapter.runPolicy({ environment, policyKey, policyConfig, rolloutIndex: i, seed }));
    }
  }

  const { scoreboard, wins } = adapter.summarizeScoreboard({ allRollouts, policies, environment });
  const metrics = metricList(environment, scoreboard);

  const runDir = resolve(workspace, "runs", runId);
  const reportDir = resolve(workspace, "reports");
  const reportPath = resolve(reportDir, `${runId}.html`);
  await mkdir(runDir, { recursive: true });
  await mkdir(reportDir, { recursive: true });

  const runContract = createRunContract({
    arenaId: arena,
    adapter: environment.adapter ?? { id: arena, type: "arena-adapter" },
    question: environment.question,
    runId, rollouts, seed, workspace,
    environmentPath: envPath,
    scoreboard, wins, reportPath
  });
  runContract.mode = "agent-tournament";
  runContract.agentSource = source;
  runContract.agentModel = model;
  runContract.agentPlans = agentPlans;

  const compact = allRollouts.map((r) => ({ policy: r.policy, policyName: r.policyName, rolloutIndex: r.rolloutIndex, metrics: r.metrics }));

  await writeFile(resolve(runDir, "run.json"), `${JSON.stringify(runContract, null, 2)}\n`);
  await writeFile(resolve(runDir, "agents.json"), `${JSON.stringify({ source, model, agents: agentPlans }, null, 2)}\n`);
  await writeFile(resolve(runDir, "rollouts.json"), `${JSON.stringify(compact, null, 2)}\n`);
  await writeFile(resolve(runDir, "scores.json"), `${JSON.stringify({ runId, arena, mode: "agent-tournament", scoreboard, wins }, null, 2)}\n`);
  await writeFile(resolve(runDir, "decision-memo.md"), `${renderMemo({ runId, model, source, agentPlans, scoreboard, wins, reportPath, metrics })}\n`);
  await writeFile(reportPath, renderReport({ runId, model, source, agentPlans, scoreboard, wins, rollouts, environment, metrics }));

  if (config) await appendProgress(config, `ran agent tournament ${runId} with ${rollouts} rollouts`);

  const winner = scoreboard[0];
  console.log(`Winner: ${winner.policyName} (${fmt(metrics[0], winner.averages[metrics[0]?.id])} ${metrics[0]?.label})`);
  console.log(`Agent source: ${source}`);
  console.log(`Report: ${reportPath}`);
  console.log(`Run directory: ${runDir}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
