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

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function offlinePlans(count, environment) {
  const shockCount = (environment?.shocks || []).length;
  const hasShocks = shockCount > 0;
  const plans = [
    {
      id: "agent-resilience",
      name: "Resilience Agent",
      thesis: `Protect service level during ${hasShocks ? 'shocks' : 'volatility'} by deploying strong fallback recovery and focusing capacity on high-priority entities. Accept moderate throughput to keep risk and backlog low.`,
      parameters: { capacityAggression: 0.42, riskTolerance: 0.28, executionAggression: 0.82, fallbackRecovery: 0.92, priorityFocus: 0.88 }
    },
    {
      id: "agent-throughput",
      name: "Throughput Agent",
      thesis: "Maximize throughput by pushing execution aggression and capacity deployment hard. Accept elevated risk and operating cost as the price of higher output across all entities.",
      parameters: { capacityAggression: 0.88, riskTolerance: 0.82, executionAggression: 1.12, fallbackRecovery: 0.55, priorityFocus: 0.48 }
    },
    {
      id: "agent-balanced",
      name: "Balanced Agent",
      thesis: `Maintain a balanced posture: moderate capacity deployment, selective priority focus on fragile entities, and enough fallback to absorb ${hasShocks ? 'shock spillover' : 'demand spikes'} without overspending.`,
      parameters: { capacityAggression: 0.58, riskTolerance: 0.52, executionAggression: 0.95, fallbackRecovery: 0.78, priorityFocus: 0.72 }
    },
    {
      id: "agent-conservative",
      name: "Conservative Agent",
      thesis: "Preserve capacity and minimize risk exposure. Accept lower throughput and service level to keep operating cost and downside risk as low as possible. Strong fallback to cover any missed demand safely.",
      parameters: { capacityAggression: 0.32, riskTolerance: 0.18, executionAggression: 0.72, fallbackRecovery: 0.88, priorityFocus: 0.65 }
    }
  ];
  return plans.slice(0, count);
}

async function callOpenAI({ model, environment, count }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("OPENAI_API_KEY is required for real LLM agents. Use --offline only for smoke tests.");

  const prompt = [
    "Create LLM decision agents for this OpsGym arena.",
    "Each agent should be meaningfully different and encoded as simulator parameters.",
    "Parameter meanings:",
    ...PARAMETER_DESCRIPTIONS.map(([name, description]) => `- ${name}: ${description}`),
    `Return exactly ${count} agents.`,
    "",
    JSON.stringify(environmentBrief(environment), null, 2)
  ].join("\n");

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      input: [
        {
          role: "system",
          content: "You design operational decision agents for simulation. Return only schema-valid JSON."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      text: {
        format: {
          type: "json_schema",
          name: "opsgym_agent_plans",
          strict: true,
          schema: AGENT_SCHEMA
        }
      }
    })
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error?.message || `OpenAI API request failed with status ${response.status}`);
  }
  const text = responseText(payload);
  if (!text) throw new Error("OpenAI response did not contain output_text.");
  return JSON.parse(text).agents;
}

function compactRollouts(allRollouts) {
  return allRollouts.map((rollout) => ({
    policy: rollout.policy,
    policyName: rollout.policyName,
    rolloutIndex: rollout.rolloutIndex,
    metrics: rollout.metrics
  }));
}

function renderAgentMemo({ runId, model, source, agentPlans, scoreboard, wins, reportPath }) {
  const winner = scoreboard[0];
  return [
    `# OpsGym Agent Tournament: ${runId}`,
    "",
    `Agent source: ${source}`,
    `Model: ${model}`,
    "",
    `Winner: ${winner.policyName}`,
    `OpsScore: ${winner.averages.opsScore?.toLocaleString("en-IN")}`,
    `Rollout wins: ${wins[winner.policy] || 0}`,
    "",
    "Agents:",
    ...agentPlans.map((agent) => `- ${agent.name}: ${agent.description}`),
    "",
    "Leaderboard:",
    ...scoreboard.map((row, index) => `${index + 1}. ${row.policyName}: ${row.averages.opsScore?.toLocaleString("en-IN")} OpsScore`),
    "",
    `Report: ${reportPath}`
  ].join("\n");
}

function renderAgentReport({ runId, model, source, agentPlans, scoreboard, wins, rollouts, environment }) {
  const winner = scoreboard[0];
  const runnerUp = scoreboard[1];
  const totalRollouts = rollouts || "?";
  const question = environment?.question || "";

  const rows = scoreboard.map((row, index) => {
    const service = Number.isFinite(row.averages.serviceLevel) ? `${Math.round(row.averages.serviceLevel * 100)}%` : "-";
    const throughput = row.averages.throughput?.toLocaleString("en-IN") ?? "-";
    const risk = row.averages.risk?.toLocaleString("en-IN") ?? "-";
    const cost = row.averages.operatingCost?.toLocaleString("en-IN") ?? "-";
    const winPct = totalRollouts !== "?" ? `${Math.round(((wins[row.policy] || 0) / totalRollouts) * 100)}%` : `${wins[row.policy] || 0}`;
    return `<tr${index === 0 ? ' class="winner-row"' : ''}><td>${index + 1}</td><td><strong>${escapeHtml(row.policyName)}</strong><br><span>${escapeHtml(row.description)}</span></td><td>${row.averages.opsScore?.toLocaleString("en-IN")}</td><td>${throughput}</td><td>${service}</td><td>${risk}</td><td>${cost}</td><td>${winPct}</td></tr>`;
  }).join("");

  const agentCards = agentPlans.map((agent, index) => {
    const rank = scoreboard.findIndex((row) => row.policy === agent.id);
    const rankLabel = rank === 0 ? '<span class="badge winner">Winner</span>' : rank >= 0 ? `<span class="badge">Rank ${rank + 1}</span>` : '';
    return `<article${rank === 0 ? ' class="winner-card"' : ''}>
    <div class="card-header"><h3>${escapeHtml(agent.name)}</h3>${rankLabel}</div>
    <p class="thesis">${escapeHtml(agent.description)}</p>
    <div class="params">
      ${Object.entries(agent.parameters || {}).map(([key, value]) => {
        const pct = Math.round(Number(value) * 100);
        return `<div class="param"><span class="param-label">${key.replace(/([A-Z])/g, ' $1').trim()}</span><div class="param-bar"><div class="param-fill" style="width:${Math.min(pct, 100)}%"></div></div><span class="param-value">${Number(value).toFixed(2)}</span></div>`;
      }).join("")}
    </div>
  </article>`;
  }).join("");

  const winnerMargin = winner && runnerUp ? (winner.averages.opsScore - runnerUp.averages.opsScore).toLocaleString("en-IN") : "N/A";

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpsGym Agent Tournament - ${runId}</title>
  <style>
    :root { color-scheme: light; --ink: #17211f; --muted: #63706d; --line: #dbe4df; --paper: #f7faf8; --white: #ffffff; --accent: #0f766e; --gold: #b7791f; --red: #b42318; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--paper); color: var(--ink); }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 48px; }
    header { display: grid; gap: 16px; grid-template-columns: 1.3fr 0.7fr; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 24px; }
    h1 { font-size: clamp(28px, 4.5vw, 50px); line-height: 1.05; margin: 0; max-width: 850px; }
    h2 { font-size: 22px; margin: 0 0 14px; }
    p { color: var(--muted); line-height: 1.55; margin: 0; }
    .hero { background: var(--ink); color: white; padding: 18px; border-radius: 8px; }
    .hero p { color: #d7e3df; }
    .hero strong { color: #8ee4d4; display: block; font-size: 22px; margin-top: 4px; }
    .hero .margin { color: #a8d8cf; font-size: 14px; margin-top: 6px; }
    .memo { border-left: 5px solid var(--accent); background: var(--white); padding: 18px; border-radius: 8px; margin-top: 24px; }
    .memo p { color: var(--ink); }
    section { margin-top: 28px; }
    table { width: 100%; border-collapse: collapse; background: var(--white); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    th, td { padding: 13px 10px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; background: #edf5f2; }
    td span { color: var(--muted); font-size: 13px; }
    .winner-row { background: #f0fdf9; }
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
    article { background: var(--white); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .winner-card { border-color: var(--accent); border-width: 2px; }
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    h3 { margin: 0; font-size: 17px; }
    .badge { font-size: 11px; padding: 3px 8px; border-radius: 4px; background: #edf5f2; color: var(--muted); font-weight: 600; }
    .badge.winner { background: var(--accent); color: white; }
    .thesis { font-size: 14px; line-height: 1.5; margin: 0 0 12px; color: var(--ink); }
    .params { display: grid; gap: 6px; }
    .param { display: grid; grid-template-columns: 140px 1fr 44px; align-items: center; gap: 8px; font-size: 12px; }
    .param-label { color: var(--muted); text-transform: capitalize; }
    .param-bar { height: 6px; background: #edf5f2; border-radius: 3px; overflow: hidden; }
    .param-fill { height: 100%; background: var(--accent); border-radius: 3px; }
    .param-value { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
    .source-tag { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 4px; background: #edf5f2; color: var(--accent); font-weight: 600; margin-top: 8px; }
    pre { overflow: auto; background: #edf5f2; border-radius: 8px; padding: 10px; font-size: 12px; }
    @media (max-width: 860px) { header { grid-template-columns: 1fr; } .grid { grid-template-columns: 1fr; } table { display: block; overflow-x: auto; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p>OpsGym Agent Tournament</p>
        <h1>${question ? escapeHtml(question) : escapeHtml(runId)}</h1>
        <span class="source-tag">${escapeHtml(source)} / ${escapeHtml(model)}</span>
      </div>
      <aside class="hero">
        <p>Winning agent</p>
        <strong>${escapeHtml(winner.policyName)}</strong>
        <p>${escapeHtml(winner.description)}</p>
        <p class="margin">OpsScore ${winner.averages.opsScore?.toLocaleString("en-IN")} | Won ${wins[winner.policy] || 0}/${totalRollouts} rollouts | +${winnerMargin} over runner-up</p>
      </aside>
    </header>

    <section class="memo">
      <h2>Why ${escapeHtml(winner.policyName)} Won</h2>
      <p>${escapeHtml(winner.policyName)} achieved the highest composite OpsScore across ${totalRollouts} rollouts by ${winner.averages.serviceLevel > (runnerUp?.averages?.serviceLevel || 0) ? 'maintaining superior service levels' : 'driving higher throughput'} while ${winner.averages.risk < (runnerUp?.averages?.risk || Infinity) ? 'keeping risk below competitors' : 'accepting calculated risk for higher returns'}. ${runnerUp ? `Runner-up ${escapeHtml(runnerUp.policyName)} scored ${runnerUp.averages.opsScore?.toLocaleString("en-IN")} but ${runnerUp.averages.risk > winner.averages.risk ? 'carried more downside risk' : 'sacrificed throughput for safety'}.` : ''}</p>
    </section>

    <section>
      <h2>Agent Leaderboard</h2>
      <table>
        <thead><tr><th>Rank</th><th>Agent</th><th>OpsScore</th><th>Throughput</th><th>Service</th><th>Risk</th><th>Cost</th><th>Win Rate</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>

    <section>
      <h2>Agent Strategies</h2>
      <div class="grid">${agentCards}</div>
    </section>
  </main>
</body>
</html>`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const configPath = configPathFromArgs(args);
  const config = await readProjectConfigMaybe(configPath);
  const arena = args.arena || config?.arenaId || "footballops-v0";
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
      allRollouts.push(adapter.runPolicy({
        environment,
        policyKey,
        policyConfig,
        rolloutIndex: i,
        seed
      }));
    }
  }

  const { scoreboard, wins } = adapter.summarizeScoreboard({ allRollouts, policies, environment });

  const runDir = resolve(workspace, "runs", runId);
  const reportDir = resolve(workspace, "reports");
  const reportPath = resolve(reportDir, `${runId}.html`);
  await mkdir(runDir, { recursive: true });
  await mkdir(reportDir, { recursive: true });

  const runContract = createRunContract({
    arenaId: arena,
    adapter: environment.adapter ?? { id: arena, type: "arena-adapter" },
    question: environment.question,
    runId,
    rollouts,
    seed,
    workspace,
    environmentPath: envPath,
    scoreboard,
    wins,
    reportPath
  });
  runContract.mode = "agent-tournament";
  runContract.agentSource = source;
  runContract.agentModel = model;
  runContract.agentPlans = agentPlans;

  await writeFile(resolve(runDir, "run.json"), `${JSON.stringify(runContract, null, 2)}\n`);
  await writeFile(resolve(runDir, "agents.json"), `${JSON.stringify({ source, model, agents: agentPlans }, null, 2)}\n`);
  await writeFile(resolve(runDir, "rollouts.json"), `${JSON.stringify(compactRollouts(allRollouts), null, 2)}\n`);
  await writeFile(resolve(runDir, "scores.json"), `${JSON.stringify({ runId, arena, mode: "agent-tournament", scoreboard, wins }, null, 2)}\n`);
  await writeFile(resolve(runDir, "decision-memo.md"), `${renderAgentMemo({ runId, model, source, agentPlans, scoreboard, wins, reportPath })}\n`);
  await writeFile(reportPath, renderAgentReport({ runId, model, source, agentPlans, scoreboard, wins, rollouts, environment }));

  if (config) await appendProgress(config, `ran LLM agent tournament ${runId} with ${rollouts} rollouts`);

  const winner = scoreboard[0];
  console.log(`Winner: ${winner.policyName} (${winner.averages.opsScore.toLocaleString("en-IN")} OpsScore)`);
  console.log(`Agent source: ${source}`);
  console.log(`Report: ${reportPath}`);
  console.log(`Run directory: ${runDir}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
