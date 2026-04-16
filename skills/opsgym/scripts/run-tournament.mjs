#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { adapterRefFromArena, loadAdapter } from "./lib/adapter-loader.mjs";
import { createRunContract } from "./lib/contracts.mjs";
import { loadEnvironment, parseArgs } from "./lib/workspace.mjs";

function formatInr(value) {
  return `₹${Math.round(value).toLocaleString("en-IN")}`;
}

function metricList(env, scoreboard) {
  const catalog = new Map((env.metrics?.catalog || []).map((metric) => [metric.id, metric]));
  const orderedIds = [env.metrics?.primary, ...(env.metrics?.raw || [])].filter(Boolean);
  const seen = new Set();
  const metricIds = [];

  for (const metricId of orderedIds) {
    if (seen.has(metricId)) continue;
    seen.add(metricId);
    metricIds.push(metricId);
  }

  for (const metricId of Object.keys(scoreboard[0]?.averages || {})) {
    if (seen.has(metricId)) continue;
    seen.add(metricId);
    metricIds.push(metricId);
  }

  return metricIds.map((metricId) => catalog.get(metricId) ?? {
    id: metricId,
    label: metricId,
    unit: "count"
  });
}

function formatMetric(metric, value) {
  if (value === null || value === undefined) return "-";
  if (metric.unit === "inr") return formatInr(value);
  if (metric.unit === "ratio" || metric.unit === "percent") return `${Math.round(Number(value) * 100)}%`;
  return Number(value).toLocaleString("en-IN");
}

function renderReport({ env, runId, scoreboard, wins, reportRows }) {
  const winner = scoreboard[0];
  const metrics = metricList(env, scoreboard);
  const entityLabel = env.summary?.entityLabel || "Entities";
  const headers = metrics.map((metric) => `<th>${metric.label}</th>`).join("");
  const rows = scoreboard.map((row, index) => `
    <tr>
      <td>${index + 1}</td>
      <td><strong>${row.policyName}</strong><br><span>${row.description}</span></td>
      ${metrics.map((metric) => `<td>${formatMetric(metric, row.averages[metric.id])}</td>`).join("")}
      <td>${wins[row.policy] || 0}</td>
    </tr>`).join("");
  const shockItems = (env.shocks || []).map((shock) => `<li>Day ${shock.day}: ${shock.label} <span>${shock.type}, severity ${shock.severity}</span></li>`).join("");
  const entityCards = (env.entities || []).map((entity) => `<article>
    <h3>${entity.name}</h3>
    <p>${entity.label || ""}</p>
    <dl>
      ${(entity.stats || []).map((stat) => `<div><dt>${stat.label}</dt><dd>${stat.value}</dd></div>`).join("")}
    </dl>
  </article>`).join("");
  const primaryMetric = metrics.find((metric) => metric.id === env.metrics?.primary) || metrics[0];

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpsGym Report - ${runId}</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17211f;
      --muted: #63706d;
      --line: #dbe4df;
      --mint: #0f766e;
      --gold: #b7791f;
      --red: #b42318;
      --blue: #2563eb;
      --paper: #f7faf8;
      --white: #ffffff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--paper); color: var(--ink); }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 48px; }
    header { display: grid; gap: 16px; grid-template-columns: 1.3fr 0.7fr; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 24px; }
    h1 { font-size: clamp(32px, 5vw, 58px); line-height: 1; margin: 0; max-width: 850px; }
    h2 { font-size: 22px; margin: 0 0 14px; }
    p { color: var(--muted); line-height: 1.55; margin: 0; }
    .winner { background: var(--ink); color: white; padding: 18px; border-radius: 8px; }
    .winner p { color: #d7e3df; }
    .winner strong { color: #8ee4d4; display: block; font-size: 24px; margin-top: 4px; }
    section { margin-top: 28px; }
    table { width: 100%; border-collapse: collapse; background: var(--white); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    th, td { padding: 13px 12px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; background: #edf5f2; }
    td span { color: var(--muted); font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
    article { background: var(--white); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    h3 { margin: 0 0 4px; font-size: 16px; }
    dl { display: grid; gap: 8px; margin: 14px 0 0; }
    dl div { display: flex; justify-content: space-between; gap: 10px; }
    dt { color: var(--muted); }
    dd { margin: 0; font-weight: 700; }
    .shocks { background: var(--white); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    .shocks ul { margin: 0; padding-left: 20px; color: var(--ink); }
    .shocks li { margin: 8px 0; }
    .shocks span { color: var(--muted); }
    .memo { border-left: 5px solid var(--mint); background: var(--white); padding: 18px; border-radius: 8px; }
    @media (max-width: 860px) {
      header { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p>OpsGym / ${env.arena}</p>
        <h1>${env.question}</h1>
      </div>
      <aside class="winner">
        <p>Winning policy</p>
        <strong>${winner.policyName}</strong>
        <p>Average ${primaryMetric.label} ${formatMetric(primaryMetric, winner.averages[primaryMetric.id])} across ${reportRows} rollouts.</p>
      </aside>
    </header>

    <section class="memo">
      <h2>Recommendation</h2>
      <p>Use ${winner.policyName}. It produced the strongest outcome on the primary score while staying competitive across the supporting metrics.</p>
    </section>

    <section>
      <h2>Policy Leaderboard</h2>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Policy</th>
            ${headers}
            <th>Wins</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>

    <section class="shocks">
      <h2>Scenario Shocks</h2>
      <ul>${shockItems}</ul>
    </section>

    <section>
      <h2>${entityLabel}</h2>
      <div class="grid">${entityCards}</div>
    </section>
  </main>
</body>
</html>`;
}

function decisionMemo({ env, runId, scoreboard, wins, reportPath }) {
  const winner = scoreboard[0];
  const runnerUp = scoreboard[1];
  const metrics = metricList(env, scoreboard);
  return [
    `# OpsGym Decision Memo: ${runId}`,
    "",
    `Question: ${env.question}`,
    "",
    `Recommendation: use ${winner.policyName}.`,
    "",
    "Why it won:",
    ...metrics.map((metric) => `- ${metric.label}: ${formatMetric(metric, winner.averages[metric.id])}`),
    `- Rollout wins: ${wins[winner.policy] || 0}`,
    "",
    runnerUp ? `Runner-up: ${runnerUp.policyName} with ${formatMetric(metrics[0], runnerUp.averages[metrics[0].id])} on ${metrics[0].label}.` : "",
    "",
    "Shock assumptions:",
    ...(env.shocks || []).map((shock) => `- Day ${shock.day}: ${shock.label} (${shock.type}, severity ${shock.severity})`),
    "",
    `Report: ${reportPath}`
  ].filter(Boolean).join("\n");
}

function compactRollouts(allRollouts) {
  return allRollouts.map((rollout) => ({
    policy: rollout.policy,
    policyName: rollout.policyName,
    rolloutIndex: rollout.rolloutIndex,
    metrics: rollout.metrics
  }));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const { resolveArenaId } = await import("./lib/adapter-loader.mjs");
  const arena = await resolveArenaId(args.arena);
  const workspace = args.workspace || ".ops-gym";
  const runId = args.run || `run-${new Date().toISOString().slice(0, 10)}`;
  const rollouts = Number(args.rollouts || 100);
  const seed = args.seed || runId;
  const envPath = resolve(workspace, "environments", arena, "environment.json");
  const env = await loadEnvironment(workspace, arena);
  const adapter = await loadAdapter(adapterRefFromArena({ arenaId: arena, adapter: env.adapter }));
  const policies = adapter.listPolicies(env);

  const allRollouts = [];
  for (let i = 0; i < rollouts; i += 1) {
    for (const policyKey of Object.keys(policies)) {
      allRollouts.push(adapter.runPolicy({
        environment: env,
        policyKey,
        rolloutIndex: i,
        seed
      }));
    }
  }

  const { scoreboard, wins } = adapter.summarizeScoreboard({ allRollouts, policies, environment: env });

  const runDir = resolve(workspace, "runs", runId);
  const reportDir = resolve(workspace, "reports");
  const reportPath = resolve(reportDir, `${runId}.html`);
  await mkdir(runDir, { recursive: true });
  await mkdir(reportDir, { recursive: true });

  const runContract = createRunContract({
    arenaId: arena,
    adapter: env.adapter ?? { id: arena, type: "arena-adapter" },
    question: env.question,
    runId,
    rollouts,
    seed,
    workspace,
    environmentPath: envPath,
    scoreboard,
    wins,
    reportPath
  });

  await writeFile(resolve(runDir, "run.json"), `${JSON.stringify(runContract, null, 2)}\n`);
  await writeFile(resolve(runDir, "rollouts.json"), `${JSON.stringify(compactRollouts(allRollouts), null, 2)}\n`);
  await writeFile(resolve(runDir, "scores.json"), `${JSON.stringify({ runId, arena, scoreboard, wins }, null, 2)}\n`);
  await writeFile(reportPath, renderReport({ env, runId, scoreboard, wins, reportRows: rollouts }));
  await writeFile(resolve(runDir, "decision-memo.md"), decisionMemo({ env, runId, scoreboard, wins, reportPath }));
  await writeFile(
    resolve(runDir, "trace.md"),
    [
      `# OpsGym Trace: ${runId}`,
      "",
      `Arena source: ${resolve(workspace, "arenas", arena, "arena.json")}`,
      `Environment: ${envPath}`,
      `Run contract: ${resolve(runDir, "run.json")}`,
      `Rollouts per policy: ${rollouts}`,
      `Seed: ${seed}`,
      "",
      "Policies:",
      ...Object.entries(policies).map(([key, policy]) => `- ${key}: ${policy.description}`)
    ].join("\n")
  );

  const winner = scoreboard[0];
  console.log(`Winner: ${winner.policyName} (${winner.averages.opsScore.toLocaleString("en-IN")} OpsScore)`);
  console.log(`Report: ${reportPath}`);
  console.log(`Run directory: ${runDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
