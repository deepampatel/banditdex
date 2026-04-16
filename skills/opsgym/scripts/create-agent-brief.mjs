#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { AGENT_SCHEMA, PARAMETER_DESCRIPTIONS, environmentBrief } from "./lib/agents.mjs";
import { appendProgress, configPathFromArgs, readProjectConfigMaybe } from "./lib/config.mjs";
import { loadEnvironment, parseArgs } from "./lib/workspace.mjs";

function exampleAgent(index) {
  return {
    id: `codex-agent-${index}`,
    name: `Codex Agent ${index}`,
    thesis: "Replace this with a distinct decision thesis before running the tournament.",
    parameters: {
      capacityAggression: 0.55,
      riskTolerance: 0.45,
      executionAggression: 0.95,
      fallbackRecovery: 0.75,
      priorityFocus: 0.75
    }
  };
}

function renderBrief({ runId, arena, agentCount, environment, brief }) {
  return [
    `# OpsGym Agent Brief: ${runId}`,
    "",
    `Arena: ${arena}`,
    `Question: ${environment.question}`,
    `Requested agents: ${agentCount}`,
    "",
    "## Codex Task",
    "",
    "Create distinct decision agents for this arena. Each agent should have a clear thesis and numeric parameters that encode how it behaves in rollouts.",
    "",
    "Do not make tiny variants of the same agent. Prefer meaningful contrasts: protect vs push, conservative vs aggressive, resilience vs throughput, risk-averse vs risk-seeking.",
    "",
    "## Parameter Contract",
    "",
    ...PARAMETER_DESCRIPTIONS.map(([name, description]) => `- ${name}: ${description}`),
    "",
    "## Output JSON Shape",
    "",
    "Write a JSON file with this shape:",
    "",
    "```json",
    JSON.stringify({ agents: [exampleAgent(1), exampleAgent(2)] }, null, 2),
    "```",
    "",
    "## Arena Context",
    "",
    "```json",
    JSON.stringify(brief, null, 2),
    "```",
    "",
    "## Strict Schema",
    "",
    "```json",
    JSON.stringify(AGENT_SCHEMA, null, 2),
    "```"
  ].join("\n");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const configPath = configPathFromArgs(args);
  const config = await readProjectConfigMaybe(configPath);
  const { resolveArenaId } = await import("./lib/adapter-loader.mjs");
  const arena = await resolveArenaId(args.arena || config?.arenaId);
  const workspace = args.workspace || config?.workspace || ".ops-gym";
  const runId = args.run || `${config?.project || "agent"}-agents`;
  const agentCount = Number.parseInt(args.agents || 3, 10);
  const environment = await loadEnvironment(workspace, arena);
  const outDir = resolve(workspace, "agent-plans");
  const briefPath = resolve(outDir, `${runId}.brief.md`);
  const templatePath = resolve(outDir, `${runId}.template.json`);

  await mkdir(outDir, { recursive: true });
  const brief = environmentBrief(environment);
  await writeFile(briefPath, `${renderBrief({ runId, arena, agentCount, environment, brief })}\n`);
  await writeFile(templatePath, `${JSON.stringify({ agents: Array.from({ length: Math.min(Math.max(agentCount, 1), 6) }, (_, index) => exampleAgent(index + 1)) }, null, 2)}\n`);

  if (config) await appendProgress(config, `created Codex agent brief ${runId}`);

  console.log(`Agent brief: ${briefPath}`);
  console.log(`Template: ${templatePath}`);
  console.log(`Next: write real agents to ${resolve(outDir, `${runId}.json`)}`);
  console.log(`Run: ./opsgym agent-run --run ${runId} --agents-file ${resolve(outDir, `${runId}.json`)}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
