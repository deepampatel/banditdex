#!/usr/bin/env node
import { configPathFromArgs, readProjectConfigMaybe } from "./lib/config.mjs";
import { describeNextAction, workspaceState } from "./lib/state.mjs";
import { parseArgs } from "./lib/workspace.mjs";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const configPath = configPathFromArgs(args);
  const config = await readProjectConfigMaybe(configPath);

  if (!config) {
    console.log(`No OpsGym config found at ${configPath}`);
    console.log("Next: ./opsgym init --arena <adapter-id> --question \"<decision to simulate>\"");
    return;
  }

  const state = await workspaceState(config, { configPath });
  console.log(`Project: ${state.project}`);
  console.log(`Config: ${state.configPath}`);
  console.log(`Workspace: ${state.workspace}`);
  console.log(`Arena: ${state.arenaId} (${state.arenaStatus})`);
  console.log(`Environment: ${state.environmentExists ? "ready" : "missing"}`);
  console.log(`Run: ${state.runId} (${state.runExists ? "ready" : "missing"})`);
  console.log(`Report: ${state.reportExists ? "ready" : "missing"}`);
  console.log(`Progress: ${state.progressPath}`);
  console.log(`Next action: ${state.nextAction}`);
  console.log(describeNextAction(state.nextAction));

  if (state.runs.length) {
    console.log("");
    console.log("Runs:");
    for (const run of state.runs) {
      console.log(`- ${run.id}${run.hasRunContract ? "" : " (missing run.json)"}`);
    }
  }
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
