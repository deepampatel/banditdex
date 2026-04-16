#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  appendProgress,
  configPathFromArgs,
  readProjectConfigMaybe,
  writeProjectConfig
} from "./lib/config.mjs";
import { describeNextAction, workspaceState } from "./lib/state.mjs";
import { parseArgs } from "./lib/workspace.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));

function runScript(script, args) {
  const result = spawnSync(process.execPath, [resolve(SCRIPT_DIR, script), ...args], {
    stdio: "inherit"
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}

function scriptArgs(config) {
  return [
    "--arena", config.arenaId,
    "--workspace", config.workspace
  ];
}

function setupArgs(config) {
  return [
    ...scriptArgs(config),
    "--question", config.question,
    "--mode", config.setupMode || "fast"
  ];
}

function runArgs(config) {
  return [
    ...scriptArgs(config),
    "--run", config.runId,
    "--rollouts", String(config.rollouts),
    "--seed", config.seed
  ];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const configPath = configPathFromArgs(args);
  let config = await readProjectConfigMaybe(configPath);

  if (!config) {
    if (!args.question) {
      console.log(`No OpsGym config found at ${configPath}`);
      console.log("Create one first:");
      console.log("./opsgym init --arena <adapter-id> --question \"<decision to simulate>\"");
      return;
    }

    config = await writeProjectConfig({
      project: args.project,
      arenaId: args.arena,
      question: args.question,
      workspace: args.workspace,
      rollouts: args.rollouts,
      runId: args.run,
      seed: args.seed,
      mode: args.auto ? "auto" : "guided"
    }, configPath);
    await appendProgress(config, `initialized project ${config.project}`);
  }

  const state = await workspaceState(config, { configPath });
  console.log(`Next action: ${state.nextAction}`);
  console.log(describeNextAction(state.nextAction));

  if (state.nextAction === "draft_arena") {
    runScript("setup-arena.mjs", setupArgs(config));
    await appendProgress(config, `drafted arena ${config.arenaId}`);
    return;
  }

  if (state.nextAction === "confirm_arena") {
    if (args.yes || config.mode === "auto") {
      runScript("confirm-arena.mjs", scriptArgs(config));
      await appendProgress(config, `confirmed arena ${config.arenaId}`);
      return;
    }
    console.log("");
    console.log("Confirmation required. Review:");
    console.log(state.paths.arenaSummary);
    console.log("");
    console.log("Then run:");
    console.log("./opsgym next --yes");
    return;
  }

  if (state.nextAction === "materialize_environment") {
    runScript("create-env.mjs", scriptArgs(config));
    await appendProgress(config, `materialized environment for ${config.arenaId}`);
    return;
  }

  if (state.nextAction === "run_tournament" || state.nextAction === "render_report") {
    runScript("run-tournament.mjs", runArgs(config));
    await appendProgress(config, `ran tournament ${config.runId} with ${config.rollouts} rollouts`);
    return;
  }

  console.log("Workspace complete.");
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
