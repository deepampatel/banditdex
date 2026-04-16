#!/usr/bin/env node
import { adapterRefFromArena, loadAdapter, resolveArenaId } from "./lib/adapter-loader.mjs";
import { loadArenaSpec, parseArgs, writeEnvironmentArtifacts } from "./lib/workspace.mjs";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const arenaId = await resolveArenaId(args.arena);
  const workspace = args.workspace || ".ops-gym";
  const allowDraft = Boolean(args["allow-draft"]);
  const arenaSpec = await loadArenaSpec(workspace, arenaId);
  const adapter = await loadAdapter(adapterRefFromArena(arenaSpec));

  if (arenaSpec.status !== "confirmed" && !allowDraft) {
    throw new Error(
      `Arena ${arenaId} is ${arenaSpec.status}. Confirm it first, or rerun with --allow-draft for debugging.`
    );
  }

  const environment = adapter.materializeEnvironment(arenaSpec);
  const { paths } = await writeEnvironmentArtifacts(workspace, arenaSpec, environment);
  console.log(`Created environment: ${paths.envJson}`);
  console.log(`Source arena: ${paths.arenaJson}`);
  console.log(`Entities: ${(environment.entities || []).map((entity) => entity.name).join(", ")}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
