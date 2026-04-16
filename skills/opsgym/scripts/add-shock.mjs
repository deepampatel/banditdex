#!/usr/bin/env node
import { access } from "node:fs/promises";
import { adapterRefFromArena, loadAdapter, resolveArenaId } from "./lib/adapter-loader.mjs";
import { arenaPaths, loadArenaSpec, parseArgs, slug, writeArenaArtifacts, writeEnvironmentArtifacts } from "./lib/workspace.mjs";

function fallbackShock(type, args) {
  return {
    id: slug(`${type}-${args.day || 4}-${args.label || type}`).slice(0, 80),
    type,
    label: args.label || `${type} shock`,
    day: Number(args.day || 4),
    severity: Math.max(0, Math.min(1, Number(args.severity || 0.35)))
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const arenaId = await resolveArenaId(args.arena);
  const workspace = args.workspace || ".ops-gym";
  const type = args.type;
  if (!type) throw new Error("--type is required. Specify the shock type for this domain.");
  const arenaSpec = await loadArenaSpec(workspace, arenaId);
  const adapter = await loadAdapter(adapterRefFromArena(arenaSpec));
  const shock = adapter.buildShock ? adapter.buildShock(type, args) : fallbackShock(type, args);

  arenaSpec.shocks = [...(arenaSpec.shocks || []).filter((item) => item.id !== shock.id), shock];
  arenaSpec.updatedAt = new Date().toISOString();

  await writeArenaArtifacts(workspace, arenaSpec);

  const paths = arenaPaths(workspace, arenaId);
  try {
    await access(paths.envJson);
    await writeEnvironmentArtifacts(workspace, arenaSpec, adapter.materializeEnvironment(arenaSpec));
  } catch {
    // Environment has not been materialized yet. The arena remains the source of truth.
  }

  console.log(`Added shock to ${paths.arenaJson}`);
  console.log(`${shock.label}: ${shock.type}, day ${shock.day}, severity ${shock.severity}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
