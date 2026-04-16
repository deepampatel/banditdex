#!/usr/bin/env node
import { resolveArenaId } from "./lib/adapter-loader.mjs";
import { loadArenaSpec, parseArgs, writeArenaArtifacts } from "./lib/workspace.mjs";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const arenaId = await resolveArenaId(args.arena);
  const workspace = args.workspace || ".ops-gym";
  const arenaSpec = await loadArenaSpec(workspace, arenaId);

  if (arenaSpec.status === "confirmed") {
    console.log(`Arena already confirmed: ${arenaId}`);
    return;
  }

  arenaSpec.status = "confirmed";
  arenaSpec.confirmedAt = new Date().toISOString();
  if (args.notes) arenaSpec.confirmationNotes = args.notes;

  const paths = await writeArenaArtifacts(workspace, arenaSpec);
  console.log(`Confirmed arena: ${paths.arenaJson}`);
  console.log(`Summary: ${paths.arenaSummary}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
