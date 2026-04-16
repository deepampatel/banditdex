import { access, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { SKILL_DIR } from "./workspace.mjs";

const REQUIRED_FUNCTION_EXPORTS = [
  "buildDraftArena",
  "materializeEnvironment",
  "listPolicies",
  "runPolicy",
  "summarizeScoreboard"
];

function adapterIdFrom(adapterRef) {
  if (typeof adapterRef === "string") return adapterRef;
  return adapterRef?.id;
}

export function adapterRefFromArena(arenaSpec) {
  return arenaSpec.adapter ?? {
    id: arenaSpec.arenaId,
    entry: `arenas/${arenaSpec.arenaId}/adapter.mjs`
  };
}

export function resolveAdapterPath(adapterRef) {
  const adapterId = adapterIdFrom(adapterRef);
  if (!adapterId) throw new Error("Adapter id is required.");
  const entry = typeof adapterRef === "object" && adapterRef?.entry
    ? adapterRef.entry
    : `arenas/${adapterId}/adapter.mjs`;
  return resolve(SKILL_DIR, entry);
}

export async function discoverFirstAdapter() {
  const arenasDir = resolve(SKILL_DIR, "arenas");
  try {
    const entries = await readdir(arenasDir, { withFileTypes: true });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (!entry.isDirectory()) continue;
      const adapterPath = resolve(arenasDir, entry.name, "adapter.mjs");
      try { await access(adapterPath); return entry.name; } catch { /* skip */ }
    }
  } catch { /* arenas dir missing */ }
  return null;
}

export async function resolveArenaId(explicit) {
  if (explicit) return explicit;
  const discovered = await discoverFirstAdapter();
  if (!discovered) throw new Error("No installed adapters found. Create one under arenas/<id>/adapter.mjs");
  return discovered;
}

export async function loadAdapter(adapterRef) {
  const adapterId = adapterIdFrom(adapterRef);
  const entryPath = resolveAdapterPath(adapterRef);
  try {
    await access(entryPath);
  } catch {
    throw new Error(`Adapter ${adapterId} is not installed at ${entryPath}`);
  }

  const adapter = await import(pathToFileURL(entryPath).href);
  if (!adapter.adapterMeta?.id) {
    throw new Error(`Adapter ${adapterId} is missing adapterMeta.id`);
  }
  for (const exportName of REQUIRED_FUNCTION_EXPORTS) {
    if (typeof adapter[exportName] !== "function") {
      throw new Error(`Adapter ${adapterId} is missing ${exportName}()`);
    }
  }
  return adapter;
}
