#!/usr/bin/env node
import { readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { SKILL_DIR } from "./lib/workspace.mjs";
import { fileExists } from "./lib/config.mjs";

async function installedAdapters() {
  const arenasDir = resolve(SKILL_DIR, "arenas");
  if (!(await fileExists(arenasDir))) return [];
  const entries = await readdir(arenasDir, { withFileTypes: true });
  const adapters = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const adapterPath = resolve(arenasDir, entry.name, "adapter.mjs");
    if (!(await fileExists(adapterPath))) continue;
    try {
      const adapter = await import(pathToFileURL(adapterPath).href);
      adapters.push({
        id: adapter.adapterMeta?.id || entry.name,
        domain: adapter.adapterMeta?.domain || "decision-ops",
        version: adapter.adapterMeta?.version || "0.1",
        description: adapter.adapterMeta?.description || "Arena adapter",
        path: adapterPath
      });
    } catch (error) {
      adapters.push({
        id: entry.name,
        domain: "unknown",
        version: "?",
        description: `failed to load: ${error.message || error}`,
        path: adapterPath
      });
    }
  }

  return adapters.sort((a, b) => a.id.localeCompare(b.id));
}

async function main() {
  const installed = await installedAdapters();
  if (!installed.length) {
    console.log("No installed adapters.");
    console.log("Create one at: arenas/<adapter-id>/adapter.mjs");
    return;
  }

  console.log("Installed adapters:");
  for (const adapter of installed) {
    console.log(`  ${adapter.id} (${adapter.domain}, v${adapter.version})`);
    console.log(`    ${adapter.description}`);
  }
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
