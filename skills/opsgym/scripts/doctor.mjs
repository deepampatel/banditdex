#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { loadAdapter } from "./lib/adapter-loader.mjs";
import { fileExists } from "./lib/config.mjs";
import { parseArgs, SKILL_DIR } from "./lib/workspace.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = resolve(SKILL_DIR, "..", "..");
const CLI_PATH = resolve(PLUGIN_ROOT, "opsgym");

const checks = [];

function pass(name, detail = "") {
  checks.push({ ok: true, name, detail });
  console.log(`ok   ${name}${detail ? ` - ${detail}` : ""}`);
}

function fail(name, detail = "") {
  checks.push({ ok: false, name, detail });
  console.log(`fail ${name}${detail ? ` - ${detail}` : ""}`);
}

function run(command, args, options = {}) {
  return spawnSync(command, args, {
    cwd: options.cwd || PLUGIN_ROOT,
    encoding: "utf8",
    stdio: options.stdio || "pipe"
  });
}

function pythonCandidates() {
  return [
    process.env.PYTHON,
    "python3",
    "/Users/deepampatel/anaconda3/bin/python",
    "python"
  ].filter(Boolean);
}

function hasYaml(python) {
  const result = run(python, ["-c", "import yaml"]);
  return result.status === 0;
}

function processOutput(result) {
  if (result.error) return result.error.message;
  return [result.stderr, result.stdout].filter(Boolean).join("\n").trim();
}

async function checkNode() {
  const major = Number(process.versions.node.split(".")[0]);
  if (major >= 18) pass("node", process.version);
  else fail("node", `expected >=18, got ${process.version}`);
}

async function checkCli() {
  if (await fileExists(CLI_PATH)) pass("cli wrapper", CLI_PATH);
  else fail("cli wrapper", `missing at ${CLI_PATH}`);

  const result = run("bash", ["-n", CLI_PATH]);
  if (result.status === 0) pass("cli syntax", "bash -n");
  else fail("cli syntax", processOutput(result) || "bash -n failed");
}

async function checkSkill() {
  const skillPath = resolve(SKILL_DIR, "SKILL.md");
  if (await fileExists(skillPath)) pass("skill file", skillPath);
  else fail("skill file", `missing at ${skillPath}`);

  const validator = "/Users/deepampatel/.codex/skills/.system/skill-creator/scripts/quick_validate.py";
  if (!(await fileExists(validator))) {
    fail("skill validation", "quick_validate.py not found");
    return;
  }
  const python = pythonCandidates().find((candidate) => hasYaml(candidate));
  if (!python) {
    fail("skill validation", "no Python interpreter with PyYAML found");
    return;
  }
  const result = run(python, [validator, SKILL_DIR]);
  if (result.status === 0) pass("skill validation", result.stdout.trim());
  else fail("skill validation", processOutput(result));
}

async function checkAdapters() {
  const { discoverFirstAdapter } = await import("./lib/adapter-loader.mjs");
  const arenasDir = resolve(SKILL_DIR, "arenas");
  try {
    const { readdir } = await import("node:fs/promises");
    const entries = await readdir(arenasDir, { withFileTypes: true });
    let found = 0;
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const adapterPath = resolve(arenasDir, entry.name, "adapter.mjs");
      try {
        const adapter = await loadAdapter(entry.name);
        pass(`adapter ${entry.name}`, adapter.adapterMeta.description);
        found += 1;
      } catch (error) {
        fail(`adapter ${entry.name}`, error.message || String(error));
      }
    }
    if (found === 0) fail("adapters", "no installed adapters found");
  } catch (error) {
    fail("adapters", error.message || String(error));
  }
}

async function checkScriptSyntax() {
  const result = run("bash", ["-lc", "for f in $(rg --files skills/opsgym/scripts -g '*.mjs'); do node --check \"$f\" || exit 1; done"], {
    cwd: PLUGIN_ROOT
  });
  if (result.status === 0) pass("script syntax", "all .mjs files");
  else fail("script syntax", processOutput(result));
}

async function checkSmokeRun() {
  const { discoverFirstAdapter } = await import("./lib/adapter-loader.mjs");
  const smokeArena = await discoverFirstAdapter();
  if (!smokeArena) { fail("smoke run", "no adapters installed to smoke-test"); return; }
  const workspace = await mkdtemp(resolve(tmpdir(), "opsgym-doctor-"));
  const config = resolve(workspace, "opsgym.json");
  try {
    const init = run(CLI_PATH, [
      "init",
      "--config", config,
      "--workspace", resolve(workspace, ".ops-gym"),
      "--project", "doctor-smoke",
      "--arena", smokeArena,
      "--rollouts", "5"
    ]);
    if (init.status !== 0) throw new Error(processOutput(init) || "init failed");

    const loop = run(CLI_PATH, [
      "loop",
      "--config", config,
      "--yes",
      "--max-iterations", "6"
    ]);
    if (loop.status !== 0) throw new Error(processOutput(loop) || "loop failed");

    const runJson = resolve(workspace, ".ops-gym", "runs", "doctor-smoke-baseline", "run.json");
    if (!(await fileExists(runJson))) throw new Error("run.json was not created");
    pass("smoke run", "5-rollout loop completed");
  } catch (error) {
    fail("smoke run", error.message || String(error));
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  console.log("OpsGym Doctor");
  console.log("");

  await checkNode();
  await checkCli();
  await checkSkill();
  await checkAdapters();
  await checkScriptSyntax();
  if (!args["skip-smoke"]) await checkSmokeRun();
  else pass("smoke run", "skipped");

  console.log("");
  const failures = checks.filter((check) => !check.ok);
  if (failures.length) {
    console.log(`${failures.length} check(s) failed.`);
    process.exit(1);
  }
  console.log("All checks passed.");
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
