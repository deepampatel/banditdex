import { access, appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { OPSGYM_SCHEMA_VERSION } from "./contracts.mjs";
import { slug } from "./workspace.mjs";

export const DEFAULT_WORKSPACE = ".ops-gym";
export const DEFAULT_CONFIG_FILE = "opsgym.json";
export const DEFAULT_ARENA_ID = null; // resolved at runtime from installed adapters
export const DEFAULT_QUESTION = null; // resolved at runtime from adapter defaults

export function configPathFromArgs(args = {}) {
  return resolve(args.config || DEFAULT_CONFIG_FILE);
}

export function defaultProjectName(question, arenaId = DEFAULT_ARENA_ID) {
  const base = slug(question || arenaId || "opsgym-run").slice(0, 48);
  return base || "opsgym-run";
}

export function normalizeConfig(input = {}) {
  const arenaId = input.arenaId || input.arena || DEFAULT_ARENA_ID;
  const question = input.question || DEFAULT_QUESTION;
  const project = input.project || defaultProjectName(question, arenaId);
  return {
    schemaVersion: OPSGYM_SCHEMA_VERSION,
    project,
    arenaId,
    domain: input.domain || "decision-ops",
    question,
    workspace: input.workspace || DEFAULT_WORKSPACE,
    setupMode: input.setupMode || input.mode || "fast",
    runId: input.runId || `${project}-baseline`,
    rollouts: Number(input.rollouts || 100),
    seed: input.seed || `${project}-seed`,
    mode: input.mode === "auto" ? "auto" : "guided",
    maxIterations: Number(input.maxIterations || 5),
    createdAt: input.createdAt || new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };
}

export async function fileExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

export async function readProjectConfig(configPath = resolve(DEFAULT_CONFIG_FILE)) {
  const text = await readFile(configPath, "utf8");
  return normalizeConfig(JSON.parse(text));
}

export async function readProjectConfigMaybe(configPath = resolve(DEFAULT_CONFIG_FILE)) {
  if (!(await fileExists(configPath))) return null;
  return readProjectConfig(configPath);
}

export async function writeProjectConfig(config, configPath = resolve(DEFAULT_CONFIG_FILE)) {
  const normalized = normalizeConfig(config);
  await writeFile(configPath, `${JSON.stringify(normalized, null, 2)}\n`);
  return normalized;
}

export function progressPath(config) {
  return resolve(config.workspace || DEFAULT_WORKSPACE, "progress.md");
}

export async function ensureProgress(config) {
  const path = progressPath(config);
  await mkdir(resolve(config.workspace || DEFAULT_WORKSPACE), { recursive: true });
  if (!(await fileExists(path))) {
    await writeFile(path, "# OpsGym Progress\n\n");
  }
  return path;
}

export async function appendProgress(config, message) {
  const path = await ensureProgress(config);
  const timestamp = new Date().toISOString();
  await appendFile(path, `- ${timestamp} ${message}\n`);
  return path;
}
