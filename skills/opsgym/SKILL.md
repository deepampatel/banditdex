---
name: opsgym
description: Create, modify, run, and evaluate OpsGym operational simulation workspaces from messy business context. Use when the user wants to train or evaluate decision agents, run policy tournaments, create business operation environments, add shocks/scenarios, or generate decision dashboards/reports. Do not use for generic coding tasks unless OpsGym or operational simulation is requested.
---

# OpsGym

OpsGym is a Codex-native workflow for creating operational simulation gyms where decision agents can test strategies before those strategies touch reality.

## Product Shape

OpsGym is generic infrastructure, not a single-domain demo. Treat every simulation as:

```text
messy situation -> arena -> environment -> agents/policies -> rollouts -> scorecard -> decision memo
```

The scalable architecture is:

- Core scripts manage workspace state, validation, orchestration, reporting, and contracts.
- Arena adapters own domain logic: actors, actions, constraints, shocks, transitions, and scoring.
- Codex owns agent design, critique, and recommendation narrative inside an interactive session.
- Optional API-backed generation exists only for standalone CLI automation outside Codex.

## Operating Model

- The skill is the operator manual and bundled tools.
- The active project is the workspace. Do not write generated runs inside the skill directory.
- Generated artifacts live under `.ops-gym/` in the current project.
- `opsgym.json` is the intended project-level session contract: project name, arena id, user question, workspace, rollouts, seed, and mode.
- `.ops-gym/progress.md` is the intended cross-run journal for state-machine progress and major decisions.
- Prefer one deep arena over many shallow arenas, but do not force a domain that the user did not ask for.
- The skill auto-discovers installed adapters at runtime. No adapter is hardcoded as the default.
- Bundled adapters ship under `arenas/`. Run `list` to see what is installed.
- Every recommendation must be backed by a tournament result, scorecard, or explicit assumption.
- Scripted policies are baselines. The intended product experience is LLM decision agents competing inside arenas and being scored by rollouts. When the user asks to "run" an arena without specifying baseline or agents, prefer the agent workflow.
- Arenas are first-class. Draft the world, review it, confirm it, then run it.
- When the user has not defined the world yet, ask 3-5 high-leverage questions, not a giant questionnaire.
- Do not hard-code a domain in the generic core. Add a new adapter when a domain needs executable behavior.

## Feature Inventory

- Project state machine: `init`, `status`, `next`, `loop`.
- Adapter discovery: `list`.
- Arena lifecycle: `setup`, `confirm`, `env`.
- Baseline tournaments: `run`.
- Codex-native agent tournaments: `agent-brief`, `agent-validate`, `agent-run --agents-file`.
- Standalone API-backed agent generation: `agent-run` without `--agents-file`, requiring `OPENAI_API_KEY`.
- Shock injection: `shock`.
- Run comparison: `compare`.
- Local health checks: `doctor`.
- Outputs: `arena.json`, `environment.json`, `run.json`, `rollouts.json`, `scores.json`, `decision-memo.md`, comparison artifacts, and HTML reports.
- Built-in runnable arenas: `footballops-v0`, `deliveryops-v0`, `hospitalops-v0`.
- Scalable extension point: `arenas/<adapter-id>/adapter.mjs`.

## Scalability Notes

- Rollouts are deterministic for a given seed and can be partitioned by `(agent, rolloutIndex)`.
- The current runner executes rollouts sequentially in-process.
- The next scale step is a worker-pool runner such as `--jobs <n>`.
- Agent design can be parallelized by Codex at the planning layer, but the packaged CLI currently validates and scores the resulting plan file as one tournament.
- Keep run artifacts append-only and contract-driven so future distributed runners can merge partial rollout files.

## Adapter Selection

- Start from the user's situation, not from a bundled demo.
- Run `list` to see which adapters are installed. Match the user's domain to the closest adapter.
- For a different domain, first design the arena in domain-native terms: actors, actions, constraints, shocks, policies, and metrics.
- If no executable adapter exists for that domain yet, say that directly and offer to create a new adapter before running simulations.
- Never force an unrelated domain just because an adapter exists.

## State-Machine Workflow

OpsGym should behave like a state machine, not an open-ended brainstorming loop. The intended command vocabulary is:

- `init`: create or update `opsgym.json` from the user's situation.
- `list`: show installed executable adapters and clearly separate designable-but-not-runnable domains.
- `status`: summarize `opsgym.json`, arena status, environment status, runs, reports, and next action.
- `next`: inspect project state and perform the next safe phase.
- `loop`: repeatedly call `next`, stopping at confirmation gates unless the user explicitly allows unattended progress.
- `compare`: compare two run contracts and emit structured comparison artifacts.
- `doctor`: verify local tooling, adapters, skill metadata, script syntax, and a smoke simulation.
- `agent-brief`: write a compact Codex prompt brief and JSON template for arena-specific agent design.
- `agent-validate`: validate Codex-authored agent JSON before scoring.
- `agent-run`: ask an LLM to create decision agents, then evaluate those agents through arena rollouts.

When OpsGym is invoked inside Codex, prefer Codex-authored agents over requiring an `OPENAI_API_KEY`:

1. Read the confirmed `environment.json`.
2. Run `agent-brief` to write `.ops-gym/agent-plans/<run-id>.brief.md`.
3. Use Codex reasoning to create 2-6 distinct agent plans.
4. Write them to `.ops-gym/agent-plans/<run-id>.json`.
5. Run `agent-validate --file .ops-gym/agent-plans/<run-id>.json`.
6. Run `agent-run --agents-file .ops-gym/agent-plans/<run-id>.json`.

Use direct OpenAI API generation only for standalone CLI automation where no Codex session is available.

Phase resolution:

1. No `opsgym.json`: initialize the session contract.
2. No `arena.json`: draft an arena from the question and adapter.
3. `arena.status = draft`: show the summary and ask for confirmation.
4. Confirmed arena but no `environment.json`: materialize the environment.
5. Environment but no run: choose the run mode based on user intent (see below).
6. Multiple runs or shock/candidate: compare and report.

**Choosing the run mode (phase 5):**

- If the user mentions **agents**, **competing**, **agent tournament**, **design agents**, **LLM agents**, or **strategies competing**: use the **Codex-authored agent workflow** — call `agent-brief`, read the brief, design 2-6 distinct agents, write the JSON, validate, and run with `agent-run --agents-file`.
- If the user asks for a **baseline**, **policy tournament**, or does not mention agents: run a **baseline policy tournament** with `run`.
- When in doubt, prefer the agent workflow — it is the primary product experience.

For a new domain, OpsGym may draft the arena concept, but it must not claim executable simulation until an adapter exists.

## Workspace Layout

Use this layout in the active project:

```text
.ops-gym/
  arenas/<arena>/
    arena.json
    arena-summary.md
  environments/<arena>/
    environment.json
    create-summary.md
  runs/<run-id>/
    run.json
    rollouts.json
    scores.json
    decision-memo.md
    trace.md
  reports/<run-id>.html
  comparisons/<comparison-id>/
    comparison.json
    comparison.md
  progress.md
opsgym.json
```

## Arena-First Flow

1. Draft an arena from the user's goal and source files.
2. Show the summary and ask for confirmation if the arena is still draft.
3. Confirm the arena once the user is happy.
4. Materialize the executable environment.
5. Run tournaments, inject shocks, and generate reports.

Default clarification questions:

- What decision are we testing?
- Who are the actors?
- Which shocks matter most?
- What actions should the agent be allowed to take?
- How should we judge success?

Prefer a draft summary plus confirmation over asking every detail up front.

## Main Workflows

Resolve script paths relative to this skill directory.

Draft an executable arena with an installed adapter:

```bash
node <skill-dir>/scripts/setup-arena.mjs \
  --arena <adapter-id> \
  --input-dir <business-data-dir> \
  --question "<decision to simulate>" \
  --mode fast \
  --workspace .ops-gym
```

Confirm the arena:

```bash
node <skill-dir>/scripts/confirm-arena.mjs \
  --arena <adapter-id> \
  --workspace .ops-gym
```

Create an executable environment from the confirmed arena:

```bash
node <skill-dir>/scripts/create-env.mjs \
  --arena <adapter-id> \
  --workspace .ops-gym
```

Run a policy tournament:

```bash
node <skill-dir>/scripts/run-tournament.mjs \
  --arena <adapter-id> \
  --run <run-id> \
  --rollouts 100 \
  --workspace .ops-gym
```

Run a Codex-authored agent tournament:

```bash
node <skill-dir>/scripts/create-agent-brief.mjs \
  --arena <adapter-id> \
  --run <run-id> \
  --agents 3 \
  --workspace .ops-gym

node <skill-dir>/scripts/run-agent-tournament.mjs \
  --arena <adapter-id> \
  --run <run-id> \
  --agents-file .ops-gym/agent-plans/<run-id>.json \
  --rollouts 100 \
  --workspace .ops-gym
```

Add a shock, then rerun:

```bash
node <skill-dir>/scripts/add-shock.mjs \
  --arena <adapter-id> \
  --type <shock-type> \
  --day 4 \
  --severity 0.45 \
  --label "<shock label>" \
  --workspace .ops-gym
```

For non-installed domains, do not run the commands above until an adapter exists.

## Bundled Demo Arena Prompts

Football showcase:

```text
Use $opsgym. Run footballops-v0. Should the club rotate heavily or push starters through fixture congestion? Create Codex-authored agents, validate them, run 50 rollouts, and explain the injury-risk tradeoff.
```

Delivery:

```text
Use $opsgym. Run deliveryops-v0. Should the marketplace accept surge demand during rain or protect delivery SLA? Create Codex-authored agents, validate them, run 50 rollouts, and explain the winning strategy.
```

Hospital:

```text
Use $opsgym. Run hospitalops-v0. Should the hospital defer elective work to protect emergency capacity during a surge? Create Codex-authored agents, validate them, run 50 rollouts, and report the clinical-risk tradeoff.
```

## Recommended Demo Prompt: Agents Competing

The primary showcase is LLM-designed agents competing inside a simulation arena. Codex reads the environment brief, designs distinct decision agents, and the runner scores them through stochastic rollouts.

```text
Use $opsgym.

Business question:
<user's operational question here>

Tasks:
1. Draft a fast arena from the question and show me the summary.
2. Confirm the arena and create the environment.
3. Run a baseline policy tournament with 100 rollouts.
4. Read the environment brief and create 4 distinct decision agents with meaningfully different strategies.
5. Validate the agents and run the agent tournament with 100 rollouts.
6. Compare baseline vs agent results.
7. Explain which agent won and why — backed by the scorecard.
```

For any bundled adapter, Codex should:

1. Call `agent-brief` to generate a brief from the confirmed environment.
2. Read the brief and design agents that contrast on the 5 parameter levers.
3. Write the agent JSON, validate, and run.
4. Interpret the scoreboard and write the recommendation narrative.

The agents must be meaningfully different — not tiny variants. Good contrasts: resilience vs throughput, conservative vs aggressive, shock-absorbing vs peak-performance.

## Showcase (CLI)

Full arena-to-agents-to-comparison pipeline in one command, using built-in archetype agents:

```bash
./opsgym showcase
./opsgym showcase --arena deliveryops-v0
./opsgym showcase --arena hospitalops-v0
```

Runs the full pipeline: setup -> confirm -> env -> baseline tournament -> 4-agent tournament -> comparison. Three HTML reports are generated under `.ops-gym/reports/`. Works with any installed adapter.

## Recommended Demo Prompt: Baseline Only

```text
Use $opsgym.

Business question:
<user's operational question here>

Tasks:
1. Draft an arena and show me the summary.
2. Ask me for any critical edits before confirmation.
3. Confirm the arena, create the environment, and run 100 rollouts.
4. Add a relevant shock and rerun the gym.
5. Compare baseline vs shocked results and render a recommendation memo.
```

## References

- Read `references/adapter-interface.md` before moving arena logic into adapters.
- Read `references/state-machine.md` when changing `opsgym.json`, progress tracking, or init/list/status/next/loop behavior.
- Read `references/arena-schema.md` when creating or changing arena files.
- Read `references/env-schema.md` when creating or changing environment files.
- Read `references/run-schema.md` when changing run outputs.
- Read `references/comparison-schema.md` when adding baseline-vs-shock workflows.
- Read `references/scoring.md` when changing tournament metrics or evaluator behavior.

## Reporting Rules

- Lead with the winning policy and the measurable reason it won.
- Include the report path and run directory.
- If the arena was created or changed in this turn, include the arena summary path too.
- State major assumptions, especially if input files were missing or defaulted.
- If a shock changes the recommendation, call out the before/after shift.
