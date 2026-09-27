# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Deep RL for scheduling a real production plant. The plant is simulated with SimPy
(`D00_Plant/`), exposed as a Gymnasium env, and controlled by a shared per-machine policy trained
with multi-agent PPO (`E00_MAPPO/`). A Dash app to set up plants, launch training and view results
is planned. See `README.md` for the plant model and `PROGRESS.md` for where things stand.

## Confidentiality (hard rule)

The plant models a real industrial process. **Its industry and product are confidential.** Never
name them or hint at them anywhere: code, comments, docs, commit messages, run names, event logs,
generated data, or Claude's local memory. Use only the generic vocabulary:
prep (component supply) → stage 1 → stage-1 buffer → stage 2 → stage-2 (assembled) buffer →
finishing stations, products `P1..Pn`, components `comp0..`. If you're unsure whether a word is
revealing, use a generic one.

## Scope

- **Ignore `projet_atelier_fab/`.** It belongs to another contributor (merged from a fork). Don't
  read it, edit it, or use it as a reference.
- **Current code:** `D00_Plant/` (plant model, env, baselines, event log), `E00_MAPPO/` (model,
  training) and `F00_Dash/` (setup & simulation app).
- **Legacy (kept for history, don't edit unless asked):** `B00_Agents/tinyshop*.py` and
  `C00_DQNs/ppo*.py` (the first 2-machine toy shop with SB3 PPO), and `A00 - Notebooks/` (quote the
  path, it has spaces). New major steps get a new `X00_` folder.

## Working across machines

The code is written on one machine and trained on another one that has a GPU. GitHub
(`git@github.com:rudloffl/ShopGymnasium.git`, branch `main`) is the only link between them.
Claude's local memory (`~/.claude`) does **not** travel, so anything a later session needs must be
written into the repo.

- **At session start:** `git pull`, then read `PROGRESS.md` (current state, next steps, past
  training runs, decisions).
- **Before the session ends:** update `PROGRESS.md` (session log entry with date and machine, new
  next steps, and a row in the training-runs table for any run that matters). Commit it together
  with the code, and offer to push. Push only when the user confirms.
- `runs/` (checkpoints, metrics, event logs) is gitignored and stays on the machine that made it.
  Write the numbers that matter into `PROGRESS.md`. Only commit a specific model if the user asks.
- Standing preferences from the user belong in this file, not in local memory.
- Don't assume paths, conda env names, or hardware. Check `python --version`,
  `nvidia-smi`, and `torch.cuda.is_available()` when it matters.

## Environment & commands

- Setup: Python 3.11+ env (conda or venv). Install the right `torch` build for the machine (CUDA on
  the GPU box, see pytorch.org), then `pip install -r requirements.txt`. Keep `requirements.txt`
  updated when adding an import.
  - Dev machine (as of 2026-09): conda env `deeprl1`, Python 3.14, torch CPU. No pandas, no SB3.
- Everything runs from the repo root as modules:
  - Tests: `python -m unittest discover tests` (about 1 s)
  - Baselines + KPIs: `python -m D00_Plant.evaluate --policy cover random idle --episodes 3 [--events runs/events.csv]`
  - Training: `python -m E00_MAPPO.train --run <name> --iterations 200 --envs 16 --hours 24`
    writes `runs/<name>/metrics.csv`, `model.pt`, `best.pt`, `plant.json`
  - Regenerate the default plant JSON: `python -m D00_Plant.config`
  - Dash app: `python -m F00_Dash.app` → http://127.0.0.1:8050 (saves plants to `D00_Plant/configs/`,
    writes the last event log to `runs/dash/events.csv`)
- Speed: the simulation, not the network, is the bottleneck (about 0.7 s per simulated 24 h with
  75 machines). Scale with `--envs` (one process per env). The GPU mostly helps the update phase.

## How the plant model works (D00_Plant)

- **Time unit is the minute.** One env step = `decision_interval` (60 min).
- `config.py`: dataclass tree ↔ JSON. Every default number is a placeholder until real plant data
  arrives. `PlantConfig.validate()` checks product shares.
- `sim.py` (`Plant`), no gym code:
  - Each building machine runs a process loop: wait for an operator → changeover if the target
    product changed → wait for inputs (`_wait_inputs`, bounded by operator `patience`) → consume →
    run → store → maybe break down (failures count operating time only).
  - `Buffer` holds per-product levels with a shared capacity. An output slot is reserved when a unit
    starts, so storing never blocks mid-cycle.
  - Operators: `_pick_task` takes the highest-priority available machine (ties: stage 2 first, then
    already set up). At each decision `apply_decision` also moves operators from low-priority
    machines to higher-priority waiting ones, effective after the current unit.
  - Prep: one delivery process per component type, triangular delay plus random disruptions.
    Deliveries for a product the machine switched away from are discarded.
  - Finishing: each station is set up for one product for the whole episode, takes `slots` units
    per cycle, and needs a finishing operator (`loaders` resource) to load.
  - All randomness goes through `rng` (the env's `np_random`), so `reset(seed=...)` is reproducible
    (there's a test).
- `env.py` (`PlantEnv`): `machines` (M × feature matrix), `shop` (per-product blocks padded to
  `max_products` + global features), and `product_mask`. The action is `MultiDiscrete` with
  [product choice (0 = idle), priority] per machine. The reward is finished units normalised by
  nominal capacity, minus the starved station-time share, minus changeovers.
- `eventlog.py`: every sim event → `(t, event, entity, product, qty, detail)`. It's off during
  training (~10^5 rows per 72 h).
- `policies.py`: `CoverPolicy` (stock-cover heuristic), `RandomPolicy`, `IdlePolicy`. The RL agent
  must beat `cover`.
- When changing observation features, update `machine_dim` / `shop_dim` and `_get_obs` together.
  Checkpoints trained on the old dims become unusable.

## How the agent works (E00_MAPPO)

- `MachinePolicy`: the same weights for every machine. Machine encoder + attention over all
  machines + shop encoder → GRU/LSTM per machine → product head (masked) + priority head.
  A centralised critic pools the machines. Nothing depends on the machine count.
- `train.py`: all envs reset together and run full episodes, so the recurrent state never resets
  mid-sequence. GAE is computed on the team reward. PPO ratio and clipping are per machine, with
  the same team advantage broadcast to every machine. The deterministic eval reward is compared
  with the `cover` heuristic (`cover_baseline` column).

## Dash app (F00_Dash/app.py)

- One file, plain Dash + Plotly (no pandas, no bootstrap). The left panel edits a plant config:
  finishing stations, stage 1/2 machines, operators, buffers, products. Prep and reward settings
  are not in the form and are kept from the loaded file (`base-config` store).
- `form_to_config` validates the form and rescales station shares to 100 %. `capacity_summary`
  gives a quick bottleneck estimate before simulating.
- "Run simulation" runs one episode with a baseline or a trained `runs/*/best.pt` policy. It shows
  KPI cards, Gantt charts for stations, machines and operators, and buffer levels.
- The Gantt charts come from `Plant.timeline` (activity intervals, recorded only when the event
  log is on; call `close_timeline()` before reading). Buffer curves come from `Plant.samples`
  (every 10 min).
- To check the UI visually without a browser session:
  `google-chrome --headless=new --screenshot=out.png --window-size=1600,1100 --virtual-time-budget=8000 http://127.0.0.1:8050/`,
  or write a figure to HTML (`fig.write_html`) and screenshot that.

## Conventions

- Match the existing style: plain numpy, SimPy and torch, no extra frameworks, short comments.
  Some comments and commit messages are in French. Either language is fine.
- Keep the simulation free of RL code and the env free of torch.
