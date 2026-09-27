# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

A Deep RL sandbox. A tiny factory is simulated with SimPy and wrapped as a Gymnasium env
(`ShopEnv`), then trained with Stable-Baselines3 PPO. See `README.md` for the shop scenario, the
observation and action spaces, and the reward.

## Scope

- **Ignore `projet_atelier_fab/`.** It belongs to another contributor (merged from a fork). Don't
  read it, edit it, or use it as a reference.
- The current code is `B00_Agents/tinyshop3.py` (env) and `C00_DQNs/ppo2.py` (training).
  `tinyshop1/2` and `ppo1` are earlier snapshots kept for history. Don't edit them unless asked.
  The versioning pattern so far: when a change is substantial, copy the file to the next number
  instead of editing in place.
- `A00 - Notebooks/` holds the original exploration notebook (the directory name has spaces, so
  quote the path).

## Environment & commands

- Python via conda env `deeprl1` (gymnasium 1.2.2, simpy 4.1.1, torch 2.9 CPU). As of 2026-09,
  `stable_baselines3` is **not installed** in any local env, so `ppo2.py` needs
  `pip install stable-baselines3[extra] pandas matplotlib tensorboard` first.
- Run scripts from the repo root. `ppo2.py` adds the root to `sys.path` and imports
  `from B00_Agents.tinyshop3 import ShopEnv`.
- Train/eval: `python C00_DQNs/ppo2.py` (the `TRAIN` flag is at the bottom of the file).
- Smoke test without SB3:
  ```bash
  ~/miniconda3/envs/deeprl1/bin/python -c "
  from B00_Agents.tinyshop3 import ShopEnv
  env=ShopEnv(); o,_=env.reset(); d=False; R=0
  while not d:
      o,r,t,tr,_=env.step(env.action_space.sample()); R+=r; d=t or tr
  print(R)"
  ```
- No tests, no linter, no packaging (`requirements.txt`/`pyproject.toml`) yet.
- Generated artifacts (`ppo_shopenv/`, `logs/`, `*.zip`, `episode_*.png`) are gitignored. Don't
  commit them.

## How ShopEnv works (tinyshop3)

- **Time unit is hours.** `step()` runs the SimPy env until `now + step_size` (1 h). Cycle times
  in `prod_assignment` are minutes and get converted with `/60`.
- **Matrices are indexed `[product, machine]`.** `prod_assignment[i, j]` is the cycle time and `0`
  means product i doesn't use machine j. Routing is currently **hard-coded** in `make_products`
  (product 1 goes raw → M0 → intermediate stock → M1 → sell). `to_stock_prod` and `to_sell_prod`
  are defined but unused.
- The long-running SimPy processes are started in `_make_simpy_env`:
  `get_operators_to_work` (polls every minute and starts the queued batch with the highest
  `ranking_next`), `sell_products`, and `steal_product_at_night`. `make_products` holds an operator
  and a machine for a whole batch, and gives up after 60 min without input.
- Reward is accumulated through instance attributes that SimPy processes mutate during the hour
  (`salesrewards`, `poormanagementpenality`, `prod_stolen`). These are reset at the start of each
  `step()`.
- Actions come in as a flat vector in [-1, 1] (so PPO can use a single `Box`) and are decoded by
  `_parse_action` using `action_indices`. Observations are a `Dict` normalised to [0, 1], which
  requires SB3's `MultiInputPolicy`.
- `render()` returns a dict of event logs. Plotting happens in `ppo2.evaluate_ppo(render=True)`.

## Known issues in tinyshop3 (found while reading; not yet fixed)

- The observation shapes use `product_count + machine_count` where they should use
  `product_count * machine_count`. This only works because 2+2 == 2×2.
- `reset()` doesn't clear `prod_trace`, `prod_log`, `pending_raw`, `current_batch`, `next_batch`,
  `ranking_next`, or `forced_stop`, so state leaks across episodes.
- Randomness uses `random` and `np.random` instead of `self.np_random`, so `reset(seed=...)` isn't
  reproducible.
- `terminated` is computed before the sim advances, so episodes last 169 steps instead of 168.
- In `get_operators_to_work`, when the chosen machine is busy the queued batch is zeroed and lost.
  The block at lines ~437-442 duplicates the assignment above it.
- `steal_product_at_night` uses `now % 6 > 3`, which is not a real night window.
- The over-order penalty (`order_qty * 10`) is very large next to sales. Random-policy returns are
  around −58k per episode.
- The `__main__` manual-control block still reads removed keys (`stockraw_free`, etc.) and would
  crash.
- In `ppo2.py`, `eval_env = ShopEnv()` and the `Monitor` log path are shared with the training env.

## Conventions

- Match the existing style: plain numpy and SimPy, no extra abstractions, short comments.
  Some comments and commit messages are in French. Either language is fine.
- When changing the observation or action space, update `_get_obs` / `_parse_action` together with
  the space definitions, and check with `stable_baselines3.common.env_checker.check_env`.
