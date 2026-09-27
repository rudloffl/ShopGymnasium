# Progress log

Shared state between machines and Claude Code sessions. Read this at the start of a session and
update it before the session ends (see `CLAUDE.md`, "Working across machines").
Confidentiality rule in `CLAUDE.md` applies here too.

## Goal

A tool a real plant can use: set up a plant (Dash app), train a scheduling agent on its
simulation, and view the results (KPIs + a complete event log). One agent should ideally serve
several plants without retraining.

## Current state

- `D00_Plant/`: config-driven SimPy plant (prep → stage 1 → stage 2 → finishing, 25 operators on
  75 machines, randomness everywhere), a Gymnasium env with hourly decisions, a full event log, and
  baseline policies. 6 unit tests pass.
- `E00_MAPPO/`: shared per-machine policy (attention + GRU/LSTM, centralised critic) and a
  multi-agent PPO training loop. It runs end to end, but hasn't been tuned.
- `F00_Dash/`: plant setup app (stations, stage 1/2 machines, operators, buffers, products;
  save/load JSON), one-episode simulation with any policy, KPI cards, Gantt charts (stations,
  machines, operators), buffer levels, event log download. Training can't be launched from the
  app yet.
- All plant numbers are **placeholders** (`D00_Plant/configs/default.json`).
- Legacy toy shop (`B00_Agents`, `C00_DQNs`) is frozen. Its known bugs were not fixed.

### Baselines on the default plant (72 h episodes, 2 seeds, dev machine)

| Policy | Reward | Finished units/h | Stations starved | Changeovers |
|---|---|---|---|---|
| cover heuristic | 62.7 | 189 | 0.0 % | 8 |
| random | −39.6 | 115 | 39.5 % | 2470 |
| idle | −68.2 | 5 | 97.2 % | 0 |

Nominal finishing capacity is 217 units/h. The cover heuristic's gap to it comes from load time
and station breakdowns. With these placeholder numbers the plant is easy for the heuristic, so
there's little headroom for RL. Calibration with real data will decide how hard the problem really
is.

## Next session: first tests on the GPU machine

Planned by the user. Run in order, record results in "Training runs" and the session log:

1. `git pull`, then set up the env: CUDA `torch` build, `pip install -r requirements.txt`. Check
   `nvidia-smi`, `python -c "import torch; print(torch.cuda.is_available())"`, and `nproc`.
2. `python -m unittest discover tests` must pass.
3. `python -m D00_Plant.evaluate --episodes 2`: compare the numbers with the baselines table
   above. Same seeds should give the same KPIs on any machine.
4. Speed check: `python -m E00_MAPPO.train --run gpu_speed --iterations 5 --envs <nproc>` on
   `--device cuda`, then again on `--device cpu`. Note seconds per iteration for each. The
   simulation is expected to dominate, so choose `--envs` from the CPU core count.
5. First real run, e.g. `--run gpu1 --iterations 1000 --envs <nproc> --hours 24`. Track
   `eval_reward` vs `cover_baseline` in `runs/gpu1/metrics.csv`.
6. Optional: `python -m F00_Dash.app` there to view a trained policy (choose `trained: gpu1` in the
   policy list).

## Next steps

1. Get real plant data from the user to replace placeholders: product mix, cycle times,
   changeovers, finishing cycle and number of stations, operators, buffer sizes, breakdown and prep
   delay statistics, and how finishing setups change over time.
2. Train MAPPO seriously on the GPU machine (more envs and iterations), and tune it until it beats
   `cover`. The check run shows a random start is very far from `cover`, so the first thing to add
   is a warm start by imitating `cover` (behaviour cloning). Other ideas if it stalls:
   - local reward shaping per machine;
   - a smaller action space (drop priority, or add a "keep current plan" action).
3. Dash app, next parts: launch and monitor training from the app (background process, reads
   `runs/<run>/metrics.csv`); prep settings once they matter; compare two policies side by side.
4. Generalisation: randomise plant configs during training (machine and product counts, times),
   then test on unseen plants.

## Training runs

Record every run that matters here, because `runs/` is not pushed.

| Date | Machine | Run | Iterations × envs × hours | Key params | Eval reward (cover baseline) | Notes |
|---|---|---|---|---|---|---|
| 2026-09-27 | dev (CPU, 16 cores) | check1 | 80 × 12 × 24 h (3.3 min) | defaults: GRU 128, lr 3e-4, ent 0.01 | −10.9 (cover +20.8) | Learns slowly from random (train ep reward −11.7 → −9). Starts with thousands of changeovers from random choices. Needs a warm start and much longer training. |

## Decisions

- 2026-09-26: GitHub (`rudloffl/ShopGymnasium`) is the only channel between the dev machine (CPU)
  and the training machine (GPU). All context lives in the repo.
- 2026-09-27: The plant's industry and product are confidential (see CLAUDE.md). Generic
  vocabulary only.
- 2026-09-27: Keep SimPy. The simulation is the speed bottleneck, and it's scaled with parallel
  envs.
- 2026-09-27: Hourly decisions. For each machine, the agent chooses a product (or idle) and a
  priority. Operators follow the priority list and fall back on it between decisions.
- 2026-09-27: The agent is MAPPO with parameter sharing: one network per machine
  (machine context + attention over the other machines + shop context → GRU/LSTM), one team
  reward, a centralised critic, and PPO clipping per machine. It was chosen so one model can run
  plants of any size.
- 2026-09-27: The Dash app is a single file with plain Dash and Plotly. Gantt charts come from
  activity intervals recorded by the sim (`Plant.timeline`), not from parsing the event log. The
  user-facing labels use generic terms only.

## Session log

- 2026-09-26 (dev machine): README, CLAUDE.md, requirements.txt, this file.
- 2026-09-27 (dev machine): built D00_Plant (sim, env, event log, baselines, tests) and E00_MAPPO
  (model, training). Short check run on CPU, see training runs. Then started F00_Dash (setup +
  simulation + Gantt charts).
