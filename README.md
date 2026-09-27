# Shop Gymnasium

A small manufacturing shop, simulated with [SimPy](https://simpy.readthedocs.io/) and exposed as a
[Gymnasium](https://gymnasium.farama.org/) environment. A Deep RL agent (PPO from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)) learns to run the shop: it decides
what to produce, in which order and batch size, and when to reorder raw material.

## The shop

| Item | Default |
|---|---|
| Operators | 2 (shared across machines) |
| Machines | 2 (`machine_0`, `machine_1`), capacity 1 each |
| Product A (`'0'`) | 1 step: machine 0 (2 min/unit), sells for 2 |
| Product B (`'1'`) | 2 steps: machine 0 (4 min/unit) → intermediate stock → machine 1 (5 min/unit), sells for 20 |
| Raw stock | Container, capacity 100, random initial level. Reorders arrive after 1 h |
| Intermediate stock | capacity 50 |
| Sell stock | capacity 50. Everything in it is sold every 5 h |
| Theft | stock above a threshold of 50 units can be stolen (10 % chance per unit) |
| Episode | 7 days at 1 decision per simulated hour (168 steps) |

Each decision step advances the SimPy clock by one hour. Inside that hour, operators pick up queued
batches by priority, machines consume inputs and produce units, and sales and theft happen.

### Observation (`Dict`, all values normalised to [0, 1])
Stock fill levels (raw, intermediate, sell), current batch remaining, queued next batch and its
priority ranking, busy operators, routing and cycle times (`prod_assignment`), pending raw-material
delivery, and time of day.

### Action (flat `Box(-1, 1, shape=(17,))`)
With 2 products × 2 machines the action splits into:
- `current_batch` (4): new size for the running batch (0–100), applied only when forced
- `force_current_batch` (4): `> 0` stops the running batch and overrides its size (costs −5)
- `next_batch` (4): size of the next batch to queue per (product, machine) (0–100)
- `ranking_next` (4): priority of each queued batch (the highest is started first)
- `order_raw_prod` (1): raw material to order (0–100). Orders of 10 or less are ignored

### Reward
+ sale value of every unit sold, +0.1 per unit produced
− 50 for queuing a batch on a machine that the product doesn't use
− 5 for each forced batch change
− 10 × quantity for ordering more raw material than the stock can hold
− 3 per stolen unit

## Repository layout

```
A00 - Notebooks/A00-FirstAgent.ipynb   first exploration: scenario, env prototype, hand-written PPO
B00_Agents/tinyshop{1,2,3}.py          successive versions of the ShopEnv (3 = current)
C00_DQNs/ppo{1,2}.py                   SB3 PPO training/evaluation scripts (2 = current, uses tinyshop3)
projet_atelier_fab/                    separate contributor's project, not part of this work
```

## Setup

```bash
conda activate deeprl1   # has gymnasium, simpy, torch
pip install stable-baselines3[extra] pandas matplotlib tensorboard
```

## Usage

Train and evaluate. Run this from the repository root, because outputs are written to the current
directory:

```bash
python C00_DQNs/ppo2.py
```

- Set `TRAIN = False` in `ppo2.py` to evaluate an existing `./ppo_shopenv/best_model` without
  retraining.
- Checkpoints go to `./ppo_shopenv/`, TensorBoard logs to `./logs/`, and per-episode plots to
  `episode_*.png` (machine Gantt chart, sales, stock levels, orders, theft, production). All of these
  are gitignored.
- To follow training: `tensorboard --logdir=./logs`

Quick sanity check of the environment:

```python
from B00_Agents.tinyshop3 import ShopEnv
env = ShopEnv(duration_max=7)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.render()   # returns a dict of logs (it does not draw)
```

## Status

This is an early prototype. The environment runs and PPO trains on it, but the reward shaping and
the dynamics are still being tuned. Known issues are listed in `CLAUDE.md`.
