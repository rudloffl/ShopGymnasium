# Shop Gymnasium

Deep reinforcement learning to schedule a production plant. The plant is a discrete-event
simulation ([SimPy](https://simpy.readthedocs.io/)) wrapped as a
[Gymnasium](https://gymnasium.farama.org/) environment. A single policy network, shared by every
machine, is trained with multi-agent PPO to decide every hour what each machine makes and how
urgent it is.

## The plant

```
prep (unlimited, random delays)
   │ 5 component types            │ 3 component types
   ▼                              ▼
stage 1 (50 slow machines) ─► stage-1 buffer ─► stage 2 (25 fast machines) ─► assembled buffer ─► finishing stations
```

- **Prep** is an unlimited supply of components. Each component type is delivered to a machine
  separately, with a random delay and occasional disruptions.
- **Stage 1** machines build a unit from 5 prep component types.
- **Stage 2** machines build a unit from 1 stage-1 unit of the same product plus 3 prep component
  types.
- **Finishing** stations are each set up for one product. It's the longest step and usually the
  bottleneck. A separate team of finishing operators loads the stations.
- **Operators:** 25 operators run the 75 building machines, so only one machine in three runs at a
  time. Operators have to move around to build enough stock to keep finishing fed.
- **Randomness:** cycle times, prep delivery delays and disruptions, machine and station
  breakdowns, and the initial stocks and machine setups.
- Changeovers cost time when a machine switches product.

Every hour the agent gives each machine a product (or "idle") and a priority. Operators take the
highest-priority machines. Between decisions they have some freedom: if their machine is starved,
blocked or broken, they move to the next machine on the priority list.

Everything is described by a JSON plant config (`D00_Plant/configs/default.json`). All numbers in it
are placeholders until real data is available.

### Event log
Every simulation event is logged as `(t, event, entity, product, qty, detail)`. Events include
component consumption, units started and finished, storage, starvation, blocking, operator
moves, changeovers, prep deliveries, breakdowns, and finishing loads. Use
`--events` in the evaluate script to get it as CSV.

## Repository layout

```
D00_Plant/        plant simulation (sim.py), gym env (env.py), config, event log, baseline policies
E00_MAPPO/        shared machine policy (model.py) and multi-agent PPO training (train.py)
tests/            unit tests for the plant
runs/             training outputs (gitignored)
PROGRESS.md       current state, next steps, training results, decisions
B00_Agents/, C00_DQNs/, A00 - Notebooks/   first toy shop with SB3 PPO (legacy)
```

## Setup

Use Python 3.11+. Install the `torch` build for your hardware first (CUDA or CPU, see
[pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
pip install -r requirements.txt
```

## Usage (from the repository root)

```bash
python -m unittest discover tests                                   # tests
python -m D00_Plant.evaluate --policy cover random idle --episodes 3  # baseline KPIs
python -m D00_Plant.evaluate --policy cover --episodes 1 --events runs/events.csv
python -m E00_MAPPO.train --run first --iterations 200 --envs 16      # training
```

Training writes `runs/<run>/metrics.csv`, `model.pt`, `best.pt` and a copy of the plant config.

Development and training happen on different machines, kept in sync through GitHub. See
`PROGRESS.md`.
