# Progress log

Shared state between machines and Claude Code sessions. Read this at the start of a session and
update it before the session ends (see `CLAUDE.md`, "Working across machines").

## Current state

- Env: `B00_Agents/tinyshop3.py`. Training: `C00_DQNs/ppo2.py` (SB3 PPO, `MultiInputPolicy`).
- The env runs end to end. A random policy scores about −58k per 169-step episode, mostly from the
  over-order penalty.
- The known bugs listed in `CLAUDE.md` are **not fixed yet**.
- No trained model is in the repo. Checkpoints are gitignored, so each machine has its own.

## Next steps

- Waiting for the user to choose the next steps after reviewing the docs.

## Training runs

Record every run that matters here, because logs and checkpoints are not pushed.

| Date | Machine | Env / script | Timesteps | Key params | Mean eval reward | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Decisions

- 2026-09-26: GitHub (`rudloffl/ShopGymnasium`) is the only channel for sharing work between the
  dev machine (CPU) and the training machine (GPU). All context has to live in the repo.

## Session log

- 2026-09-26 (dev machine): wrote README, CLAUDE.md, requirements.txt, and this file. No code
  changes.
