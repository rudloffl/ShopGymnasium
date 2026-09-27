"""Run baseline policies on a plant and print KPIs; optionally write the event log of the first episode.

python -m D00_Plant.evaluate --policy cover random idle --episodes 3 --events runs/events.csv
"""
import argparse
import time
import numpy as np

from .config import PlantConfig, default_config
from .env import PlantEnv
from .policies import POLICIES


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(env, obs))
        done = terminated or truncated
    return info['kpi']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='plant config JSON (default: built-in default plant)')
    ap.add_argument('--policy', nargs='+', default=['cover', 'random', 'idle'], choices=list(POLICIES))
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--hours', type=float, help='episode length override')
    ap.add_argument('--events', help='CSV path for the event log of the first episode of the first policy')
    args = ap.parse_args()

    cfg = PlantConfig.load(args.config) if args.config else default_config()
    print(f'nominal finishing capacity: {cfg.nominal_finish_rate():.0f} units/h')
    for i, name in enumerate(args.policy):
        policy = POLICIES[name]()
        kpis, t0 = [], time.time()
        for ep in range(args.episodes):
            log = args.events is not None and i == 0 and ep == 0
            env = PlantEnv(cfg, log_events=log, episode_hours=args.hours)
            kpis.append(run_episode(env, policy, seed=ep))
            if log:
                env.plant.log.to_csv(args.events)
                print(f'  wrote {len(env.plant.log.rows)} events to {args.events}')
        hours = env.episode_hours
        fin = np.array([k['finished'] for k in kpis])
        print(f'{name:>7}: reward {np.mean([k["reward"] for k in kpis]):7.2f} | '
              f'finished {fin.mean() / hours:6.1f} units/h | '
              f'stations starved {100 * np.mean([k["starved_share"] for k in kpis]):5.1f}% | '
              f'changeovers {np.mean([k["changeovers"] for k in kpis]):6.0f} | '
              f'{(time.time() - t0) / args.episodes:.1f} s/episode')


if __name__ == '__main__':
    main()
