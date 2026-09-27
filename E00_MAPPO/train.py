"""MAPPO training: shared machine policy, one team reward, centralised critic.

python -m E00_MAPPO.train --run test --iterations 200 --envs 16
Outputs in runs/<run>/: metrics.csv (one row per iteration), model.pt (latest), best.pt, plant.json.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, AutoresetMode

from D00_Plant import PlantConfig, PlantEnv, default_config
from D00_Plant.env import machine_dim, shop_dim
from D00_Plant.evaluate import run_episode
from D00_Plant.policies import CoverPolicy
from .model import MachinePolicy, distributions, obs_to_tensors, TorchPolicy


def make_env(cfg, hours):
    return lambda: PlantEnv(cfg, episode_hours=hours)


def collect(venv, model, n_envs, n_machines, device, seed):
    """Run one full episode in every env. Returns stacked tensors with shape (T, B, ...)."""
    obs, _ = venv.reset(seed=seed)
    state = model.initial_state(n_envs, n_machines, device)
    buf = {k: [] for k in ('machines', 'shop', 'mask', 'a_prod', 'a_prio', 'logp', 'value', 'reward')}
    finished, starved, done = 0.0, 0.0, False
    with torch.no_grad():
        while not done:
            m, s, mask = obs_to_tensors(obs, device)
            pl, ql, value, state = model(m, s, mask, state)
            dp, dq = distributions(pl, ql)
            a_prod, a_prio = dp.sample(), dq.sample()
            logp = dp.log_prob(a_prod) + dq.log_prob(a_prio)
            action = torch.stack([a_prod[0], a_prio[0]], -1).reshape(n_envs, -1).cpu().numpy()
            obs, reward, terminated, truncated, info = venv.step(action)
            done = bool(np.all(terminated | truncated))
            for k, v in (('machines', m), ('shop', s), ('mask', mask), ('a_prod', a_prod), ('a_prio', a_prio),
                         ('logp', logp), ('value', value)):
                buf[k].append(v[0])
            buf['reward'].append(torch.as_tensor(reward, dtype=torch.float32, device=device))
            finished += np.mean(info['finished'])
            starved += np.mean(info['starved_share'])
    data = {k: torch.stack(v) for k, v in buf.items()}
    T = data['reward'].shape[0]
    stats = {'episode_reward': data['reward'].sum(0).mean().item(), 'finished_per_step': finished / T,
             'starved_share': starved / T}
    return data, stats


def gae(reward, value, gamma, lam):
    """reward, value (T,B); every sequence ends with a terminal step."""
    T = reward.shape[0]
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(reward[0])
    for t in reversed(range(T)):
        next_value = value[t + 1] if t + 1 < T else torch.zeros_like(value[t])
        delta = reward[t] + gamma * next_value - value[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + value


def update(model, opt, data, adv, returns, args, n_machines, device):
    B = adv.shape[1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = []
    for _ in range(args.epochs):
        for idx in torch.randperm(B, device=device).split(args.minibatch_envs):
            pl, ql, value, _ = model(data['machines'][:, idx], data['shop'][:, idx], data['mask'][:, idx],
                                     model.initial_state(len(idx), n_machines, device))
            dp, dq = distributions(pl, ql)
            logp = dp.log_prob(data['a_prod'][:, idx]) + dq.log_prob(data['a_prio'][:, idx])
            ratio = (logp - data['logp'][:, idx]).exp()                 # per-machine ratio
            a = adv[:, idx].unsqueeze(-1)                                  # team advantage for every machine
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            v_loss = (value - returns[:, idx]).pow(2).mean()
            entropy = (dp.entropy() + dq.entropy()).mean()
            loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            stats.append((pg.item(), v_loss.item(), entropy.item(), (ratio - 1).abs().mean().item()))
    return dict(zip(('pg_loss', 'v_loss', 'entropy', 'ratio_dev'), np.mean(stats, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='mappo')
    ap.add_argument('--config', help='plant config JSON (default: built-in default plant)')
    ap.add_argument('--hours', type=float, default=24, help='training episode length')
    ap.add_argument('--iterations', type=int, default=200)
    ap.add_argument('--envs', type=int, default=8)
    ap.add_argument('--sync', action='store_true', help='run envs in-process (debugging)')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--cell', choices=('gru', 'lstm'), default='gru')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatch-envs', type=int, default=4)
    ap.add_argument('--vf-coef', type=float, default=0.5)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--max-grad-norm', type=float, default=0.5)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cfg = PlantConfig.load(args.config) if args.config else default_config()
    out = os.path.join('runs', args.run)
    os.makedirs(out, exist_ok=True)
    cfg.save(os.path.join(out, 'plant.json'))
    device = torch.device(args.device)

    fns = [make_env(cfg, args.hours) for _ in range(args.envs)]
    venv = (SyncVectorEnv(fns, autoreset_mode=AutoresetMode.DISABLED) if args.sync
            else AsyncVectorEnv(fns, autoreset_mode=AutoresetMode.DISABLED))
    M = cfg.n_machines
    model_kwargs = dict(machine_dim=machine_dim(cfg), shop_dim=shop_dim(cfg), n_choices=cfg.max_products + 1,
                        n_priorities=cfg.n_priorities, hidden=args.hidden, cell=args.cell)
    model = MachinePolicy(**model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    baseline = run_episode(PlantEnv(cfg, episode_hours=args.hours), CoverPolicy(), seed=10_000)['reward']
    print(f'device {device} | cover-heuristic eval reward {baseline:.2f}')

    best = -np.inf
    with open(os.path.join(out, 'metrics.csv'), 'w', newline='') as f:
        writer = None
        for it in range(args.iterations):
            t0 = time.time()
            model.eval()
            data, stats = collect(venv, model, args.envs, M, device, seed=it * args.envs)
            adv, returns = gae(data['reward'], data['value'], args.gamma, args.lam)
            model.train()
            stats.update(update(model, opt, data, adv, returns, args, M, device))
            stats.update(iteration=it, seconds=time.time() - t0, cover_baseline=baseline, eval_reward='')

            if (it + 1) % args.eval_every == 0 or it == args.iterations - 1:
                policy = TorchPolicy(model, deterministic=True, device=device)
                stats['eval_reward'] = run_episode(PlantEnv(cfg, episode_hours=args.hours), policy, seed=10_000)['reward']
                ckpt = {'model_kwargs': model_kwargs, 'state_dict': model.state_dict(), 'iteration': it,
                        'eval_reward': stats['eval_reward'], 'args': vars(args)}
                torch.save(ckpt, os.path.join(out, 'model.pt'))
                if stats['eval_reward'] > best:
                    best = stats['eval_reward']
                    torch.save(ckpt, os.path.join(out, 'best.pt'))

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(stats))
                writer.writeheader()
            writer.writerow(stats)
            f.flush()
            print(f"it {it:4d} | ep reward {stats['episode_reward']:7.2f} | eval {stats['eval_reward']!s:>7.7} | "
                  f"starved {100 * stats['starved_share']:5.1f}% | entropy {stats['entropy']:.2f} | "
                  f"{stats['seconds']:.1f}s")
    venv.close()


if __name__ == '__main__':
    main()
