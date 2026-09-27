"""Gymnasium wrapper around the SimPy plant: one step = one decision interval (1 h by default).

Observation (Dict):
  machines      (n_machines, MACHINE_DIM)   one row per building machine, values in [0, 1]
  shop          (SHOP_DIM,)                 per-product block (padded to max_products) + global features
  product_mask  (max_products + 1,)         1 where the product choice is valid (index 0 = idle, always valid)
Action (MultiDiscrete, flat): for each machine i, [product choice, priority]
  product choice 0 = idle, k >= 1 = product k-1 (choices beyond the real products are treated as idle)
Reward per step: finished units / nominal finishing capacity - starved station-time share - changeover cost.
"""
from typing import Optional
import numpy as np
import gymnasium as gym

from .config import default_config
from .eventlog import EventLog
from .sim import Plant, N_STATUS, RUNNING, STARVED_INPUT, BROKEN

PRODUCT_FEATURES = 11
GLOBAL_FEATURES = 8


def machine_dim(cfg):
    return 2 + N_STATUS + 2 * (cfg.max_products + 1) + 5


def shop_dim(cfg):
    return PRODUCT_FEATURES * cfg.max_products + GLOBAL_FEATURES


class PlantEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg=None, log_events=False, episode_hours=None):
        self.cfg = cfg if cfg is not None else default_config()
        self.log_events = log_events
        self.episode_hours = episode_hours or self.cfg.episode_hours
        self.interval = self.cfg.decision_interval
        self.n_steps = int(round(self.episode_hours * 60 / self.interval))

        K, M, P = self.cfg.max_products, self.cfg.n_machines, self.cfg.n_priorities
        self.observation_space = gym.spaces.Dict({
            'machines': gym.spaces.Box(0, 1, (M, machine_dim(self.cfg)), np.float32),
            'shop': gym.spaces.Box(-1, 1, (shop_dim(self.cfg),), np.float32),
            'product_mask': gym.spaces.Box(0, 1, (K + 1,), np.float32),
        })
        self.action_space = gym.spaces.MultiDiscrete(np.tile([K + 1, P], M))
        self.plant = None

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.plant = Plant(self.cfg, self.np_random, EventLog(self.log_events))
        self.t = 0
        self.kpi = {k: 0.0 for k in ('finished', 'starved_share', 'changeovers', 'reward')}
        self._last = self._empty_interval()
        return self._get_obs(), {}

    def step(self, action):
        targets, priorities = self.decode_action(action)
        self.plant.apply_decision(targets, priorities)
        changeovers0, finished0 = self.plant.changeovers, self.plant.finished.sum()

        self.t += 1
        self.plant.run_until(self.t * self.interval)
        (m_time, m_prod), (s_time, s_prod) = self.plant.flush_interval()
        self._last = {'m_time': m_time, 'm_prod': m_prod, 's_time': s_time}

        finished = self.plant.finished.sum() - finished0
        changeovers = self.plant.changeovers - changeovers0
        starved_share = s_time[:, STARVED_INPUT].sum() / (len(self.plant.stations) * self.interval)
        rc = self.cfg.reward
        nominal = self.cfg.nominal_finish_rate() * self.interval / 60
        reward = (rc.finished * finished / nominal
                  - rc.starved * starved_share
                  - rc.changeover * changeovers)

        self.kpi['finished'] += finished
        self.kpi['starved_share'] += starved_share / self.n_steps
        self.kpi['changeovers'] += changeovers
        self.kpi['reward'] += reward
        info = {'finished': int(finished), 'starved_share': float(starved_share),
                'changeovers': int(changeovers), 'kpi': dict(self.kpi)}
        terminated = self.t >= self.n_steps
        return self._get_obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ actions
    def decode_action(self, action):
        a = np.asarray(action, dtype=int).reshape(self.cfg.n_machines, 2)
        choice = a[:, 0]
        targets = np.where((choice >= 1) & (choice <= len(self.cfg.products)), choice - 1, -1)
        priorities = np.clip(a[:, 1], 0, self.cfg.n_priorities - 1)
        return targets, priorities

    def encode_action(self, targets, priorities):
        """Inverse of decode_action: targets (-1 = idle), priorities -> flat MultiDiscrete action."""
        return np.stack([np.asarray(targets) + 1, np.asarray(priorities)], axis=1).reshape(-1)

    # ------------------------------------------------------------------ observations
    def _empty_interval(self):
        return {'m_time': np.zeros((self.cfg.n_machines, N_STATUS)),
                'm_prod': np.zeros(self.cfg.n_machines),
                's_time': np.zeros((self.cfg.finishing.n_stations, N_STATUS))}

    def _get_obs(self):
        cfg, plant = self.cfg, self.plant
        K, P = cfg.max_products, cfg.n_priorities
        n1 = cfg.stage1.n_machines

        rows = np.zeros((cfg.n_machines, machine_dim(cfg)), np.float32)
        for i, m in enumerate(plant.machines):
            st = cfg.stage1 if m.stage == 1 else cfg.stage2
            fastest = min(p.stage1_time if m.stage == 1 else p.stage2_time for p in cfg.products)
            c = 0
            rows[i, c + m.stage - 1] = 1; c += 2
            rows[i, c + m.status] = 1; c += N_STATUS
            rows[i, c + (0 if m.product is None else m.product + 1)] = 1; c += K + 1
            rows[i, c + (0 if m.target is None else m.target + 1)] = 1; c += K + 1
            rows[i, c] = m.operator is not None or m.claimed is not None
            rows[i, c + 1] = m.priority / max(P - 1, 1)
            rows[i, c + 2] = min(m.comp.min() / st.kit_batch, 1.0)
            rows[i, c + 3] = min(self._last['m_prod'][i] * fastest / self.interval, 1.0)
            rows[i, c + 4] = self._last['m_time'][i, RUNNING] / self.interval

        shop = np.zeros(shop_dim(cfg), np.float32)
        demand = plant.demand_rate()
        horizon = 8.0   # hours of cover mapped to 1
        station_prod = np.array([s.product for s in plant.stations])
        targets = np.array([-1 if m.target is None else m.target for m in plant.machines])
        for p, prod in enumerate(cfg.products):
            b = p * PRODUCT_FEATURES
            d = max(demand[p], 1e-6)
            n_st = (station_prod == p).sum()
            starved = sum(s.status == STARVED_INPUT for s in plant.stations if s.product == p)
            shop[b:b + PRODUCT_FEATURES] = (
                min(plant.inter.level[p] / d / horizon, 1),
                min(plant.assembled.level[p] / d / horizon, 1),
                plant.inter.level[p] / plant.inter.capacity,
                plant.assembled.level[p] / plant.assembled.capacity,
                n_st / len(plant.stations),
                starved / max(n_st, 1),
                (targets[:n1] == p).sum() / n1,
                (targets[n1:] == p).sum() / cfg.stage2.n_machines,
                prod.stage1_time / 10, prod.stage2_time / 10, prod.finish_time / 60,
            )
        now = plant.env.now
        g = PRODUCT_FEATURES * K
        shop[g:] = (
            np.sin(2 * np.pi * now / 1440), np.cos(2 * np.pi * now / 1440),
            self.t / self.n_steps,
            sum(op.machine is None and op.status == 'idle' for op in plant.operators) / len(plant.operators),
            plant.loaders.count / plant.loaders.capacity,
            plant.inter.level.sum() / plant.inter.capacity,
            plant.assembled.level.sum() / plant.assembled.capacity,
            sum(s.status == BROKEN for s in plant.stations) / len(plant.stations),
        )

        mask = np.zeros(K + 1, np.float32)
        mask[:len(cfg.products) + 1] = 1
        return {'machines': np.clip(rows, 0, 1), 'shop': np.clip(shop, -1, 1), 'product_mask': mask}
