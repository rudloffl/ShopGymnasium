"""Baseline policies. A learned agent is only useful if it beats these.

Each policy is a callable (env, obs) -> flat action for PlantEnv.
"""
import numpy as np


class IdlePolicy:
    """Every machine idle: lower bound (finishing only consumes the initial stock)."""

    def __call__(self, env, obs):
        return env.encode_action(np.full(env.cfg.n_machines, -1), np.zeros(env.cfg.n_machines, int))


class RandomPolicy:
    def __call__(self, env, obs):
        return env.action_space.sample()


class CoverPolicy:
    """Staff machines for the products with the least downstream cover.

    Operators are split between stages in proportion to the stage cycle times, so both stages have the same
    throughput. Machine slots go one by one to the product with the lowest projected cover (hours of
    finishing demand in stock); earlier slots get higher priority. Machines already set up for a product are
    reused first to avoid changeovers. `spare` requests a few extra machines as fallback for operators.
    """

    def __init__(self, spare=1.2):
        self.spare = spare

    def __call__(self, env, obs):
        cfg, plant = env.cfg, env.plant
        P = len(cfg.products)
        demand = plant.demand_rate(include_broken=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            cover2 = np.where(demand > 0, plant.assembled.level / demand, np.inf)
            cover1 = np.where(demand > 0, (plant.inter.level + plant.assembled.level) / demand, np.inf)

        w = demand / max(demand.sum(), 1e-9)
        t1 = sum(w[p] * cfg.products[p].stage1_time for p in range(P))
        t2 = sum(w[p] * cfg.products[p].stage2_time for p in range(P))
        n_ops = cfg.operators.n_operators
        ops1 = int(round(n_ops * t1 / (t1 + t2)))
        ops2 = n_ops - ops1

        targets = np.full(cfg.n_machines, -1)
        priorities = np.zeros(cfg.n_machines, int)
        n1 = cfg.stage1.n_machines
        for stage, ops, cover, machines in ((2, ops2, cover2, plant.machines[n1:]),
                                            (1, ops1, cover1, plant.machines[:n1])):
            times = np.array([p.stage1_time if stage == 1 else p.stage2_time for p in cfg.products])
            gain = np.where(demand > 0, (60 / times) / np.maximum(demand, 1e-9), np.inf)
            n_slots = min(len(machines), int(np.ceil(ops * self.spare)))
            order = self._allocate(n_slots, cover, gain)
            free = [m for m in machines if not m.broken]
            for rank, p in enumerate(order):
                if not free:
                    break
                m = next((m for m in free if m.product == p), free[0])
                free.remove(m)
                targets[m.idx] = p
                priorities[m.idx] = cfg.n_priorities - 1 - rank * cfg.n_priorities // max(len(order), 1)
        return env.encode_action(targets, priorities)

    @staticmethod
    def _allocate(n_slots, cover, gain):
        assigned = np.zeros(len(cover))
        order = []
        for _ in range(n_slots):
            projected = cover + assigned * gain
            if np.isinf(projected).all():
                break
            p = int(np.argmin(projected))
            assigned[p] += 1
            order.append(p)
        return order


POLICIES = {'idle': IdlePolicy, 'random': RandomPolicy, 'cover': CoverPolicy}
