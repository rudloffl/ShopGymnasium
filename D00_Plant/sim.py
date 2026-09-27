"""SimPy model of the plant: prep -> stage 1 -> intermediate buffer -> stage 2 -> assembled buffer -> finishing.

- Prep is an unlimited supply; each component type is delivered to a machine separately, with a random delay.
- Stage-1 units need 1 of each stage-1 component type. Stage-2 units need 1 stage-1 unit of the same product
  plus 1 of each stage-2 component type.
- A building machine only runs with an operator. The agent sets each machine's product (or idle) and priority;
  operators take the highest-priority machines and fall back to the next one when theirs starves or breaks.
- Finishing stations are each set up for one product and loaded by a separate pool of finishing operators.

Time unit: minute. No gym code here, see env.py.
"""
import math
import numpy as np
import simpy

from .eventlog import EventLog

# Machine / station status
IDLE, WAIT_OP, CHANGEOVER, RUNNING, STARVED_COMP, STARVED_INPUT, BLOCKED, BROKEN = range(8)
STATUS_NAMES = ('idle', 'wait_op', 'changeover', 'running', 'starved_comp', 'starved_input', 'blocked', 'broken')
N_STATUS = len(STATUS_NAMES)


class Buffer:
    """Per-product stock sharing one capacity. A slot is reserved when a unit starts, so it always has room when done."""

    def __init__(self, env, name, n_products, capacity):
        self.env, self.name, self.capacity = env, name, capacity
        self.level = np.zeros(n_products, dtype=int)
        self.reserved = 0
        self._item_waiters = [[] for _ in range(n_products)]
        self._space_waiters = []

    def has_space(self):
        return self.level.sum() + self.reserved < self.capacity

    def reserve(self):
        self.reserved += 1

    def put_reserved(self, p):
        self.reserved -= 1
        self.level[p] += 1
        self._wake(self._item_waiters[p])

    def take(self, p, n=1):
        self.level[p] -= n
        self._wake(self._space_waiters)

    def item_event(self, p):
        ev = self.env.event()
        self._item_waiters[p].append(ev)
        return ev

    def space_event(self):
        ev = self.env.event()
        self._space_waiters.append(ev)
        return ev

    @staticmethod
    def _wake(waiters):
        for ev in waiters:
            if not ev.triggered:
                ev.succeed()
        waiters.clear()


class Entity:
    """Anything with a status whose time share is tracked per decision interval."""

    def __init__(self, name):
        self.name = name
        self.status = IDLE
        self.status_since = 0.0
        self.status_time = np.zeros(N_STATUS)
        self.produced = 0          # units completed during the current interval
        self.ttf = math.inf        # operating minutes left before the next failure
        self.broken = False


class Machine(Entity):
    def __init__(self, idx, stage, name, n_components):
        super().__init__(name)
        self.idx, self.stage = idx, stage
        self.product = None        # product the machine is set up for
        self.target = None         # product requested by the agent (None = idle)
        self.priority = 0
        self.operator = None       # operator at the machine
        self.claimed = None        # operator walking to the machine
        self.release = False       # operator must leave after the current unit
        self.comp = np.zeros(n_components, dtype=int)
        self.comp_product = None   # product the component stock belongs to
        self.in_flight = np.zeros(n_components, dtype=bool)
        self.wake = None


class Station(Entity):
    def __init__(self, idx, name, product):
        super().__init__(name)
        self.idx, self.product = idx, product


class Operator:
    def __init__(self, idx, name):
        self.idx, self.name = idx, name
        self.machine = None
        self.status = 'idle'       # idle | travel | work
        self.wake = None


class Plant:
    def __init__(self, cfg, rng, log=None):
        self.cfg, self.rng = cfg, rng
        self.log = log if log is not None else EventLog(enabled=False)
        self.env = simpy.Environment()
        self.n_products = len(cfg.products)

        self.inter = Buffer(self.env, 'stage1', self.n_products, cfg.buffers.stage1_capacity)
        self.assembled = Buffer(self.env, 'stage2', self.n_products, cfg.buffers.stage2_capacity)

        n1, n2 = cfg.stage1.n_machines, cfg.stage2.n_machines
        self.machines = ([Machine(i, 1, f'S1-{i:02d}', cfg.stage1.n_components) for i in range(n1)] +
                         [Machine(n1 + i, 2, f'S2-{i:02d}', cfg.stage2.n_components) for i in range(n2)])
        self.operators = [Operator(k, f'OP-{k:02d}') for k in range(cfg.operators.n_operators)]
        self.stations = [Station(k, f'F-{k:02d}', p) for k, p in enumerate(self._station_products())]
        self.loaders = simpy.Resource(self.env, capacity=cfg.finishing.n_operators)
        self._dispatch_ev = self.env.event()

        self.finished = np.zeros(self.n_products, dtype=int)   # cumulative
        self.changeovers = 0                                    # cumulative

        self._init_state()
        for m in self.machines:
            self.env.process(self._machine_proc(m))
        for op in self.operators:
            self.env.process(self._operator_proc(op))
        for s in self.stations:
            self.env.process(self._station_proc(s))

    # ------------------------------------------------------------------ setup
    def _station_products(self):
        """Split stations between products by finish_share (largest remainder)."""
        n = self.cfg.finishing.n_stations
        shares = np.array([p.finish_share for p in self.cfg.products]) * n
        counts = np.floor(shares).astype(int)
        for p in np.argsort(counts - shares)[:n - counts.sum()]:
            counts[p] += 1
        return [p for p, c in enumerate(counts) for _ in range(c)]

    def demand_rate(self, include_broken=True):
        """Units per hour each product's finishing stations absorb."""
        f = self.cfg.finishing
        rate = np.zeros(self.n_products)
        for s in self.stations:
            if include_broken or not s.broken:
                rate[s.product] += f.slots * 60 / self.cfg.products[s.product].finish_time
        return rate

    def _init_state(self):
        b, rng = self.cfg.buffers, self.rng
        demand = self.demand_rate()
        for buf, hours in ((self.inter, b.stage1_init_hours), (self.assembled, b.stage2_init_hours)):
            level = np.rint(rng.uniform(0, hours, self.n_products) * demand).astype(int)
            if level.sum() > buf.capacity:
                level = np.floor(level * buf.capacity / level.sum()).astype(int)
            buf.level[:] = level
        for m in self.machines:
            st = self._stage(m)
            m.product = m.comp_product = int(rng.integers(self.n_products))
            m.comp[:] = rng.integers(0, st.kit_batch + 1, st.n_components)
            m.ttf = rng.exponential(st.mtbf) if st.mtbf > 0 else math.inf
        for s in self.stations:
            s.ttf = rng.exponential(self.cfg.finishing.mtbf) if self.cfg.finishing.mtbf > 0 else math.inf

    def _stage(self, m):
        return self.cfg.stage1 if m.stage == 1 else self.cfg.stage2

    # ------------------------------------------------------------------ helpers
    def _cycle(self, mean):
        cv = self.cfg.cycle_cv
        if cv <= 0:
            return mean
        sigma = math.sqrt(math.log(1 + cv ** 2))
        return float(self.rng.lognormal(math.log(mean) - sigma ** 2 / 2, sigma))

    def _pname(self, p):
        return '' if p is None else self.cfg.products[p].name

    def _set_status(self, e, status):
        now = self.env.now
        e.status_time[e.status] += now - e.status_since
        e.status, e.status_since = status, now

    def _wake(self, m):
        if m.wake is not None and not m.wake.triggered:
            m.wake.succeed()

    def _dispatch(self):
        if not self._dispatch_ev.triggered:
            self._dispatch_ev.succeed()
        self._dispatch_ev = self.env.event()

    def flush_interval(self):
        """Close status timers; return (status_time[entities, N_STATUS], produced[entities]) for the machines
        and stations since the previous call, then reset the counters."""
        out = []
        for group in (self.machines, self.stations):
            for e in group:
                self._set_status(e, e.status)
            times = np.array([e.status_time for e in group])
            produced = np.array([e.produced for e in group])
            for e in group:
                e.status_time[:] = 0
                e.produced = 0
            out.append((times, produced))
        return out

    # ------------------------------------------------------------------ agent interface
    def apply_decision(self, targets, priorities):
        """targets[i]: product index for machine i, or -1 for idle. priorities[i]: 0 (low) .. n_priorities-1."""
        now = self.env.now
        for m, t, pr in zip(self.machines, targets, priorities):
            t = None if t < 0 else int(t)
            pr = int(pr)
            if t != m.target or pr != m.priority:
                self.log(now, 'decision', m.name, self._pname(t), 0, f'prio={pr}')
            m.target, m.priority = t, pr
            m.release = False
            if t is not None and m.comp_product != t:
                # Prep starts preparing the new product's components right away
                m.comp_product = t
                m.comp[:] = 0
                m.in_flight[:] = False
                self._request_components(m)
            self._wake(m)

        # Move operators from low-priority machines to higher-priority ones that are waiting;
        # idle operators take the top of the waiting list first.
        waiting = sorted((m for m in self.machines if self._available(m)), key=lambda m: -m.priority)
        n_idle = sum(op.machine is None and op.status == 'idle' for op in self.operators)
        staffed = sorted((m for m in self.machines if m.operator is not None and m.target is not None),
                         key=lambda m: m.priority)
        for w, s in zip(waiting[n_idle:], staffed):
            if w.priority <= s.priority:
                break
            s.release = True
            self._wake(s)
        self._dispatch()

    def run_until(self, t):
        self.env.run(until=t)

    # ------------------------------------------------------------------ operators
    def _available(self, m):
        """Machine requested, free of operators, and likely able to run."""
        if m.target is None or m.broken or m.operator is not None or m.claimed is not None:
            return False
        if m.product == m.target and (m.comp_product != m.target or (m.comp <= 0).any()):
            return False     # set up but components missing: an operator would only wait
        if m.stage == 2 and self.inter.level[m.target] <= 0:
            return False
        out = self.inter if m.stage == 1 else self.assembled
        return out.has_space()

    def _pick_task(self):
        best, best_key = None, None
        for m in self.machines:
            if self._available(m):
                key = (m.priority, m.stage, m.product == m.target, -m.idx)   # ties: downstream first
                if best_key is None or key > best_key:
                    best, best_key = m, key
        return best

    def _operator_proc(self, op):
        env, ocfg = self.env, self.cfg.operators
        while True:
            if op.machine is None:
                m = self._pick_task()
                if m is None:
                    if op.status != 'idle':
                        self.log(env.now, 'op_idle', op.name)
                    op.status = 'idle'
                    yield env.any_of([self._dispatch_ev, env.timeout(ocfg.recheck)])
                    continue
                m.claimed = op
                op.status = 'travel'
                self.log(env.now, 'op_dispatch', op.name, self._pname(m.target), 0, m.name)
                yield env.timeout(ocfg.travel_time)
                m.claimed = None
                if m.broken or m.target is None or m.operator is not None:
                    continue   # task vanished while walking
                m.operator, op.machine, op.status = op, m, 'work'
                self.log(env.now, 'op_arrive', op.name, self._pname(m.target), 0, m.name)
                self._wake(m)
            op.wake = env.event()
            yield op.wake

    def _release(self, m, reason):
        op = m.operator
        m.operator, m.release = None, False
        op.machine, op.status = None, 'idle'
        self.log(self.env.now, 'op_release', op.name, self._pname(m.product), 0, f'{m.name}: {reason}')
        if op.wake is not None and not op.wake.triggered:
            op.wake.succeed()

    # ------------------------------------------------------------------ prep
    def _request_components(self, m):
        st = self._stage(m)
        for c in range(st.n_components):
            if m.comp[c] <= st.reorder_point and not m.in_flight[c]:
                m.in_flight[c] = True
                self.log(self.env.now, 'kit_request', m.name, self._pname(m.comp_product), st.kit_batch, f'comp{c}')
                self.env.process(self._delivery_proc(m, c, m.comp_product, st.kit_batch))

    def _delivery_proc(self, m, c, product, qty):
        pc = self.cfg.prep
        delay = self.rng.triangular(pc.delay_min, pc.delay_mode, pc.delay_max)
        if self.rng.random() < pc.disruption_prob:
            delay += self.rng.uniform(pc.disruption_min, pc.disruption_max)
        yield self.env.timeout(delay)
        if m.comp_product != product:
            self.log(self.env.now, 'kit_discarded', m.name, self._pname(product), qty, f'comp{c}')
            return
        m.comp[c] += qty
        m.in_flight[c] = False
        self.log(self.env.now, 'kit_delivered', 'prep', self._pname(product), qty, f'{m.name} comp{c}')
        self._wake(m)
        self._request_components(m)

    # ------------------------------------------------------------------ building machines
    def _machine_proc(self, m):
        env = self.env
        st = self._stage(m)
        out = self.inter if m.stage == 1 else self.assembled
        while True:
            if m.operator is None:
                self._set_status(m, WAIT_OP if m.target is not None else IDLE)
                m.wake = env.event()
                yield m.wake
                continue
            if m.target is None or m.release:
                self._release(m, 'reassigned' if m.target is not None else 'machine set idle')
                continue
            if m.product != m.target:
                self._set_status(m, CHANGEOVER)
                self.changeovers += 1
                self.log(env.now, 'changeover_start', m.name, self._pname(m.target), 0, f'from {self._pname(m.product)}')
                yield env.timeout(st.changeover_time)
                m.product = m.target
                self.log(env.now, 'changeover_end', m.name, self._pname(m.product))
                continue

            res = yield from self._wait_inputs(m, out)
            if res == 'timeout':
                self._release(m, STATUS_NAMES[m.status])
                continue
            if res == 'changed':
                continue

            p = m.product
            m.comp -= 1
            self.log(env.now, 'consume', m.name, self._pname(p), 1, 'prep components')
            if m.stage == 2:
                self.inter.take(p)
                self.log(env.now, 'consume', m.name, self._pname(p), 1, 'stage1 buffer')
            out.reserve()
            self._request_components(m)
            self._set_status(m, RUNNING)
            prod = self.cfg.products[p]
            dur = self._cycle(prod.stage1_time if m.stage == 1 else prod.stage2_time)
            self.log(env.now, 'unit_start', m.name, self._pname(p), 1)
            yield env.timeout(dur)
            out.put_reserved(p)
            m.produced += 1
            self.log(env.now, 'unit_done', m.name, self._pname(p), 1)
            self.log(env.now, 'store', m.name, self._pname(p), 1, out.name)

            m.ttf -= dur
            if m.ttf <= 0:
                self.log(env.now, 'breakdown', m.name, self._pname(p))
                m.broken = True
                self._release(m, 'breakdown')
                self._set_status(m, BROKEN)
                yield env.timeout(self.rng.exponential(st.mttr))
                m.broken = False
                m.ttf = self.rng.exponential(st.mtbf)
                self.log(env.now, 'repaired', m.name)
                self._dispatch()

    def _wait_inputs(self, m, out):
        """Wait until components, stage-1 input (stage 2) and output space are all there.
        Returns 'ok', 'changed' (the agent or a release changed the plan) or 'timeout' (operator patience ran out)."""
        env = self.env
        deadline = env.now + self.cfg.operators.patience
        while True:
            p = m.product
            if m.target != p or m.release or m.operator is None:
                return 'changed'
            if m.comp_product != p or (m.comp <= 0).any():
                missing, what = STARVED_COMP, 'components ' + ','.join(f'comp{c}' for c in np.flatnonzero(m.comp <= 0))
            elif m.stage == 2 and self.inter.level[p] <= 0:
                missing, what = STARVED_INPUT, 'stage1 buffer empty'
            elif not out.has_space():
                missing, what = BLOCKED, f'{out.name} buffer full'
            else:
                return 'ok'
            if m.status != missing:
                self.log(env.now, 'blocked' if missing == BLOCKED else 'starved', m.name, self._pname(p), 0, what)
                self._set_status(m, missing)
            remaining = deadline - env.now
            if remaining <= 0:
                return 'timeout'
            m.wake = env.event()
            events = [m.wake, env.timeout(remaining)]
            if missing == STARVED_INPUT:
                events.append(self.inter.item_event(p))
            elif missing == BLOCKED:
                events.append(out.space_event())
            yield env.any_of(events)

    # ------------------------------------------------------------------ finishing
    def _station_proc(self, s):
        env, f = self.env, self.cfg.finishing
        p = s.product
        pname = self._pname(p)
        while True:
            if self.assembled.level[p] < f.slots:
                if s.status != STARVED_INPUT:
                    self.log(env.now, 'starved', s.name, pname, 0, 'stage2 buffer empty')
                    self._set_status(s, STARVED_INPUT)
                yield self.assembled.item_event(p)
                continue
            self.assembled.take(p, f.slots)
            self.log(env.now, 'consume', s.name, pname, f.slots, 'stage2 buffer')
            self._set_status(s, WAIT_OP)
            with self.loaders.request() as req:
                yield req
                yield env.timeout(f.load_time)
            self.log(env.now, 'load', s.name, pname, f.slots)
            self._set_status(s, RUNNING)
            dur = self._cycle(self.cfg.products[p].finish_time)
            yield env.timeout(dur)
            self.finished[p] += f.slots
            s.produced += f.slots
            self.log(env.now, 'finish_done', s.name, pname, f.slots)

            s.ttf -= dur
            if s.ttf <= 0:
                self.log(env.now, 'breakdown', s.name, pname)
                s.broken = True
                self._set_status(s, BROKEN)
                yield env.timeout(self.rng.exponential(f.mttr))
                s.broken = False
                s.ttf = self.rng.exponential(f.mtbf)
                self.log(env.now, 'repaired', s.name, pname)
