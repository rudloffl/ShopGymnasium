"""Plant configuration: a plain dataclass tree that round-trips to JSON (the Dash app edits these files).

Time unit is the minute everywhere in the simulation.
Every numeric default below is a placeholder until real plant data is available.
"""
from dataclasses import dataclass, field, asdict
import json


@dataclass
class ProductCfg:
    name: str
    stage1_time: float      # minutes per unit on a stage-1 machine
    stage2_time: float      # minutes per unit on a stage-2 machine
    finish_time: float      # minutes per finishing cycle
    finish_share: float     # share of finishing stations set up for this product


@dataclass
class StageCfg:
    n_machines: int
    n_components: int       # component types from prep consumed per unit (1 of each)
    changeover_time: float  # minutes to switch a machine to another product
    kit_batch: int          # units of one component type per prep delivery
    reorder_point: int      # a delivery is requested when a component count drops to this level
    mtbf: float             # mean operating minutes between failures (0 = never fails)
    mttr: float             # mean minutes to repair


@dataclass
class FinishingCfg:
    n_stations: int = 36
    slots: int = 2               # units loaded per cycle
    n_operators: int = 6         # dedicated finishing operators (load/unload)
    load_time: float = 2.0       # operator minutes per load, travel included
    mtbf: float = 1200.0
    mttr: float = 45.0


@dataclass
class PrepCfg:
    delay_min: float = 5.0       # triangular delivery delay (minutes)
    delay_mode: float = 15.0
    delay_max: float = 45.0
    disruption_prob: float = 0.05     # chance a delivery gets an extra delay
    disruption_min: float = 60.0
    disruption_max: float = 180.0


@dataclass
class OperatorCfg:
    n_operators: int = 25
    travel_time: float = 2.0     # minutes to walk to another machine
    patience: float = 10.0       # minutes an operator waits on a starved/blocked machine before moving on
    recheck: float = 5.0         # minutes an idle operator waits before looking for work again


@dataclass
class BufferCfg:
    stage1_capacity: int = 600         # intermediate buffer, all products together
    stage2_capacity: int = 800         # assembled buffer, all products together
    stage1_init_hours: float = 2.0     # initial stock drawn uniformly in [0, x] hours of finishing demand
    stage2_init_hours: float = 3.0


@dataclass
class RewardCfg:
    finished: float = 1.0        # per unit finished, normalised by nominal finishing capacity per hour
    starved: float = 1.0         # per fraction of station-time starved
    changeover: float = 0.02     # per changeover started


@dataclass
class PlantConfig:
    products: list = field(default_factory=list)
    stage1: StageCfg = None
    stage2: StageCfg = None
    finishing: FinishingCfg = field(default_factory=FinishingCfg)
    prep: PrepCfg = field(default_factory=PrepCfg)
    operators: OperatorCfg = field(default_factory=OperatorCfg)
    buffers: BufferCfg = field(default_factory=BufferCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    cycle_cv: float = 0.1            # coefficient of variation of cycle times (lognormal)
    decision_interval: float = 60.0  # minutes between agent decisions
    episode_hours: float = 72.0
    n_priorities: int = 4
    max_products: int = 8            # observation/action padding, lets one model serve several plants

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d['products'] = [ProductCfg(**p) for p in d['products']]
        for key, sub in (('stage1', StageCfg), ('stage2', StageCfg), ('finishing', FinishingCfg),
                         ('prep', PrepCfg), ('operators', OperatorCfg), ('buffers', BufferCfg),
                         ('reward', RewardCfg)):
            d[key] = sub(**d[key])
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def validate(self):
        assert 0 < len(self.products) <= self.max_products, 'product count must be in [1, max_products]'
        assert abs(sum(p.finish_share for p in self.products) - 1) < 1e-6, 'finish_share must sum to 1'
        assert self.n_priorities >= 1

    @property
    def n_machines(self):
        return self.stage1.n_machines + self.stage2.n_machines

    def nominal_finish_rate(self):
        """Units per hour the finishing stations can absorb when never starved."""
        f = self.finishing
        return sum(p.finish_share * f.n_stations * f.slots * 60 / p.finish_time for p in self.products)


def default_config():
    products = [
        ProductCfg('P1', stage1_time=4.0, stage2_time=2.0, finish_time=20.0, finish_share=0.25),
        ProductCfg('P2', stage1_time=4.5, stage2_time=2.2, finish_time=22.0, finish_share=0.20),
        ProductCfg('P3', stage1_time=3.5, stage2_time=1.8, finish_time=18.0, finish_share=0.20),
        ProductCfg('P4', stage1_time=4.2, stage2_time=2.0, finish_time=20.0, finish_share=0.15),
        ProductCfg('P5', stage1_time=3.8, stage2_time=2.1, finish_time=19.0, finish_share=0.10),
        ProductCfg('P6', stage1_time=4.4, stage2_time=1.9, finish_time=21.0, finish_share=0.10),
    ]
    cfg = PlantConfig(
        products=products,
        stage1=StageCfg(n_machines=50, n_components=5, changeover_time=15.0, kit_batch=30,
                        reorder_point=12, mtbf=600.0, mttr=30.0),
        stage2=StageCfg(n_machines=25, n_components=3, changeover_time=10.0, kit_batch=40,
                        reorder_point=25, mtbf=600.0, mttr=30.0),
    )
    cfg.validate()
    return cfg


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'D00_Plant/configs/default.json'
    default_config().save(path)
    print('wrote', path)
