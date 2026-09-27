"""Run with: python -m unittest discover tests"""
import os
import tempfile
import unittest
from collections import Counter

import numpy as np
from gymnasium.utils.env_checker import check_env

from D00_Plant import PlantConfig, PlantEnv, default_config
from D00_Plant.policies import CoverPolicy, IdlePolicy


def run(env, policy, seed=0):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(env, obs))
        done = terminated or truncated
    return info['kpi']


class TestPlant(unittest.TestCase):
    def test_config_roundtrip(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'plant.json')
            cfg.save(path)
            self.assertEqual(PlantConfig.load(path).to_dict(), cfg.to_dict())

    def test_gym_api(self):
        check_env(PlantEnv(episode_hours=3), skip_render_check=True)

    def test_seed_is_reproducible(self):
        a = run(PlantEnv(episode_hours=6), CoverPolicy(), seed=3)
        b = run(PlantEnv(episode_hours=6), CoverPolicy(), seed=3)
        self.assertEqual(a, b)

    def test_idle_plant_builds_nothing(self):
        env = PlantEnv(log_events=True, episode_hours=4)
        run(env, IdlePolicy())
        events = Counter(r[1] for r in env.plant.log.rows)
        self.assertEqual(events['unit_done'], 0)

    def test_stage1_buffer_is_conserved(self):
        env = PlantEnv(log_events=True, episode_hours=8)
        env.reset(seed=1)
        init = env.plant.inter.level.sum()
        policy = CoverPolicy()
        obs, done = None, False
        while not done:
            obs, _, done, _, _ = env.step(policy(env, obs))
        rows = env.plant.log.rows
        stored = sum(1 for r in rows if r[1] == 'store' and r[5] == 'stage1')
        taken = sum(1 for r in rows if r[1] == 'consume' and r[5] == 'stage1 buffer')
        self.assertGreater(stored, 0)
        self.assertEqual(init + stored - taken, env.plant.inter.level.sum())

    def test_cover_beats_idle(self):
        cover = run(PlantEnv(episode_hours=12), CoverPolicy())
        idle = run(PlantEnv(episode_hours=12), IdlePolicy())
        self.assertGreater(cover['reward'], idle['reward'])


if __name__ == '__main__':
    unittest.main()
