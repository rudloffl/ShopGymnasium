import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .machines import Machine
from .stock import Stock
from .delivery import DeliveryQueue
from .market import Market


class WorkshopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # -----------------------------------
        #  ACTION SPACE (201 actions)
        # -----------------------------------
        self.action_space = spaces.Discrete(201)

        # -----------------------------------
        #  OBSERVATION SPACE (12 variables)
        # -----------------------------------
        high = np.array([
            1440,   # time
            1,      # M1_busy
            1000,   # M1_time_left
            1,      # M2_busy
            1000,   # M2_time_left
            50,     # stock_raw
            50,     # stock_p1
            50,     # stock_p2_inter
            50,     # stock_p2
            200,    # next_delivery_time
            50,     # demande_p1
            50      # demande_p2
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.zeros(12, dtype=np.float32),
            high=high,
            dtype=np.float32
        )

        # Sous-modules
        self.m1 = Machine()
        self.m2 = Machine()
        self.stock = Stock()
        self.delivery = DeliveryQueue()
        self.market = Market()

        self.time = 0
        self.demande_p1 = 0
        self.demande_p2 = 0
        self.next_delivery_time = 0

        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0

        self.m1 = Machine()
        self.m2 = Machine()
        self.stock = Stock()
        self.delivery = DeliveryQueue()
        self.market = Market()

        self.demande_p1 = 0
        self.demande_p2 = 0
        self.next_delivery_time = 0

        return self._get_obs(), {}


    def step(self, action):

        reward = 0

        # ---------------------------------------
        # 1) DÉCODAGE + VALIDATION
        # ---------------------------------------

        if 0 <= action <= 49:
            action_type = "P1"
            k = action + 1

        elif 50 <= action <= 99:
            action_type = "P2STEP1"
            k = action - 49

        elif 100 <= action <= 149:
            action_type = "P2STEP2"
            k = action - 99

        elif 150 <= action <= 199:
            action_type = "ORDER"
            q = action - 149

        else:
            action_type = "WAIT"

        # Vérification M1
        if action_type in ["P1", "P2STEP1"] and self.m1.busy:
            action_type = "WAIT"

        # Vérification M2
        if action_type == "P2STEP2" and self.m2.busy:
            action_type = "WAIT"

        # Vérification stock
        if action_type in ["P1", "P2STEP1"] and self.stock.raw < k:
            action_type = "WAIT"

        if action_type == "P2STEP2" and self.stock.p2_inter < k:
            action_type = "WAIT"

        # ---------------------------------------
        # 2) EXECUTION ACTION
        # ---------------------------------------

        if action_type == "P1":
            duration = 3 * k
            self.stock.consume_raw(k)
            self.m1.start_batch(duration, k, "P1")

        elif action_type == "P2STEP1":
            duration = 10 * k
            self.stock.consume_raw(k)
            self.m1.start_batch(duration, k, "P2STEP1")

        elif action_type == "P2STEP2":
            duration = 15 * k
            self.stock.consume_p2_inter(k)
            self.m2.start_batch(duration, k, "P2STEP2")

        elif action_type == "ORDER":
            reward -= q   # prix MP = 1
            self.next_delivery_time = self.time + 120
            self.delivery.schedule(q, self.next_delivery_time)

        # WAIT → rien à faire

        # ---------------------------------------
        # 3) AVANCEMENT DU TEMPS D’UNE MINUTE
        # ---------------------------------------

        self.time += 1

        m1_finished = self.m1.tick()
        m2_finished = self.m2.tick()

        # ---------------------------------------
        # 4) RECUPERATION DES PRODUCTIONS FINIES
        # ---------------------------------------

        if m1_finished:
            if self.m1.batch_type == "P1":
                self.stock.add_p1(self.m1.batch_k)
            elif self.m1.batch_type == "P2STEP1":
                self.stock.add_p2_inter(self.m1.batch_k)

        if m2_finished:
            if self.m2.batch_type == "P2STEP2":
                self.stock.add_p2(self.m2.batch_k)

        # ---------------------------------------
        # 5) LIVRAISONS MP
        # ---------------------------------------

        delivered = self.delivery.tick(self.time)
        if delivered > 0:
            self.stock.add_raw(delivered)

        # ---------------------------------------
        # 6) DEMANDE & VENTES
        # ---------------------------------------

        self.demande_p1, self.demande_p2 = self.market.sample_demand()
        reward += self.market.compute_sales(
            self.stock,
            self.demande_p1,
            self.demande_p2
        )

        # ---------------------------------------
        # 7) VOL NOCTURNE
        # ---------------------------------------

        if self.time == 1440:
            self.stock.p1 = int(self.stock.p1 * 0.9)
            self.stock.p2 = int(self.stock.p2 * 0.9)

        # ---------------------------------------
        # 8) FIN EPISODE
        # ---------------------------------------

        done = self.time >= 1440

        return self._get_obs(), reward, done, False, {}


    def _get_obs(self):
        return np.array([
            self.time,

            # Machines
            float(self.m1.busy),
            float(self.m1.time_left),
            float(self.m2.busy),
            float(self.m2.time_left),

            # Stocks
            float(self.stock.raw),
            float(self.stock.p1),
            float(self.stock.p2_inter),
            float(self.stock.p2),

            # Approvisionnement
            float(self.next_delivery_time - self.time
                  if self.next_delivery_time > self.time else 0),

            # Demande minute courante
            float(self.demande_p1),
            float(self.demande_p2),

        ], dtype=np.float32)
