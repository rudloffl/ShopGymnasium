import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .machines import Machine
from .stock import Stock
from .delivery import DeliveryQueue
from .market import Market


class WorkshopEnv(gym.Env):
    """
    Environnement Gymnasium pour l’atelier :
    - 1 step = 1 minute
    - épisode = 7 jours (10080 minutes)
    - production P1 (M1) et P2 (M1 puis M2)
    - commandes de MP avec délai (file de livraisons)
    - demande = backlog (demande résiduelle)
    - ventes agrégées 1 fois par heure (time % 60 == 0)
    - reward : uniquement lors des ventes (et coût MP)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.max_time = 7 * 24 * 60  # 10080 minutes
        self.raw_capacity = 50

        # Vol nocturne : 5 minutes avant minuit, consultable de l’extérieur
        self.theft_time = 1435

        # État interne
        self.time = 0
        self.demande_p1 = 0
        self.demande_p2 = 0

        # Modules internes
        self.m1 = Machine()
        self.m2 = Machine()
        self.stock = Stock(capacity=self.raw_capacity)
        self.delivery = DeliveryQueue()
        self.market = Market()

        # ---------------------------------------------------------
        # OBSERVATION SPACE
        # ---------------------------------------------------------
        low = np.array([
            0.0,  # time
            0.0, 0.0,  # M1 busy, time_left
            0.0, 0.0,  # M2 busy, time_left
            0.0,       # RAW
            0.0,       # P1
            0.0,       # P2_inter
            0.0,       # P2
            0.0,       # next_delivery_countdown
            0.0,       # backlog P1
            0.0,       # backlog P2
            0.0        # en_route (MP en transit)
        ], dtype=np.float32)

        high = np.array([
            float(self.max_time),
            1.0, 100.0,
            1.0, 100.0,
            float(self.raw_capacity),
            float(self.raw_capacity),
            float(self.raw_capacity),
            float(self.raw_capacity),
            10080.0,
            1000.0,
            1000.0,
            1000.0
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ---------------------------------------------------------
        # ACTION SPACE
        # ---------------------------------------------------------
        # 0–49    : produire P1 sur M1
        #           k = action + 1  (1 ≤ k ≤ 50)
        #           durée = 3 * k minutes, consomme k MP, produit k P1
        #
        # 50–99   : produire P2_inter (STEP1) sur M1
        #           k = action - 49 (1 ≤ k ≤ 50)
        #           durée = 10 * k minutes, consomme k MP, produit k P2_inter
        #
        # 100–149 : produire P2 (STEP2) sur M2
        #           k = action - 99 (1 ≤ k ≤ 50)
        #           durée = 15 * k minutes, consomme k P2_inter, produit k P2
        #
        # 150–199 : commander k unités de MP
        #           k = action - 149 (1 ≤ k ≤ 50)
        #
        # 200     : WAIT (ne rien faire)
        self.action_space = spaces.Discrete(201)

    # =============================================================
    # RESET
    # =============================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        self.demande_p1 = 0
        self.demande_p2 = 0

        self.m1.reset()
        self.m2.reset()
        self.stock.reset()
        self.delivery.reset()
        self.market = Market()

        return self._get_obs(), {}

    # =============================================================
    # STEP
    # =============================================================
    def step(self, action: int):

        reward = 0.0

        # ---------------------------------------------------------
        # 1) ACTION AGENT
        # ---------------------------------------------------------
        if action == 200:
            # WAIT : ne rien faire
            pass

        # -------------------- P1 sur M1 -------------------------
        elif 0 <= action <= 49:
            k = action + 1  # 1 à 50
            if (not self.m1.busy) and self.stock.consume_raw(k):
                # Durée = 3 minutes par unité de P1
                duration = 3 * k
                self.m1.start_batch(duration=duration, k=k, batch_type="P1_MULTI")

        # ----------------- P2 STEP1 sur M1 ----------------------
        elif 50 <= action <= 99:
            k = action - 49  # 1 à 50
            if (not self.m1.busy) and self.stock.consume_raw(k):
                # Durée = 10 minutes par unité de P2_inter
                duration = 10 * k
                self.m1.start_batch(duration=duration, k=k, batch_type="P2STEP1_MULTI")

        # ----------------- P2 STEP2 sur M2 ----------------------
        elif 100 <= action <= 149:
            k = action - 99  # 1 à 50
            if (not self.m2.busy) and self.stock.p2_inter >= k:
                # On consomme k unités de P2_inter d'un coup
                self.stock.p2_inter -= k
                # Durée = 15 minutes par unité de P2
                duration = 15 * k
                self.m2.start_batch(duration=duration, k=k, batch_type="P2STEP2_MULTI")

        # ----------------- Commande MP --------------------------
        elif 150 <= action <= 199:
            k = action - 149  # 1 à 50
            q = k
            reward -= float(q)

            # Jitter ± 2 minutes
            jitter = np.random.randint(-2, 3)
            arrival_time = self.time + 120 + jitter
            arrival_time = max(arrival_time, self.time + 1)

            self.delivery.schedule(q, arrival_time)

        # Toute valeur d'action hors [0,200] ne devrait pas arriver avec Discrete(201)

        # ---------------------------------------------------------
        # 2) MACHINES — AVANCEMENT
        # ---------------------------------------------------------
        if self.m1.tick():
            if self.m1.batch_type == "P1_MULTI":
                self.stock.add_p1(self.m1.batch_k)
            elif self.m1.batch_type == "P2STEP1_MULTI":
                self.stock.add_p2_inter(self.m1.batch_k)
            self.m1.reset_after_batch()

        if self.m2.tick():
            if self.m2.batch_type == "P2STEP2_MULTI":
                self.stock.add_p2(self.m2.batch_k)
            self.m2.reset_after_batch()

        # ---------------------------------------------------------
        # 3) LIVRAISONS
        # ---------------------------------------------------------
        delivered = self.delivery.tick(self.time)
        if delivered > 0:
            self.stock.add_raw(delivered)

        # ---------------------------------------------------------
        # 4) DEMANDE + VENTES (t % 60 == 0)
        # ---------------------------------------------------------
        if self.time > 0 and (self.time % 60 == 0):

            new_d1, new_d2 = self.market.sample_demand(self.time, 60)
            self.demande_p1 += int(new_d1)
            self.demande_p2 += int(new_d2)

            sold_p1, sold_p2 = self.market.compute_sales(
                self.stock, self.demande_p1, self.demande_p2
            )

            reward += 2.0 * sold_p1 + 20.0 * sold_p2
            self.demande_p1 -= sold_p1
            self.demande_p2 -= sold_p2

        # ---------------------------------------------------------
        # 5) INCRÉMENT DU TEMPS
        # ---------------------------------------------------------
        self.time += 1

        # ---------------------------------------------------------
        # 6) VOL NOCTURNE (5 min avant minuit)
        # ---------------------------------------------------------
        if self.time % 1440 == self.theft_time:
            self.market.apply_theft(self.stock)

        # ---------------------------------------------------------
        # 7) TERMINATION
        # ---------------------------------------------------------
        terminated = self.time >= self.max_time

        return self._get_obs(), reward, terminated, False, {}

    # =============================================================
    # OBSERVATION
    # =============================================================
    def _get_obs(self):

        if self.delivery.queue:
            next_t = min(t for (q, t) in self.delivery.queue)
            next_delivery_countdown = max(next_t - self.time, 0)
            q_total = sum(q for (q, t) in self.delivery.queue)
        else:
            next_delivery_countdown = 0
            q_total = 0

        return np.array([
            float(self.time),
            float(self.m1.busy),
            float(self.m1.time_left),
            float(self.m2.busy),
            float(self.m2.time_left),
            float(self.stock.raw),
            float(self.stock.p1),
            float(self.stock.p2_inter),
            float(self.stock.p2),
            float(next_delivery_countdown),
            float(self.demande_p1),
            float(self.demande_p2),
            float(q_total)
        ], dtype=np.float32)
