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
        # 0–49   → produire P1 (k = action + 1)
        # 50–99  → produire P2 STEP 1 (k = action - 49)
        # 100–149→ produire P2 STEP 2 (k = action - 99)
        # 150–199→ commander MP (q = action - 149)
        # 200    → WAIT
        self.action_space = spaces.Discrete(201)

        # -----------------------------------
        #  OBSERVATION SPACE (13 variables)
        # -----------------------------------
        # 0  : time
        # 1  : M1_busy
        # 2  : M1_time_left
        # 3  : M2_busy
        # 4  : M2_time_left
        # 5  : stock_raw
        # 6  : stock_p1
        # 7  : stock_p2_inter
        # 8  : stock_p2
        # 9  : next_delivery_countdown
        # 10 : backlog_p1 (demande résiduelle P1)
        # 11 : backlog_p2 (demande résiduelle P2)
        # 12 : q_total_en_route
        high = np.array([
            10080,  # time max = 7 jours
            1,      # M1_busy
            5000,   # M1_time_left
            1,      # M2_busy
            5000,   # M2_time_left
            50,     # stock_raw
            50,     # stock_p1
            50,     # stock_p2_inter
            50,     # stock_p2
            200,    # next delivery countdown
            500,    # backlog_p1 (borne large)
            500,    # backlog_p2 (borne large)
            500     # q_total_en_route (borne large)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.zeros(13, dtype=np.float32),
            high=high,
            dtype=np.float32
        )

        # Sous-modules
        self.m1 = Machine()
        self.m2 = Machine()
        self.stock = Stock()
        self.delivery = DeliveryQueue()
        self.market = Market()  # version jour/nuit

        self.time = 0
        self.demande_p1 = 0  # backlog P1
        self.demande_p2 = 0  # backlog P2
        self.next_delivery_time = 0

        self.reset()

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0

        self.m1 = Machine()
        self.m2 = Machine()
        self.stock = Stock()
        self.delivery = DeliveryQueue()
        self.market = Market()

        # Backlog initial nul
        self.demande_p1 = 0
        self.demande_p2 = 0

        self.next_delivery_time = 0

        return self._get_obs(), {}

    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):

        reward = 0.0

        # ---------------------------------------
        # 1) DÉCODAGE ACTION
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

        elif action == 200:
            action_type = "WAIT"

        else:
            raise ValueError(f"Action invalide : {action}. WAIT = 200 est la seule action hors intervalles.")

        # ---------------------------------------
        # 2) VALIDATION DES CONTRAINTES
        # ---------------------------------------

        if action_type in ["P1", "P2STEP1"] and self.m1.busy:
            action_type = "WAIT"

        if action_type == "P2STEP2" and self.m2.busy:
            action_type = "WAIT"

        if action_type in ["P1", "P2STEP1"] and action_type != "WAIT":
            if self.stock.raw < k:
                action_type = "WAIT"

        if action_type == "P2STEP2" and self.stock.p2_inter < k:
            action_type = "WAIT"

        # ---------------------------------------
        # 3) LANCEMENT DES ACTIONS
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
            reward -= q
            self.next_delivery_time = self.time + 120
            self.delivery.schedule(q, self.next_delivery_time)

        # WAIT ne fait rien

        # ---------------------------------------
        # 4) AVANCEMENT DU TEMPS (1 minute)
        # ---------------------------------------

        self.time += 1

        m1_finished = self.m1.tick()
        m2_finished = self.m2.tick()

        # ---------------------------------------
        # 5) BATCHS TERMINÉS → MISE À JOUR STOCKS
        # ---------------------------------------

        if m1_finished:
            if self.m1.batch_type == "P1":
                self.stock.add_p1(self.m1.batch_k)
            elif self.m1.batch_type == "P2STEP1":
                self.stock.add_p2_inter(self.m1.batch_k)
            self.m1.reset_after_batch()

        if m2_finished:
            if self.m2.batch_type == "P2STEP2":
                self.stock.add_p2(self.m2.batch_k)
            self.m2.reset_after_batch()

        # ---------------------------------------
        # 6) LIVRAISONS
        # ---------------------------------------

        delivered = self.delivery.tick(self.time)
        if delivered > 0:
            self.stock.add_raw(delivered)

        # ---------------------------------------
        # 7) DEMANDE & VENTES (toutes les heures, backlog)
        # ---------------------------------------
        # → self.demande_p1 / p2 = carnet de commandes non satisfaites

        if self.time % 60 == 0 and self.time > 0:

            # 7.1 Nouvelle demande horaire
            new_d1, new_d2 = self.market.sample_demand(
                time_minute=self.time,
                period_minutes=60
            )

            # Ajout au backlog
            self.demande_p1 += new_d1
            self.demande_p2 += new_d2

            # 7.2 Ventes : on sert le backlog avec le stock dispo
            revenue, sold_p1, sold_p2 = self.market.compute_sales(
                self.stock,
                self.demande_p1,
                self.demande_p2
            )

            reward += revenue

            # 7.3 Mise à jour du backlog (demande résiduelle)
            self.demande_p1 -= sold_p1
            self.demande_p2 -= sold_p2

        # IMPORTANT : on ne remet plus jamais la demande à 0,
        # elle représente le backlog global.

        # ---------------------------------------
        # 8) VOL NOCTURNE (chaque 1440 minutes)
        # ---------------------------------------

        if self.time % 1440 == 0 and self.time > 0:
            self.stock.p1 = int(self.stock.p1 * 0.9)
            self.stock.p2 = int(self.stock.p2 * 0.9)

        # ---------------------------------------
        # 9) FIN D'ÉPISODE (7 jours)
        # ---------------------------------------

        done = self.time >= 10080   # 7 jours

        return self._get_obs(), reward, done, False, {}

    # ============================================================
    # OBSERVATION
    # ============================================================
    def _get_obs(self):

        # Total d’unités commandées mais non livrées
        q_total_en_route = float(sum(q for (q, _) in self.delivery.queue))

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

            float(self.next_delivery_time - self.time
                  if self.next_delivery_time > self.time else 0),

            float(self.demande_p1),   # backlog P1
            float(self.demande_p2),   # backlog P2

            q_total_en_route
        ], dtype=np.float32)
