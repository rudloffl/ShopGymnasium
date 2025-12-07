
# ================================================================
# WORKSHOP ENVIRONMENT — FINAL VERSION (OPTION C, FULLY COMMENTED)
# Production au fil de l'eau + commentaires pédagogiques complets
# ================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .machines import Machine
from .stock import Stock
from .delivery import DeliveryQueue
from .market import Market


class WorkshopEnv(gym.Env):
    """
    ============================================================
    ENVIRONNEMENT ATELIER — VERSION EXPLICATIVE ET STRUCTURÉE
    ============================================================

    MODELISATION GÉNÉRALE
    ----------------------
    - L'environnement simule un atelier industriel minute par minute.
    - Un épisode complet dure 7 jours : 7 × 24 × 60 = 10 080 minutes.
    - Deux machines :
        M1 : P1 et P2_STEP1
        M2 : P2_STEP2
    - Stock de matières premières (raw), P1, P2_inter, P2.
    - Commandes de matières premières avec délai.
    - Demande client toutes les 15 minutes.
    - Système de backlog pénalisant.
    - Production « au fil de l'eau » : 1 unité visible dès qu'elle est produite.

    OBJECTIFS DE L'AGENT
    ---------------------
    - Produire la bonne quantité au bon moment.
    - Minimiser le backlog (pénalité chaque minute).
    - Honorer la demande pour gagner des récompenses de vente.
    - Optimiser l'utilisation des machines et des stocks.

    STRUCTURE DU STEP()
    --------------------
    1) Traitement de l'action (production / commande / attente)
    2) Avancement des machines minute par minute + production unitaire
    3) Livraison potentielle de matières premières
    4) Passage du temps (+1 min)
    5) Demande + ventes (toutes les 15 minutes)
    6) Vol nocturne (1 fois par jour)
    7) Pénalité backlog
    8) Construction de l'observation
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Durée maximale d'un épisode
        self.max_time = 7 * 24 * 60  # 10 080 minutes

        # Capacité des stocks
        self.raw_capacity = 50

        # Vol planifié chaque jour (minute 1435 = 23h55)
        self.theft_time = 1435

        # Initialisation du temps et des backlogs
        self.time = 0
        self.demande_p1 = 0
        self.demande_p2 = 0

        # Machines
        self.m1 = Machine()  # production P1 + STEP1
        self.m2 = Machine()  # production STEP2

        # Différents modules
        self.stock = Stock(capacity=self.raw_capacity)
        self.delivery = DeliveryQueue()
        self.market = Market()

        # -----------------------------------------------------------
        # ESPACE D'OBSERVATION (13 DIMENSIONS)
        # -----------------------------------------------------------
        low = np.zeros(13, dtype=np.float32)
        high = np.array([
            float(self.max_time),  # minute courante
            1.0, 100.0,            # état M1
            1.0, 100.0,            # état M2
            float(self.raw_capacity), float(self.raw_capacity),
            float(self.raw_capacity), float(self.raw_capacity),
            10080.0,               # délai prochaine livraison
            1000.0, 1000.0, 1000.0 # backlog + MP en transit
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # -----------------------------------------------------------
        # ESPACE D'ACTIONS (201 ACTIONS)
        # -----------------------------------------------------------
        # 0–49    → Produire P1 (k = a+1)
        # 50–99   → Produire P2_STEP1 (k = a-49)
        # 100–149 → Produire P2_STEP2 (k = a-99)
        # 150–199 → Commander MP (k = a-149)
        # 200     → WAIT
        self.action_space = spaces.Discrete(201)

    # ================================================================
    # RESET DE L'ÉPISODE
    # ================================================================
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

    # ================================================================
    # ACTION MASK (OPTIONS PERMISES À L'AGENT)
    # ================================================================
    def get_action_mask(self):
        mask = np.ones(201, dtype=bool)

        if self.m1.busy:
            mask[0:100] = False

        if self.m2.busy:
            mask[100:150] = False

        for a in range(0, 100):
            k = (a + 1) if a < 50 else (a - 49)
            if self.stock.raw < k:
                mask[a] = False

        for a in range(100, 150):
            k = a - 99
            if self.stock.p2_inter < k:
                mask[a] = False

        return mask

    # ================================================================
    # STEP — UNE MINUTE DE SIMULATION
    # ================================================================
    def step(self, action: int):

        reward = 0.0

        # -----------------------------------------------------------
        # 1) TRAITEMENT DE L'ACTION
        # -----------------------------------------------------------

        # Action WAIT
        if action == 200:
            reward -= 0.2  # légère pénalité

        # Production P1
        elif 0 <= action <= 49:
            k = action + 1
            if (not self.m1.busy) and self.stock.consume_raw(k):
                duration = 3 * k  # durée totale
                self.m1.start_batch(duration=duration, k=k, batch_type="P1_MULTI")
                reward += 0.5 * k
            else:
                reward -= 1

        # Production P2 — Étape 1
        elif 50 <= action <= 99:
            k = action - 49
            if (not self.m1.busy) and self.stock.consume_raw(k):
                duration = 10 * k
                self.m1.start_batch(duration=duration, k=k, batch_type="P2STEP1_MULTI")
                reward += 5 * k
            else:
                reward -= 1

        # Production P2 — Étape 2
        elif 100 <= action <= 149:
            k = action - 99
            if (not self.m2.busy) and self.stock.p2_inter >= k:
                self.stock.p2_inter -= k
                duration = 15 * k
                self.m2.start_batch(duration=duration, k=k, batch_type="P2STEP2_MULTI")
                reward += 15 * k
            else:
                reward -= 1

        # Commande MP
        elif 150 <= action <= 199:
            k = action - 149
            reward -= float(k)
            jitter = np.random.randint(-2, 3)
            arrival_time = max(self.time + 1, self.time + 120 + jitter)
            self.delivery.schedule(k, arrival_time)

        # -----------------------------------------------------------
        # 2) PRODUCTION AU FIL DE L'EAU — MACHINE M1
        # -----------------------------------------------------------
        res_m1 = self.m1.tick()

        if res_m1 in ("unit", "last_unit"):
            if self.m1.batch_type == "P1_MULTI":
                self.stock.add_p1(1)
            elif self.m1.batch_type == "P2STEP1_MULTI":
                self.stock.add_p2_inter(1)

            if res_m1 == "last_unit":
                self.m1.reset_after_batch()

        # -----------------------------------------------------------
        # 3) PRODUCTION AU FIL DE L'EAU — MACHINE M2
        # -----------------------------------------------------------
        res_m2 = self.m2.tick()

        if res_m2 in ("unit", "last_unit"):
            if self.m2.batch_type == "P2STEP2_MULTI":
                self.stock.add_p2(1)

            if res_m2 == "last_unit":
                self.m2.reset_after_batch()

        # -----------------------------------------------------------
        # 4) LIVRAISONS DE MP
        # -----------------------------------------------------------
        delivered = self.delivery.tick(self.time)
        if delivered > 0:
            self.stock.add_raw(delivered)

        # -----------------------------------------------------------
        # 5) INCRÉMENT DU TEMPS
        # -----------------------------------------------------------
        self.time += 1

        # -----------------------------------------------------------
        # 6) DEMANDE + VENTES — toutes les 15 minutes
        # -----------------------------------------------------------
        if self.time % 15 == 0:

            new_d1, new_d2 = self.market.sample_demand(self.time, 15)
            self.demande_p1 += int(new_d1)
            self.demande_p2 += int(new_d2)

            sold_p1, sold_p2 = self.market.compute_sales(
                self.stock, self.demande_p1, self.demande_p2
            )

            reward += 2.0 * sold_p1 + 20.0 * sold_p2

            self.demande_p1 -= sold_p1
            self.demande_p2 -= sold_p2

            # -----------------------------------------------------------
            # PÉNALITÉ BACKLOG — CHAQUE MINUTE
            # -----------------------------------------------------------
            backlog = self.demande_p1 + self.demande_p2
            reward -= 0.02 * float(backlog)

        # -----------------------------------------------------------
        # 7) VOL QUOTIDIEN (minute 1435)
        # -----------------------------------------------------------
        if self.time % 1440 == self.theft_time:
            self.market.apply_theft(self.stock)


        # -----------------------------------------------------------
        # 8) TERMINAISON
        # -----------------------------------------------------------
        terminated = self.time >= self.max_time

        return self._get_obs(), reward, terminated, False, {}

    # ================================================================
    # CONSTRUCTION DE L'OBSERVATION POUR SB3
    # ================================================================
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
