import numpy as np


class Market:
    def __init__(self, lambda_day=0.1, lambda_night=0.02):
        self.lambda_day = lambda_day
        self.lambda_night = lambda_night

    def sample_demand(self, current_time: int, period_minutes: int = 60):
        """
        Génère la demande P1 et P2 sur une période donnée.
        Le marché est plus actif en journée qu'en nuit.
        """

        hour = (current_time // 60) % 24
        is_day = 8 <= hour < 20

        lam = self.lambda_day if is_day else self.lambda_night
        mean_requests = lam * period_minutes

        # demande totale
        total = np.random.poisson(mean_requests)
        p1_frac = 0.7
        p1 = int(total * p1_frac)
        p2 = total - p1
        return p1, p2

    def compute_sales(self, stock, backlog_p1: int, backlog_p2: int):
        """
        Détermine UNIQUEMENT les quantités vendues.
        Le reward est calculé dans l'environnement.
        """

        s1 = min(stock.p1, backlog_p1)
        s2 = min(stock.p2, backlog_p2)

        # débit du stock
        stock.p1 -= s1
        stock.p2 -= s2

        # Retourne EXACTEMENT 2 valeurs (pas 3 !)
        return s1, s2

    def apply_theft(self, stock):
        """Applique le vol nocturne : 10 % de pertes sur P1 et P2.

        La règle est :
            P1 <- int(P1 * 0.9)
            P2 <- int(P2 * 0.9)

        On retourne les quantités volées (stolen_p1, stolen_p2) à titre informatif.
        """
        old_p1 = stock.p1
        old_p2 = stock.p2

        stock.p1 = int(stock.p1 * 0.9)
        stock.p2 = int(stock.p2 * 0.9)

        stolen_p1 = old_p1 - stock.p1
        stolen_p2 = old_p2 - stock.p2

        return stolen_p1, stolen_p2
