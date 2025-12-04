import numpy as np


class Market:
    """
    Modélise la demande avec :
    - profil jour / nuit
    - agrégation sur une période (par défaut : 60 minutes)
    """

    def __init__(self,
                 lambda_p1_day=1.0,     # demande moyenne P1 / minute en journée
                 lambda_p1_night=0.1,   # demande moyenne P1 / minute la nuit
                 lambda_p2_day=0.2,     # demande moyenne P2 / minute en journée
                 lambda_p2_night=0.02   # demande moyenne P2 / minute la nuit
                 ):
        self.lambda_p1_day = lambda_p1_day
        self.lambda_p1_night = lambda_p1_night
        self.lambda_p2_day = lambda_p2_day
        self.lambda_p2_night = lambda_p2_night

    def _get_lambdas_for_time(self, time_minute: int):
        """
        time_minute : minute globale dans l'épisode.
        On en déduit la minute dans la journée (0-1439) puis on choisit
        les lambdas jour / nuit.
        """
        minute_in_day = time_minute % 1440

        # JOUR = 8h à 20h  => 480 à 1200
        if 480 <= minute_in_day < 1200:
            lambda_p1 = self.lambda_p1_day
            lambda_p2 = self.lambda_p2_day
        else:
            lambda_p1 = self.lambda_p1_night
            lambda_p2 = self.lambda_p2_night

        return lambda_p1, lambda_p2

    def sample_demand(self, time_minute: int, period_minutes: int = 60):
        """
        Génère la demande agrégée sur 'period_minutes' minutes.
        On utilise un Poisson(λ * période), avec λ dépendant du moment de la journée.
        """
        lambda_p1, lambda_p2 = self._get_lambdas_for_time(time_minute)

        d1 = np.random.poisson(lambda_p1 * period_minutes)
        d2 = np.random.poisson(lambda_p2 * period_minutes)

        return d1, d2

    def compute_sales(self, stock, backlog_p1: int, backlog_p2: int):
        """
        Calcule les ventes à partir :
        - du stock actuel (stock.p1, stock.p2)
        - du carnet de commandes (backlog_p1, backlog_p2)

        Retourne :
        - revenue : revenu total
        - s1 : quantités de P1 vendues
        - s2 : quantités de P2 vendues
        et met à jour le stock.
        """
        s1 = min(stock.p1, backlog_p1)
        s2 = min(stock.p2, backlog_p2)

        revenue = 2 * s1 + 20 * s2

        stock.p1 -= s1
        stock.p2 -= s2

        return revenue, s1, s2
