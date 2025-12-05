class Stock:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.raw = 10        # MP initiales
        self.p1 = 0
        self.p2_inter = 0
        self.p2 = 0

    # -------------------------------------------------------------
    # MÉTHODES D’AJOUT
    # -------------------------------------------------------------
    def add_raw(self, q):
        self.raw = min(self.capacity, self.raw + q)

    def add_p1(self, q):
        self.p1 = min(self.capacity, self.p1 + q)

    def add_p2_inter(self, q):
        self.p2_inter = min(self.capacity, self.p2_inter + q)

    def add_p2(self, q):
        self.p2 = min(self.capacity, self.p2 + q)

    # -------------------------------------------------------------
    # MÉTHODES DE CONSO
    # -------------------------------------------------------------
    def consume_raw(self, q):
        """
        Consomme q unités de MP si possible.
        Retourne True si la consommation est faite, sinon False.
        """
        if self.raw >= q:
            self.raw -= q
            return True
        return False

    def consume_p2_inter(self, q):
        """
        Consomme q unités de P2_INTER si possible.
        """
        if self.p2_inter >= q:
            self.p2_inter -= q
            return True
        return False

    # -------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------
    def reset(self):
        self.raw = 10   # valeur initiale
        self.p1 = 0
        self.p2_inter = 0
        self.p2 = 0
