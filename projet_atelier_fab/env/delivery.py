class DeliveryQueue:
    def __init__(self):
        self.queue = []

    def schedule(self, quantity, arrival_time):
        self.queue.append((quantity, arrival_time))

    def tick(self, current_time):
        """
        Retourne la quantité livrée au temps courant.
        """
        delivered = 0
        remaining = []

        for q, t in self.queue:
            if t == current_time:
                delivered += q
            else:
                remaining.append((q, t))

        self.queue = remaining
        return delivered
