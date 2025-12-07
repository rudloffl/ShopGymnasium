class DeliveryQueue:
    def __init__(self):
        self.queue = []

    def schedule(self, q, arrival_time):
        self.queue.append((q, arrival_time))

    def tick(self, current_time):
        delivered = 0
        remaining = []
        for (q, t) in self.queue:
            if t == current_time:
                delivered += q
            else:
                remaining.append((q, t))
        self.queue = remaining
        return delivered

    def reset(self):
        self.queue = []
