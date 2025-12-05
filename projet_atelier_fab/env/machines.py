class Machine:
    def __init__(self):
        self.busy = False
        self.time_left = 0
        self.batch_k = 0
        self.batch_type = None

    def start_batch(self, duration, k, batch_type):
        self.busy = True
        self.time_left = duration
        self.batch_k = k
        self.batch_type = batch_type

    def tick(self):
        if not self.busy:
            return False

        self.time_left -= 1

        if self.time_left == 0:
            return True

        return False

    def reset_after_batch(self):
        self.busy = False
        self.time_left = 0
        self.batch_k = 0
        self.batch_type = None
