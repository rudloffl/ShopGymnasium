class Machine:
    def __init__(self):
        self.busy = False
        self.time_left = 0
        self.batch_k = 0
        self.batch_type = None

    def start_batch(self, duration, k, batch_type):
        """
        Lance un batch de production.
        """
        self.busy = True
        self.time_left = duration
        self.batch_k = k
        self.batch_type = batch_type

    def tick(self):
        """
        Fait avancer la machine d'une minute.
        Retourne True si un batch se termine pendant cette minute.
        """
        if not self.busy:
            return False

        self.time_left -= 1

        if self.time_left <= 0:
            self.busy = False
            return True  # batch terminé

        return False  # batch toujours en cours
