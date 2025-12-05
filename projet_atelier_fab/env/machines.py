class Machine:
    def __init__(self):
        self.busy = False
        self.time_left = 0
        self.batch_type = None
        self.batch_k = 0

    def start_batch(self, duration, k, batch_type):
        """
        Lance un batch sur cette machine.
        La machine devient busy pour `duration` minutes.
        """
        self.busy = True
        self.time_left = duration
        self.batch_type = batch_type
        self.batch_k = k

    def tick(self):
        """
        Fait avancer le temps d’une minute sur la machine.
        Retourne True si un batch se termine à cet instant.
        """
        if not self.busy:
            return False

        self.time_left -= 1
        if self.time_left <= 0:
            # Fin de batch
            return True
        return False

    def reset_after_batch(self):
        """
        Remet la machine à l'état idle après un batch terminé.
        """
        self.busy = False
        self.time_left = 0
        self.batch_type = None
        self.batch_k = 0

    def reset(self):
        """
        Remise à zéro complète (au début d’un épisode).
        """
        self.busy = False
        self.time_left = 0
        self.batch_type = None
        self.batch_k = 0
