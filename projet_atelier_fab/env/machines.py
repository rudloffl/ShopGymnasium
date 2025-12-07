
class Machine:
    def __init__(self):
        self.busy = False
        self.time_left = 0
        self.batch_type = None
        self.batch_k = 0
        # Nouveau : production au fil de l'eau
        self.unit_duration = 0           # durée pour produire UNE unité
        self.time_since_last_unit = 0    # compteur interne depuis la dernière unité produite

    def start_batch(self, duration, k, batch_type):
        """
        Lance un batch sur cette machine.
        La machine devient busy pour `duration` minutes.
        On suppose que `duration` est un multiple de k : chaque unité
        prend `duration // k` minutes à produire.
        """
        self.busy = True
        self.time_left = duration

        # Durée d'une unité et reset du compteur interne
        self.unit_duration = duration // k if k > 0 else duration
        self.time_since_last_unit = 0

        self.batch_type = batch_type
        self.batch_k = k

    def tick(self):
        """
        Fait avancer le temps d'une minute sur la machine.
        Retourne :
          - "none"      : aucune unité produite à cette minute
          - "unit"      : une unité produite, mais le batch continue
          - "last_unit" : une unité produite et le batch est terminé
        """
        if not self.busy:
            return "none"  # rien à produire

        # Avancement du temps
        self.time_left -= 1
        self.time_since_last_unit += 1

        # Par sécurité : éviter les divisions bizarres
        if self.unit_duration <= 0:
            self.unit_duration = 1

        # Une unité vient d'être produite ?
        if self.time_since_last_unit >= self.unit_duration:
            self.batch_k -= 1
            self.time_since_last_unit = 0

            # Si c'était la dernière → fin de batch
            if self.batch_k <= 0:
                self.batch_k = 0
                return "last_unit"

            return "unit"   # unité intermédiaire produite

        # Pas encore de nouvelle unité produite
        return "none"

    def reset_after_batch(self):
        """
        Remet la machine à l'état idle après un batch terminé.
        """
        self.busy = False
        self.time_left = 0
        self.batch_type = None
        self.batch_k = 0
        self.unit_duration = 0
        self.time_since_last_unit = 0

    def reset(self):
        """
        Remise à zéro complète (au début d’un épisode).
        """
        self.busy = False
        self.time_left = 0
        self.batch_type = None
        self.batch_k = 0
        self.unit_duration = 0
        self.time_since_last_unit = 0
