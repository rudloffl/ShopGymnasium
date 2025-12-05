class Stock:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.raw = 10
        self.p1 = 0
        self.p2_inter = 0
        self.p2 = 0

    def consume_raw(self, q):
        if self.raw >= q:
            self.raw -= q
            return True
        return False

    def consume_p2_inter(self, q):
        if self.p2_inter >= q:
            self.p2_inter -= q
            return True
        return False

    def add_raw(self, q):
        self.raw = min(self.raw + q, self.capacity)

    def add_p1(self, q):
        self.p1 = min(self.p1 + q, self.capacity)

    def add_p2_inter(self, q):
        self.p2_inter = min(self.p2_inter + q, self.capacity)

    def add_p2(self, q):
        self.p2 = min(self.p2 + q, self.capacity)
