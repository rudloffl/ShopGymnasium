import numpy as np

class Market:
    def __init__(self, lambda_p1=1, lambda_p2=0.2):
        self.lambda_p1 = lambda_p1
        self.lambda_p2 = lambda_p2

    def sample_demand(self):
        d1 = np.random.poisson(self.lambda_p1)
        d2 = np.random.poisson(self.lambda_p2)
        return d1, d2

    def compute_sales(self, stock, d1, d2):
        s1 = min(stock.p1, d1)
        s2 = min(stock.p2, d2)
        revenue = 2 * s1 + 20 * s2
        stock.p1 -= s1
        stock.p2 -= s2
        return revenue
