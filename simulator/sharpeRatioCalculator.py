import numpy as np

class SharpeRatioCalculator:
    def __init__(self, riskFreeReturn):
        self.riskFreeReturn = riskFreeReturn

    def calculate(self, returns):
        returns = np.asarray(returns).reshape(1, len(returns))
        mean = np.mean(returns, axis = 1)
        std = np.sqrt(np.var(returns, axis = 1))
        s = (mean - self.riskFreeReturn) / std
        return s.item(0)
