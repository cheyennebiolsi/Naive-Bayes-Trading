import numpy as np

class PortfolioCalculator:
    def __init__(self):
        pass

    def getDailyPortfolioReturns(self, portfolioWeights, returns):
        portfolio = []
        for i in range(len(portfolioWeights)):
            dayWeight = np.asarray(portfolioWeights[i])
            dayReturns = np.asarray(returns[i]).T
            portfolio.append(np.dot(dayWeight, dayReturns))
        return portfolio

    def getTangencyPortfolioWeights(self, volatilities, expectedReturns):
        n = len(volatilities)
        m = np.asmatrix(expectedReturns).T
        e = np.asmatrix(np.ones((n, 1)))
        cov = np.asmatrix(np.diag(volatilities))
        invCov = cov.I
        w = (invCov * m) / (e.T * invCov * m)
        return(w.reshape(1,-1).tolist()[0])

    def getMu(self, weights, expectedReturns):
        w = np.asmatrix(weights)
        p = np.asmatrix(expectedReturns)
        mu = (w * p.T).item(0)
        return(mu)

    def getSigma(self, weights, volatilities):
        w = np.asmatrix(weights)
        C = np.asmatrix(np.diag(volatilities))
        sigma = np.sqrt(w * C * w.T).item(0)
        return(sigma)
