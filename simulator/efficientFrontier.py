from datetime import date, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np

class EfficientFrontierFactory:
    def __init__(self, portfolioCalculator, markowitzCalculator):
        self.portfolioCalculator = portfolioCalculator
        self.markowitzCalculator = markowitzCalculator

    def build(self, tanWeights, expectedReturns, volatilities, day):
        M = []; V = []
        n = len(tanWeights)
        tanRet = self.portfolioCalculator.getMu(tanWeights, expectedReturns)
        tanRisk = self.portfolioCalculator.getSigma(tanWeights, volatilities)

        eM, eV = self.calculateFrontier(tanRet, volatilities, expectedReturns)

        # Generate 500 random portfolios
        for i in range(500):
            w = np.asmatrix(self.randomWeights(n))
            m = self.portfolioCalculator.getMu(w, expectedReturns)
            v = self.portfolioCalculator.getSigma(w, volatilities)
            M.append(m)
            V.append(v)

        #create function to add percentage signs
        def to_percent(y, pos=0):
            s = str(y)
            return s + '%'

        formatter = FuncFormatter(to_percent)

        # Draw plot
        fig, ax1 = plt.subplots(1, figsize=(16,9))
        ax1.set_xlabel("Standard Deviation")
        ax1.set_ylabel("Expected Return")
        title = datetime.strftime(day, "Efficient frontier for %b %d, %Y")
        fig.suptitle(title, fontsize = 14, fontweight = 'bold')
        ax1.set_title('Using Naive-Bayes classifier', fontsize = 10)
        ax1.plot(eV, eM, color = "red", label = "efficient frontier")
        ax1.plot(V, M, 'o', color = "green", label = "random portfolios")
        ax1.plot([tanRisk], [tanRet], 'o', markersize = 7, color = "cyan", label = "optimal weights")

        #Create tangent line
        xp = np.linspace(0.0, 2*tanRisk, 200)
        tan = np.polyfit([0.0, tanRisk], [0.0, tanRet], 1)
        p2 = np.poly1d(tan)
        ax1.plot(xp, p2(xp), color = "blue")

        # Draw arrow for sharpe ratio
        sharpe = "Sharpe: " + format((tanRet/tanRisk), '.2f')
        ax1.annotate(sharpe, xy = (tanRisk, tanRet), xytext = (-20, 20),
                         textcoords = 'offset points', ha = 'right', va = 'bottom',
                         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        ax1.yaxis.set_major_formatter(formatter)
        ax1.legend(loc = 4, numpoints=1)
        plt.grid()
        plt.close()
        return fig

    def randomWeights(self, n):
        k = np.random.rand(n)
        return k / sum(k)

    def calculateFrontier(self, tanRet, volatilities, expectedReturns):
        eM = []; eV = []
        endWeights = self.markowitzCalculator.getWeights(volatilities, expectedReturns, baseline=1.8*tanRet)
        endM = self.portfolioCalculator.getMu(np.asarray([endWeights]), expectedReturns)
        endV = self.portfolioCalculator.getSigma(np.asarray([endWeights]), volatilities)

        # Calculate optimal weights of frontier and get risks and returns
        mus = np.linspace(-5*tanRet, 5*tanRet, 500)
        frontier = []
        for mu in mus:
            w = self.markowitzCalculator.getWeights(volatilities, expectedReturns, baseline=mu)
            frontier.append(w)

        for w in frontier:
            m = self.portfolioCalculator.getMu(np.asarray([w]), expectedReturns)
            v = self.portfolioCalculator.getSigma(np.asarray([w]), volatilities)
            #Create an arc that is about symmetrical about the x axis
            if v <= endV:
                eM.append(m)
                eV.append(v)
        return eM, eV


class MarkowitzCalculator:
    ##
    # Implements Markowitz solution for portfolio problem
    # using closed-form solution
    #
    def getWeights(self, volatilities, expectedReturns, baseline = 0.01):
        n = len(volatilities)
        m = np.asmatrix(expectedReturns).T
        e = np.asmatrix(np.ones((n,1)))
        cov = np.asmatrix(np.diag(volatilities))
        invCov = cov.I

        delta = (m.T * invCov * m) * (e.T*invCov*e) - ((m.T * invCov * e)*(m.T * invCov * e))
        alpha = (1/delta) * (baseline * (m.T*invCov*e) * (e.T*invCov * e) - ((m.T * invCov * e)*(m.T * invCov * e)))
        minvar = invCov*e * (1/ (e.T*invCov*e))
        mk = invCov*m * (1/ (e.T*invCov*m))
        w = np.asarray((1-alpha.item(0))*minvar + alpha.item(0)*mk)
        w = w.reshape(1,-1).flatten().tolist()
        return(w)
