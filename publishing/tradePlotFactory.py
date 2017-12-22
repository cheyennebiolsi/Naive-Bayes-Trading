import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

class TradePlotFactory:
    def __init__(self, equityDataSet):
        self.equityNames = [equityData.name[2:-4] for equityData in equityDataSet]

    def build(self, weights, portfolio, expected, dates):
        weights = np.asarray(weights).T
        x = []
        fig, (ax1, ax2) = plt.subplots(2, figsize=(16,9))

        title = "Results using Naive Bayes classifier"
        fig.suptitle(title, fontsize = 18, fontweight = 'bold')

        ax1.plot(dates, expected, label = "Expected returns")
        ax1.plot(dates, portfolio, label = "Actual returns")
        ax1.set_ylabel("returns")

        start = dates[0].strftime('%b %d, %Y')
        end = dates[-1].strftime('%b %d, %Y')

        ax1.set_title("Actual vs expected returns between {} and {}".format(start, end))
        ax1.legend(loc = 0)
        ax1.grid()
        ax1.fmt_xdata = mdates.DateFormatter('%m-%d-%Y')

        for i in range(weights.shape[0]):
            ax2.plot(dates, weights[i,:], label = self.equityNames[i][2:-4])
        ax2.fmt_xdata = mdates.DateFormatter('%m-%d-%Y')
        ax2.legend(loc = 0)
        ax2.set_ylabel("weights")
        ax2.set_title("Optimal portfolio weights between {} and {}".format(start, end))
        ax2.grid()
        plt.close()

        return fig
