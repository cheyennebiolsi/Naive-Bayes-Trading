import os
import pandas
import numpy as np

class PortfolioWeightsPublisher:
    def __init__(self, resultsDirectory, configFilePath, equityDataSet):
        configFileName = configFilePath.split("/")[-1]
        self.resultsFileName = os.path.join(resultsDirectory, configFileName + ".csv")
        self.equityNames = [equityData.name[2:-4] for equityData in equityDataSet]

    def publish(self, weights, dates):
        weights = np.asarray(weights).T
        df = pandas.DataFrame(weights, self.equityNames, dates)
        df.to_csv(self.resultsFileName)
