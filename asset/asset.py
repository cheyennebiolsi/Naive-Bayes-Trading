from arch.univariate import arch_model
import numpy as np

class Assets:
    def __init__(self, trainingDataSet, numberOfTrainingWindowDays):
        self.assets = []
        for trainingData in trainingDataSet:
            assetHistory = trainingData.X[:numberOfTrainingWindowDays, ]
            identificationValues = trainingData.Y[:numberOfTrainingWindowDays]
            asset = Asset(trainingData.equityData.name, assetHistory = assetHistory, identificationValues = identificationValues)
            asset.recountAverages()
            self.assets.append(asset)

    def __iter__(self):
        for asset in self.assets:
            yield asset

class Asset:
    def __init__(self, name, assetHistory = [], identificationValues = [], positiveCount = 0,
                 positiveTotal = 0.0, negativeCount = 0, negativeTotal = 0.0,
                 lastVolatility = 0):
        self.name = name
        self.assetHistory = assetHistory
        self.identificationValues = identificationValues
        self.positiveCount = positiveCount
        self.positiveTotal = positiveTotal
        self.negativeCount = negativeCount
        self.negativeTotal = negativeTotal
        self.lastVolatility = lastVolatility

    def recountAverages(self):
        increase = 0.0; increaseCount = 0.0
        decrease = 0.0; decreaseCount = 0.0
        for row in self.assetHistory:
            if row[0]>0:
                increase+=row[0]
                increaseCount+=1
            else:
                decrease+=row[0]
                decreaseCount+=1
        self.positiveCount = increaseCount
        self.positiveTotal = increase
        self.negativeCount = decreaseCount
        self.negativeTotal = decrease

    def averages(self):
        return self.positiveTotal/self.positiveCount, self.negativeTotal/self.negativeCount

    def update(self, data, classValue):
        self.assetHistory = np.concatenate((self.assetHistory, data), axis = 0)
        self.identificationValues = np.append(self.identificationValues, classValue)
        if data[0,0] > 0:
            self.positiveCount+=1
            self.positiveTotal+=data[0,0]
        else:
            self.negativeCount+=1
            self.negativeTotal+=data[0,0]
        if self.assetHistory[0,0] > 0:
            self.positiveCount-=1
            self.positiveTotal-=self.assetHistory[0,0]
        else:
            self.negativeCount-=1
            self.negativeTotal-=self.assetHistory[0,0]
        self.assetHistory = self.assetHistory[1:,:]
        self.identificationValues = self.identificationValues[1:]

    def updateVolatility(self):
        returns = self.assetHistory[:, 0].tolist()
        am = arch_model(returns)
        res=am.fit(disp='off')
        omega = res.params[1]
        alpha = res.params[2]
        beta = res.params[3]
        if self.lastVolatility == 0:
            prevVol = np.array(returns).var()
        else:
            prevVol = self.lastVolatility
        sigma2 = omega + alpha*(returns[-1]**2) + beta*prevVol
        self.lastVolatility = sigma2
        return(self.lastVolatility)
