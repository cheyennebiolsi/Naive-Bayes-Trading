from sklearn.naive_bayes import GaussianNB
from .simulationResults import *
import numpy as np

class ClassifierFactory:
    def __init__(self):
        pass

    def build(self, asset):
        classifier = GaussianNB()
        classifier.fit(asset.assetHistory, asset.identificationValues)
        return classifier

class Simulator:
    def __init__(self, numberOfDaysToSimulate, plotResults, classifierFactory, portfolioCalculator, efficientFrontierFactory):
        self.numberOfDaysToSimulate = numberOfDaysToSimulate
        self.plotResults = plotResults
        self.classifierFactory = classifierFactory
        self.portfolioCalculator = portfolioCalculator
        self.efficientFrontierFactory = efficientFrontierFactory

    def simulate(self, assets, tradingData):
        simulationResults = SimulationResults()
        for day in range(self.numberOfDaysToSimulate):
            self.simulateDay(assets, tradingData, simulationResults, day)
        return simulationResults

    def simulateDay(self, assets, tradingData, simulationResults, day):
        expectations = []
        volatilities = []
        for index, asset in enumerate(assets):
            testSet = tradingData.getTestSet(index, day)
            classification = tradingData.getClassification(index, day)
            classifier = self.classifierFactory.build(asset)
            e = self.getExpectedReturn(classifier, testSet, asset)
            v = asset.updateVolatility()
            asset.update(testSet, classification)
            expectations.append(e)
            volatilities.append(v)
        date = tradingData.getDate(index, day)

        tanWeights = self.portfolioCalculator.getTangencyPortfolioWeights(volatilities, expectations)
        mu = self.portfolioCalculator.getMu(tanWeights, expectations)

        figs = []
        if self.plotResults:
            figs.append(self.efficientFrontierFactory.build(tanWeights, expectations, volatilities, date))

        if mu < 0:
            tanWeights = -1*np.array(tanWeights)

        simulationResults.update(date, tanWeights, expectations, figs)

    def getExpectedReturn(self, model, testSet, asset):
        averageIncrease, averageDecrease = asset.averages()
        probability = model.predict_proba(testSet)
        positiveProb = probability[0][0]
        negativeProb = probability[0][1]
        movement = positiveProb*averageIncrease + negativeProb*averageDecrease
        return(movement)
