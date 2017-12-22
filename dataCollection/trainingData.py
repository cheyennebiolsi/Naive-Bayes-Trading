import numpy as np
from dateutil.parser import parse

class TrainingDataSet:
    def __init__(self,
                equityDataSet,
                numberOfPredictorDays,
                numberOfTrainingWindowDays,
                numberOfDaysToSimulate,
                startDate,
                additionalArgs):
        self.trainingDataSet = []
        self.numberOfTrainingWindowDays = numberOfTrainingWindowDays
        self.numberOfDaysToSimulate = numberOfDaysToSimulate
        for equityData in equityDataSet:
            indexOfStartDate = equityData.getIndexOfDate(startDate)
            startIndex = indexOfStartDate - numberOfTrainingWindowDays
            endIndex = indexOfStartDate + numberOfDaysToSimulate
            trainingData = TrainingData(equityData, startIndex, endIndex, numberOfPredictorDays, additionalArgs)
            self.trainingDataSet.append(trainingData)

    def __iter__(self):
        for trainingData in self.trainingDataSet:
            yield trainingData

    def getReturnsForSimulationPeriod(self):
        allAssetReturns = []
        for day in range(self.numberOfDaysToSimulate):
            returnsForDay = []
            for trainingData in self:
                returnsForDay.append(trainingData.X[self.numberOfTrainingWindowDays + day + 1, 0])
            allAssetReturns.append(returnsForDay)
        return allAssetReturns

class TrainingData:
    def __init__(self, equityData, startIndex, endIndex, numberOfPredictorDays, additionalArgs):
        self.numberOfPredictorDays = numberOfPredictorDays
        self.additionalArgs = additionalArgs
        self.equityData = equityData
        self.dates = []
        historicalData = []
        identificationData = []
        for index in range(startIndex, endIndex + 1):
            historicalDataForIndex = []
            historicalDataForIndex += self.getReturnDataForEachPredictorDay(index)
            historicalDataForIndex += self.getSupplementaryDataForAdditionalArgs(index)
            historicalData.append(historicalDataForIndex)
            identificationData.append(self.equityData.getReturnForIndex(index) <= 0)
            self.dates.append(parse(self.equityData.getValueForIndex(index, "date")))
        self.X = np.asarray(historicalData)
        self.Y = np.asarray(identificationData)

    def getReturnDataForEachPredictorDay(self, index):
        historicalDataForPredictorDays = []
        for p in range(1, self.numberOfPredictorDays + 1):
            historicalDataForPredictorDays.append(self.equityData.getReturnForIndex(index - p))
        return historicalDataForPredictorDays

    def getSupplementaryDataForAdditionalArgs(self, index):
        historicalDataForAdditionalArgs = []
        for arg in self.additionalArgs:
            for p in range(1, self.numberOfPredictorDays + 1):
                historicalDataForAdditionalArgs.append(self.equityData.getValueForIndex(index - p, arg))
        return historicalDataForAdditionalArgs
