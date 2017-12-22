class TradingData:
    def __init__(self, trainingDataSet, numberOfTrainingWindowDays):
        self.tradingData = []
        for trainingData in trainingDataSet:
            datum = trainingData.X[numberOfTrainingWindowDays:]
            classification = trainingData.Y[numberOfTrainingWindowDays:]
            dates = trainingData.dates[numberOfTrainingWindowDays:]
            self.tradingData.append((datum, classification, dates))

    def getTestSet(self, assetIndex, day):
        return self.tradingData[assetIndex][0][day].reshape(1, -1)

    def getClassification(self, assetIndex, day):
        return self.tradingData[assetIndex][1][day]

    def getDate(self, assetIndex, day):
        return self.tradingData[assetIndex][2][day]

    def __iter__(self):
        for datum in self.tradingData:
            yield datum
