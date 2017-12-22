import os
import csv
from dateutil.parser import parse
import numpy as np

class EquityDataSet:
    def __init__(self, dataDirectory):
        self.equityDataSet = []
        for root, dirs, fileNames in os.walk(dataDirectory):
            for fileName in fileNames:
                try:
                    filePath = os.path.join(dataDirectory, fileName)
                    self.equityDataSet.append(EquityData(filePath, fileName))
                except:
                    print("Failed to read data in {}".format(fileName))

    def __iter__(self):
        for equityData in self.equityDataSet:
            yield equityData

class EquityData:
    def __init__(self, filePath, fileName):
        with open(filePath, "r") as dataFile:
            reader = csv.reader(dataFile, delimiter = ",")
            self.equityData = list(reader)
        self.name = fileName

    def getReturnForIndex(self, index):
        closeColumn = self.getIndexOfColumn("close")
        try:
            return(100*np.log(float(self.equityData[index][closeColumn])/float(self.equityData[index-1][closeColumn])))
        except ValueError:
            print("Not enough data in dataset for {}".format(self.name))

    def getValueForIndex(self, index, columnName):
        columnIndex = self.getIndexOfColumn(columnName)
        return self.equityData[index][columnIndex]

    def getIndexOfColumn(self, columnName):
        try:
            columnNames = self.equityData[0]
            return(next(i for i, v in enumerate(columnNames) if v.lower().strip() == columnName))
        except:
            print("Improperly named column")

    def getIndexOfDate(self, date):
        dateColumn = self.getIndexOfColumn("date")
        try:
            start = parse(date)
        except:
            print("Invalid start date entered")
        for i in range(1, len(self.equityData)):
            if start <= parse(self.equityData[i][dateColumn]):
                return i
        return -1
