import os
import json
import copy
import sys
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
systemConfigFilePath = os.path.abspath(__file__)
setupDirectory = os.path.abspath(os.path.join(systemConfigFilePath,os.pardir))
PARENT_DIRECTORY = os.path.abspath(os.path.join(setupDirectory,os.pardir))

class Config:
    def __init__(self, configurationPath):
        configurationFilePath = os.path.join(PARENT_DIRECTORY, configurationPath)
        self.configurationPath = configurationPath
        self.configuration = None
        try:
            with open(configurationFilePath, encoding='utf-8') as ConfigFile:
                self.configuration = json.load(ConfigFile)
        except IOError as e:
            raise IOError("Cannot read file {0}. Please make sure this file exists.".format(configurationFilePath))
        self.dataDirectory = self._get("dataDirectory")
        self.startDate = self._get("startDate")
        self.numberOfDaysToSimulate = int(self._get("numberOfDaysToSimulate"))
        self.numberOfTrainingWindowDays = int(self._get("numberOfTrainingWindowDays"))
        self.numberOfPredictorDays = int(self._get("numberOfPredictorDays"))
        self.additionalArgs = self._get("additionalArgs")
        self.riskFreeReturn = int(self._get("riskFreeReturn"))
        self.plotResults = self._get("plotResults")
        self.resultsDirectory = self._get("resultsDirectory")

    def getDataDirectory(self):
        return self.dataDirectory

    def getStartDate(self):
        return self.startDate

    def getNumberOfDaysToSimulate(self):
        return self.numberOfDaysToSimulate

    def getNumberOfTrainingWindowDays(self):
        return self.numberOfTrainingWindowDays

    def getNumberOfPredictorDays(self):
        return self.numberOfPredictorDays

    def getAdditionalArgs(self):
        # if self.getAdditionalArgs == None:
        #     return []
        return self.additionalArgs

    def getRiskFreeReturn(self):
        return self.riskFreeReturn

    def getPlotResults(self):
        # if self.getAdditionalArgs == None:
        #     return []
        return self.plotResults

    def getResultsDirectory(self):
        return self.resultsDirectory

    def _get(self, item):
        """ Returns the configuration mapped to by item. """
        return copy.deepcopy(self.configuration.get(item))
#
# class StartAndEndDatesConfig:
#     def __init__(self, dateTimeConverter, countryStartAndEndDatesFile):
#         countryStartAndEndDatesFilePath = os.path.join(PARENT_DIRECTORY, countryStartAndEndDatesFile)
#         with open(countryStartAndEndDatesFilePath, encoding='utf-8') as startAndEndDatesConfig:
#             countryRowData = list(csv.reader(startAndEndDatesConfig))
#         header = countryRowData[0]
#         countryNameIndex = header.index('Country')
#         startDateIndex = header.index('Start')
#         endDateIndex = header.index('End')
#         self.startAndEndDatesStore = {}
#         for row in countryRowData[1:]:
#             countryName = row[countryNameIndex]
#             startDate = dateTimeConverter.convertToDateTime(row[startDateIndex])
#             endDate = dateTimeConverter.convertToDateTime(row[endDateIndex])
#             self.startAndEndDatesStore[countryName] = (startDate, endDate)
#
#     def tryGetStartDate(self, countryName):
#         if countryName in self.startAndEndDatesStore:
#             return self.startAndEndDatesStore[countryName][0]
#         return None
#
#     def tryGetEndDate(self, countryName):
#         if countryName in self.startAndEndDatesStore:
#             return self.startAndEndDatesStore[countryName][1]
#         return None
#
# class ConfigFactory:
#     def __init__(self, StartAndEndDatesConfigFactory):
#         self.startAndEndDatesConfigFactory = StartAndEndDatesConfigFactory
#
#     def Build(self, configurationPath):
#         configuration = Config(self.startAndEndDatesConfigFactory, configurationPath)
#         return configuration;
#
# class StartAndEndDatesConfigFactory:
#     def __init__(self, dateTimeConverter):
#         self.dateTimeConverter = dateTimeConverter
#
#     def Build(self, countryStartAndEndDatesFile):
#         startAndEndDatesConfig = StartAndEndDatesConfig(self.dateTimeConverter, countryStartAndEndDatesFile)
#         return startAndEndDatesConfig;
#
# # Optimizer["BASINHOPPING"]
# #
# # dateTimeConverter = DateTimeConverter()
# # startAndEndDatesConfigFactory = StartAndEndDatesConfigFactory(dateTimeConverter)
# # configBuilder = ConfigFactory(startAndEndDatesConfigFactory);
# # config = configBuilder.Build(sys.argv[1])
# # startAndEndDatesConfig = config.getStartAndEndDatesConfig()
# # print(config.getCountries())
# # print(startAndEndDatesConfig.tryGetStartDate('Chile'))
# # print(startAndEndDatesConfig.tryGetEndDate('Chile'))
# # print(config.getResultsDirectory())
