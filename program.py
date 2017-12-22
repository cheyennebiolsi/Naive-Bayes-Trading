from asset.asset import *

from dataCollection.equityData import *
from dataCollection.tradingData import *
from dataCollection.trainingData import *

from publishing.plotPublisher import *
from publishing.portfolioWeightsPublisher import *
from publishing.tradePlotFactory import *

from setup.config import *

from simulator.efficientFrontier import *
from simulator.portfolioCalculator import *
from simulator.sharpeRatioCalculator import *
from simulator.simulator import *

def run():
    if len(sys.argv) == 1:
        print("Error: expected configuration file")
        return

    configFilePath = str(sys.argv[1])
    config = Config(configFilePath)
    dataDirectory = config.getDataDirectory()
    startDate = config.getStartDate()
    numberOfDaysToSimulate = config.getNumberOfDaysToSimulate()
    numberOfTrainingWindowDays = config.getNumberOfTrainingWindowDays()
    numberOfPredictorDays = config.getNumberOfPredictorDays()
    riskFreeReturn = config.getRiskFreeReturn()
    additionalArgs = config.getAdditionalArgs()
    plotResults = config.getPlotResults()
    resultsDirectory = config.getResultsDirectory()

    classifierFactory = ClassifierFactory()
    portfolioCalculator = PortfolioCalculator()
    markowitzCalculator = MarkowitzCalculator()
    efficientFrontierFactory = EfficientFrontierFactory(portfolioCalculator,
                                                        markowitzCalculator)
    simulator = Simulator(numberOfDaysToSimulate,
                        plotResults,
                        classifierFactory,
                        portfolioCalculator,
                        efficientFrontierFactory)
    equityDataSet = EquityDataSet(dataDirectory)
    trainingDataSet = TrainingDataSet(equityDataSet,
                                    numberOfPredictorDays,
                                    numberOfTrainingWindowDays,
                                    numberOfDaysToSimulate,
                                    startDate,
                                    additionalArgs)
    assets = Assets(trainingDataSet, numberOfTrainingWindowDays)
    tradingData = TradingData(trainingDataSet, numberOfTrainingWindowDays)

    simulationResults = simulator.simulate(assets, tradingData)
    theorizedReturns = portfolioCalculator.getDailyPortfolioReturns(
                                            simulationResults.portfolioWeights,
                                            simulationResults.expectedReturns)

    realizedReturns = trainingDataSet.getReturnsForSimulationPeriod()
    portfolioReturns = portfolioCalculator.getDailyPortfolioReturns(
                                            simulationResults.portfolioWeights,
                                            realizedReturns)

    if plotResults:
        tradePlotFactory = TradePlotFactory(equityDataSet)
        tradeFig = tradePlotFactory.build(simulationResults.portfolioWeights,
                                            portfolioReturns,
                                            theorizedReturns,
                                            simulationResults.dates)
        plotPublisher = PlotPublisher(resultsDirectory, configFilePath)
        plotPublisher.publish(simulationResults.frontierGraphs, tradeFig)

    portfolioWeightsPublisher = PortfolioWeightsPublisher(resultsDirectory, configFilePath, equityDataSet)
    portfolioWeightsPublisher.publish(simulationResults.portfolioWeights, simulationResults.dates)
    sharpeRatioCalculator = SharpeRatioCalculator(riskFreeReturn)
    print("Actual Sharpe Ratio: {}".format(sharpeRatioCalculator.calculate(portfolioReturns)))
    print("Expected Sharpe Ratio: {}".format(sharpeRatioCalculator.calculate(theorizedReturns)))

if __name__ == "__main__":
    run()
