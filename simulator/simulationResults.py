class SimulationResults:
    def __init__(self):
        self.dates = []
        self.portfolioWeights = []
        self.expectedReturns = []
        self.frontierGraphs = []

    def update(self, dates, portfolioWeights, expectedReturns, frontierGraphs):
        self.dates.append(dates)
        self.portfolioWeights.append(portfolioWeights)
        self.expectedReturns.append(expectedReturns)
        self.frontierGraphs += frontierGraphs
