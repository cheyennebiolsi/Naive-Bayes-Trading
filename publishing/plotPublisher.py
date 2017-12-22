import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

class PlotPublisher:
    def __init__(self, resultsDirectory, configFilePath):
        configFileName = configFilePath.split("/")[-1]
        self.resultsFileName = os.path.join(resultsDirectory, configFileName + ".pdf")

    def publish(self, frontierGraphs, tradePlot):
        with PdfPages(self.resultsFileName) as pdf:
            for fig in frontierGraphs:
                pdf.savefig(fig)
            pdf.savefig(tradePlot)
