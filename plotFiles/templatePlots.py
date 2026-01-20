###############################
# Plots in this file:
# 1. 2d Histogram of relative timestamp on relative rows. Fitted is the template used to find perfect clusters
###############################

import sys
from plotClass import plotGenerator
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from dataAnalysis.handlers._handler_perfectClusterHandler import getRelativeRowTS,calcTemplate

config = configLoader.loadConfig("config.json")
dataFiles = initDataFiles(config)

def runTemplate(dataFiles,plotGen,config):
    maxRow = 25
    TSRange = 30
    for dataFile in dataFiles:
        layer = 4
        path = f"PerfectClusters/{dataFile.fileName}/"
        clusters = dataFile.get_clusters(layer=4,excludeCrossTalk = True)
        relativeRowList, relativeTSList = getRelativeRowTS(clusters,TSRange=TSRange,maxRow=maxRow,minExpectedClusterSize=int(np.ceil(np.sqrt(dataFile.voltage/48.6) * 8)))
        estimate, spread = calcTemplate(relativeRowList[relativeRowList<=maxRow],relativeTSList[relativeRowList<=maxRow])
        plot = plotGen.newPlot(path,sizePerPlot = (6,4))
        array, yedges, xedges = np.histogram2d(relativeTSList,relativeRowList,range=((-0.5,TSRange+0.5),(-0.5,maxRow+0.5)),bins=(TSRange+1,maxRow+1))
        plot.axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
        plot.axs.scatter(np.arange(len(estimate)),estimate,marker="x",color=plot.colorPalette[0],label="Gaussian Fitting on each Row")
        plot.axs.errorbar(
                np.arange(len(estimate)),
                estimate,
                yerr=spread,
                fmt="none",
                color=plot.colorPalette[0],
                elinewidth=1,
                capsize=3,
            )
        plot.set_config(plot.axs,
            title="Row vs TS",
            xlabel="Relative Row [px]",
            ylabel="Relative Timestamp [TS]",
            xlim = (xedges[0],xedges[-1]),
            ylim = (yedges[0],yedges[-1]),
            legend=True,
            labelcolor="w",
            xticks=[5,1],
            yticks=[5,1],
            )
        plot.saveToPDF("Template") 

if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runTemplate(dataFiles,plotGen,config)