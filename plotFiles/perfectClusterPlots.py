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
from dataAnalysis.handlers._genericClusterFuncs import gaussianFunc

def runPerfect(dataFiles,plotGen,config):
    maxRow = 25
    TSRange = 20
    for dataFile in dataFiles:
        layer = 4
        path = f"PerfectClusters/{dataFile.fileName}/"
        clusters = dataFile.get_clusters(layer=layer,excludeCrossTalk = True)
        minExpectedClusterSize = int(np.ceil(np.sqrt(dataFile.voltage/48.6) * 8))
        relativeRowList, relativeTSList = getRelativeRowTS(clusters,TSRange=TSRange,maxRow=maxRow,minExpectedClusterSize=minExpectedClusterSize)
        estimate, spread = calcTemplate(relativeRowList[relativeRowList<=maxRow],relativeTSList[relativeRowList<=maxRow])
        array, yedges, xedges = np.histogram2d(relativeTSList,relativeRowList,range=((-0.5,TSRange+0.5),(-0.5,len(estimate)+0.5)),bins=(TSRange+1,len(estimate)+1))
        plotTemplate(plotGen,path,array, yedges, xedges,estimate, spread)
        for slice in np.arange(0,len(estimate),4):
            plotSlice(plotGen,path,array, yedges, xedges,estimate, spread, slice)

if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runPerfect(dataFiles,plotGen,config)
