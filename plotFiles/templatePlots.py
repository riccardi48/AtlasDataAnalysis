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
def plotTemplate(plotGen,path,array, yedges, xedges,estimate, spread):
    plot = plotGen.newPlot(path,sizePerPlot = (6,4))
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


def plotSlice(plotGen,path,array, yedges, xedges,estimate, spread, slice):
    plot = plotGen.newPlot(path,sizePerPlot = (6,4))
    height = array[:,slice]
    binCentres = (yedges[:-1] + yedges[1:]) / 2
    cut = np.mean(binCentres[np.where(height==np.max(height))[0]])+2
    plot.axs.stairs(height,yedges, color=plot.colorPalette[0], baseline=None,label = "Data")
    x = np.linspace(yedges[0],yedges[-1],100)
    y = gaussianFunc(x, estimate[slice], spread[slice], np.sum(height))
    plot.axs.plot(x[x<=cut],y[x<=cut],color=plot.colorPalette[2],label="Gaussian Fitting")
    plot.axs.plot(x[x>cut],y[x>cut],color=plot.colorPalette[2],linestyle="dashed")
    plot.set_config(plot.axs,
        title=f"Row vs TS Slice {slice}",
        xlabel="Relative Row [px]",
        ylabel="Relative Timestamp [TS]",
        xlim = (yedges[0],yedges[-1]),
        ylim = (0,None),
        legend=True,
        xticks=[5,1],
        yticks=[5,1],
        )
    plot.saveToPDF(f"Template_Slice_{slice}") 

def runTemplate(dataFiles,plotGen,config):
    maxRow = 25
    TSRange = 20
    for dataFile in dataFiles:
        layer = 4
        path = f"PerfectClusters/{dataFile.fileName}/"
        clusters = dataFile.get_clusters(layer=layer,excludeCrossTalk = True)
        minExpectedClusterSize = int(np.ceil(np.sqrt(dataFile.voltage/48.6) * 8))
        if dataFile.angle != 86.5:
            minExpectedClusterSize = 2
        relativeRowList, relativeTSList = getRelativeRowTS(clusters,TSRange=TSRange,maxRow=maxRow,minExpectedClusterSize=minExpectedClusterSize)
        estimate, spread = calcTemplate(relativeRowList[relativeRowList<=maxRow],relativeTSList[relativeRowList<=maxRow])
        array, yedges, xedges = np.histogram2d(relativeTSList,relativeRowList,range=((-0.5,TSRange+0.5),(-0.5,len(estimate)+0.5)),bins=(TSRange+1,len(estimate)+1))
        plotTemplate(plotGen,path,array, yedges, xedges,estimate, spread)
        for slice in np.arange(0,len(estimate),4):
            plotSlice(plotGen,path,array, yedges, xedges,estimate, spread, slice)

if __name__ == "__main__":
    config = configLoader.loadConfig()
    config["filterDict"] = {"angle":86.5,"telescope":"kit"}
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runTemplate(dataFiles,plotGen,config)