###############################
# Plots in this file:
# 1. 2d Histogram of relative timestamp on relative rows. Fitted is the template used to find perfect clusters
###############################

import sys
from plotClass import plotGenerator
from functions.genericFuncs import colorGen,getName
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from dataAnalysis.handlers._handler_perfectClusterHandler import getRelativeRowTS,calcTemplate
from dataAnalysis.handlers._genericClusterFuncs import gaussianFunc,logGaussian,logGaussianCDFFunc
from scipy.stats import lognorm
from scipy.optimize import root_scalar

def findLogPercentile(mu,sig,percentile):
    func = lambda x : logGaussianCDFFunc(np.array(x)-0.5,mu,sig) - percentile
    return root_scalar(func, bracket= [0, 1000]).root-1

def findMPV(estimate,spread):
    x = np.linspace(0.5,30,1000)
    y = logGaussian(x,estimate, spread, 1)
    return x[np.argmax(y)]-0.5

def plotTemplate(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e):
    plot = plotGen.newPlot(path,sizePerPlot = (3.4,2.4), rect=(0.07,0.09,0.995,0.995))
    plot.axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    #median = np.array([findLogPercentile(mu,sig,0.5) for mu,sig in zip(estimate[estimate>0],spread[estimate>0])])
    #upper = np.array([findLogPercentile(mu,sig,0.5+0.34) for mu,sig in zip(estimate[estimate>0],spread[estimate>0])])
    #lower = np.array([findLogPercentile(mu,sig,0.5-0.34) for mu,sig in zip(estimate[estimate>0],spread[estimate>0])])
    mpvs = np.array([findMPV(mu,sig) for mu,sig in zip(estimate[estimate>0],spread[estimate>0])])
    #print(lower)
    #print(upper)
    for i in range(array.shape[1]-1):
        if estimate[i]<=0:
            continue
        arrayRaw = array[:,i]
        arrayRaw = np.concat([np.repeat(j,arrayRaw[j]) for j in range(len(arrayRaw))])
        plot.axs.boxplot(
            arrayRaw,
            positions=[i],
            widths=0.5,
            showfliers=False,
            boxprops=dict(color=plot.colorPalette[1],linewidth=0.5),
            medianprops=dict(color=plot.colorPalette[1],linewidth=0.5),
            whiskerprops=dict(color=plot.colorPalette[1],linewidth=0.5),
            capprops=dict(color=plot.colorPalette[1],linewidth=0.5),
            label="Data Distribution" if i == 0 else None,
        )
    plot.axs.scatter(np.arange(len(estimate))[estimate>0],mpvs,marker="^",color=plot.colorPalette[0],label="MPV",s=6)
    plot.axs.errorbar(
            np.arange(len(estimate))[estimate>0],
            mpvs,
            yerr=abs(mpvs*(estimate_e[estimate>0]/estimate[estimate>0])),
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=1,
        )
    
    """
    plot.axs.scatter(np.arange(len(median)),median,marker="x",color=plot.colorPalette[0],label="Gaussian Fitting on each Row")
    plot.axs.errorbar(
            np.arange(len(median)),
            median,
            yerr=[median-lower,upper-median],
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
    """    
    plot.set_config(plot.axs,
        #title="Row vs TS",
        xlabel="Relative Row [px]",
        ylabel="Relative Timestamp [TS]",
        xlim = (xedges[0],xedges[-1]),
        ylim = (-0.5,20.5),#(yedges[0],yedges[-1]),
        legend=True,
        labelcolor="w",
        xticks=[5,1],
        yticks=[5,1],
        )
    plot.saveToPDF("Template") 


def plotSlice(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e, slice):
    plot = plotGen.newPlot(path,sizePerPlot = (3.4,2.4), rect=(0.08,0.09,0.995,0.995))
    height = array[:,slice]
    binCentres = (yedges[:-1] + yedges[1:]) / 2
    plot.axs.stairs(height,yedges, color=plot.colorPalette[0], baseline=None,label = "Data")
    if estimate[slice]>0:
        x = np.linspace(yedges[0],yedges[-1],1000)
        y = logGaussian(x+0.5,estimate[slice], spread[slice], np.sum(height))
        plot.axs.plot(x,y,color=plot.colorPalette[2],label=f"Log Normal")#\n{estimate[slice]:.2f}$\\pm${estimate_e[slice]:.4f},{spread[slice]:.2f}$\\pm${spread_e[slice]:.4f}")
        #plot.axs.vlines(x[np.argmax(y)],0,np.max(y),color=plot.colorPalette[5],linestyle="--",label=f"MPV\n{x[np.argmax(y)]:.2f}")
    plot.set_config(plot.axs,
        #title=f"Row vs TS Slice {slice}",
        xlabel="Relative Timestamp [TS]",
        ylabel="Count",
        xlim = (yedges[0],yedges[-1]),
        ylim = (0,None),
        legend=True,
        xticks=[5,1],
        yticks=[50,10],
        )
    plot.saveToPDF(f"Template_Slice_{slice}") 

def plotLogSlice(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e, slice):
    plot = plotGen.newPlot(path,sizePerPlot = (3.4,2.4), rect=(0.08,0.09,0.995,0.995))
    height = array[:,slice]
    yedges = yedges + 0.5
    binCentres = (yedges[:-1] + yedges[1:]) / 2
    yedges[0] = 0.1
    plot.axs.stairs(height,np.log(yedges), color=plot.colorPalette[0], baseline=None,label = "Data")
    if estimate[slice]>0:
        x = np.linspace(np.log(yedges[0]),np.log(yedges[-1]),1000)
        y = gaussianFunc(np.log(np.exp(x)+0.5),estimate[slice], spread[slice], np.sum(height*(np.log(yedges[1:]+0.5)-np.log(yedges[:-1]+0.5))))
        plot.axs.plot(x,y,color=plot.colorPalette[2],label=f"Log Normal")#\n{estimate[slice]:.2f}$\\pm${estimate_e[slice]:.4f},{spread[slice]:.2f}$\\pm${spread_e[slice]:.4f}")
        #plot.axs.vlines(x[np.argmax(y)],0,np.max(y),label=f"MPV\n{x[np.argmax(y)]:.2f}",color=plot.colorPalette[5],linestyle="--")
    #plot.axs.set_xscale("log")
    plot.set_config(plot.axs,
        #title=f"Row vs TS Slice {slice}",
        xlabel="Log Relative Timestamp [Ln TS]",
        ylabel="Count",
        xlim = (-1,5),
        ylim = (0,None),
        legend=True,
        #xticks=[5,1],
        yticks=[50,10],
        )
    
    plot.saveToPDF(f"Template_Slice_{slice}_log") 



def runTemplate(dataFiles,plotGen,config):
    gen = colorGen()
    colorDict = dict(zip([dataFile.fileName for dataFile in dataFiles],[next(gen) for dataFile in dataFiles]))
    def getColor(dataFile):
        return colorDict[dataFile.fileName]
    maxRow = 25
    TSRange = 35
    combinedPlot = plotGen.newPlot("Combined/",sizePerPlot = (7,5))
    for dataFile in dataFiles:
        layer = 4
        path = f"PerfectClusters/{dataFile.fileName}/"
        clusters = dataFile.get_clusters(layer=layer,excludeCrossTalk = True)
        minExpectedClusterSize = 8
        minExpectedClusterSize = int(np.floor(np.sqrt((minExpectedClusterSize**2)*(dataFile.voltage/48.6))))
        if dataFile.angle != 86.5:
            minExpectedClusterSize = 2
        relativeRowList, relativeTSList = getRelativeRowTS(clusters,TSRange=TSRange,maxRow=maxRow,minExpectedClusterSize=minExpectedClusterSize,numberOfClustersUsed=1000)
        estimate, spread, estimate_e, spread_e = calcTemplate(relativeRowList[relativeRowList<=maxRow],relativeTSList[relativeRowList<=maxRow])
        #estimate, spread = dataFile.get_timeStampTemplate(layer=4)
        mpvs = np.array([findMPV(mu,sig) for mu,sig in zip(estimate[estimate>0],spread[estimate>0])])
        combinedPlot.axs.plot(
            np.arange(len(estimate))[estimate>0][1:],
            mpvs[1:],
            color=getColor(dataFile),
            label=f"{getName(dataFile)}",
        )
        combinedPlot.axs.errorbar(
            np.arange(len(estimate))[estimate>0],
            mpvs,
            yerr=abs(mpvs*(estimate_e[estimate>0]/estimate[estimate>0])),
            ls="",
            marker="s",
            color=getColor(dataFile),
        )
        array, yedges, xedges = np.histogram2d(relativeTSList,relativeRowList,range=((-0.5,TSRange+0.5),(-0.5,len(estimate)+0.5)),bins=(TSRange+1,len(estimate)+1))
        plotTemplate(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e)
        for slice in np.arange(0,len(estimate),4):
            plotSlice(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e, slice)
            plotLogSlice(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e, slice)
        #plotSlice(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e, 21)
        #plotLogSlice(plotGen,path,array, yedges, xedges,estimate, spread, estimate_e, spread_e, 21)
    combinedPlot.set_config(combinedPlot.axs,
        title="Row vs TS",
        xlabel="Relative Row [px]",
        ylabel="Relative Timestamp [TS]",
        xlim = (-0.5,None),
        ylim = (-0.5,None),
        legend=True,
        xticks=[5,1],
        yticks=[5,1],
        )
    combinedPlot.saveToPDF("Template") 
if __name__ == "__main__":
    config = configLoader.loadConfig()
    #config["filterDict"] = {"fileName": "angle6_4Gev_kit_2"}
    config["filterDict"] = {
        "fileName": [
            "angle6_6Gev_kit_4",
            "angle6_6Gev_kitHV30_kit_5",
            "angle6_6Gev_kitHV20_kit_6",
            "angle6_6Gev_kitHV15_kit_7",
            "angle6_6Gev_kitHV10_kit_8",
            "angle6_6Gev_kitHV8_kit_9",
            "angle6_6Gev_kitHV6_kit_10",
            "angle6_6Gev_kitHV4_kit_12",
            "angle6_6Gev_kitHV2_kit_13",
            "angle6_6Gev_kitHV0_kit_14",
        ]
    }
    config = configLoader.loadConfig()
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runTemplate(dataFiles[:10],plotGen,config)