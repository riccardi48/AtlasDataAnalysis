###############################
# Plots in this file:
# 1. Plot of efficiency at each relative Row
# 2. Same as above with smaller y axis
# 3. Combined Plot with all included dataFiles
###############################
from plotClass import plotGenerator
from functions.efficiencyFuncs import calcEfficiency,getPercentFromDict
from functions.genericFuncs import getColor,getName
import sys
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader

def runEfficiency(dataFiles,plotGen,config):
    bigPlot = plotGen.newPlot("Combined/",sizePerPlot=(8,6))
    for i,dataFile in enumerate(dataFiles):
        path = f"Efficiency/{dataFile.fileName}/"
        clusters = dataFile.get_perfectClusters(excludeCrossTalk = True,layers=config["layers"])
        print(len(clusters))
        efficiencyDict = calcEfficiency(clusters,maxWidth=30)
        pList,errors = getPercentFromDict(efficiencyDict)
        plot = plotGen.newPlot(path,sizePerPlot = (6,4))
        axs = plot.axs
        axs.scatter(
            np.arange(pList.size),
            pList,
            color=plot.colorPalette[0],
            marker="x",
        )
        axs.errorbar(
            np.arange(pList.size),
            pList,
            yerr=errors,
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
        plot.set_config(
            axs,
            title="Efficiency by relative Row",
            xlabel="Relative Row",
            ylabel="Efficiency",
            legend=True,
            ylim=(0.97, 1),
            xticks=[5,1],
            yticks=[0.005,0.001],
            yticksSig=3,
        )
        plot.axs.grid(True)
        plot.saveToPDF(f"Efficiency_Relative_Row_ShortAxis",close=False)

        plot.set_config(
            axs,
            title="Efficiency by relative Row",
            xlabel="Relative Row",
            ylabel="Efficiency",
            legend=True,
            ylim=(0, 1),
            xticks=[5,1],
            yticks=[0.1,0.02],
            yticksSig=1,
        )
        plot.saveToPDF(f"Efficiency_Relative_Row")

        bigPlot.axs.plot(
            np.arange(pList.size),
            pList,
            color=getColor(dataFile),
            label=f"{getName(dataFile)}",
        )
        bigPlot.axs.errorbar(np.arange(pList.size), pList, errors, ls='', marker='s',color=getColor(dataFile),)
        """     
        bigPlot.axs.fill_between(
            np.arange(pList.size),
            pList+errors[1],
            pList-errors[0],
            alpha=0.2,
            color=getColor(dataFile),
        )
        """
    bigPlot.set_config(
        bigPlot.axs,
        title="Efficiency by relative Row",
        xlabel="Row",
        ylabel="Efficiency",
        legend=True,
        ylim=(0, 1),
        xticks=[5,1],
        yticks=[0.1,0.02],
        yticksSig=1,
        #ncols=2,
        #loc="lower left",
    )
    bigPlot.axs.grid(True)
    bigPlot.saveToPDF(f"Efficiency_Relative_Row",close=False)

    bigPlot.set_config(
        bigPlot.axs,
        title="Efficiency by relative Row",
        xlabel="Row",
        ylabel="Efficiency",
        legend=True,
        ylim=(0.97, 1),
        xticks=[5,1],
        yticks=[0.005,0.001],
        yticksSig=3,
        #ncols=2,
        #loc="lower left",
    )
    bigPlot.saveToPDF(f"Efficiency_Relative_Row_ShortAxis")

if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runEfficiency(dataFiles,plotGen,config)



