###############################
# Plots in this file:
# 1. Plot of efficiency at each relative Row
# 2. Same as above with smaller y axis
# 3. Combined Plot with all included dataFiles
###############################
from plotClass import plotGenerator
from functions.efficiencyFuncs import calcEfficiency, getPercentFromDict
from functions.genericFuncs import getName,colorGen
import sys

sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader


def runEfficiency(dataFiles, plotGen, config):
    gen = colorGen()
    colorDict = dict(zip([dataFile.fileName for dataFile in dataFiles],[next(gen) for dataFile in dataFiles]))
    
    def getColor(dataFile):
        return colorDict[dataFile.fileName]
    maxWidth = 35
    bigPlot = plotGen.newPlot("Combined/", sizePerPlot=(7, 5), rect=(0.04, 0.04, 0.995, 0.995))
    for i, dataFile in enumerate(dataFiles):
        path = f"Efficiency/{dataFile.fileName}/"
        clusters = dataFile.get_perfectClusters(excludeCrossTalk=True, layers=config["layers"])
        print(len(clusters))
        efficiencyDict = calcEfficiency(clusters, maxWidth=maxWidth)
        pList, errors = getPercentFromDict(efficiencyDict, maxWidth=maxWidth)
        plot = plotGen.newPlot(path, sizePerPlot=(3.4, 2.5), rect=(0.09,0.09,0.995,0.995),)
        axs = plot.axs
        axs.scatter(
            np.arange(pList.size),
            pList,
            color=plot.colorPalette[0],
            marker=".",
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
            # title="Efficiency by relative Row",
            xlabel="Relative Row",
            ylabel="Efficiency",
            legend=True,
            ylim=(0.97, 1),
            xticks=[5, 1],
            yticks=[0.01, 0.002],
            yticksSig=2,
            grid=True,
        )
        plot.saveToPDF(f"Efficiency_Relative_Row_ShortAxis_{dataFile.voltage}V", close=False)

        plot.set_config(
            axs,
            # title="Efficiency by relative Row",
            xlabel="Relative Row",
            ylabel="Efficiency",
            legend=True,
            ylim=(0, 1),
            xticks=[5, 1],
            yticks=[0.1, 0.02],
            yticksSig=1,
            grid=True,
        )
        plot.saveToPDF(f"Efficiency_Relative_Row_{dataFile.voltage}V")

        bigPlot.axs.plot(
            np.arange(pList.size),
            pList,
            color=getColor(dataFile),
            label=f"{getName(dataFile)}",
        )
        bigPlot.axs.errorbar(
            np.arange(pList.size),
            pList,
            errors,
            ls="",
            marker="s",
            color=getColor(dataFile),
        )
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
        # title="Efficiency by Pixel in Cluster",
        xlabel="Distance from Seed",
        ylabel="Efficiency",
        legend=True,
        ylim=(0, 1),
        xticks=[5, 1],
        yticks=[0.1, 0.02],
        yticksSig=1,
        # ncols=2,
        # loc="lower left",
        grid=True,
    )

    bigPlot.saveToPDF(f"Efficiency_Relative_Row", close=False)

    bigPlot.set_config(
        bigPlot.axs,
        # title="Efficiency by relative Row",
        xlabel="Row",
        ylabel="Efficiency",
        legend=True,
        ylim=(0.97, 1),
        xticks=[5, 1],
        yticks=[0.005, 0.001],
        yticksSig=3,
        # ncols=2,
        # loc="lower left",
        grid=True,
    )
    bigPlot.saveToPDF(f"Efficiency_Relative_Row_ShortAxis")


if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
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
    dataFiles = initDataFiles(config)
    
    # dataFiles = [dataFiles[0]] + dataFiles[2:]
    plotGen = plotGenerator(config["pathToOutput"])
    runEfficiency(dataFiles, plotGen, config)
