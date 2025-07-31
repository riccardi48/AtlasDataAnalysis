from plotAnalysis import depthAnalysis, plotClass, correlationPlotter, fitAndPlotCCE, fit_dataFile
from dataAnalysis import dataAnalysis, crossTalkFinder, initDataFiles
from lowLevelFunctions import (
    calcDepth,
    adjustPeakVoltage,
    histogramErrors,
    landauFunc,
    lambert_W_ToT_to_u,
    chargeCollectionEfficiencyFunc,
    print_mem_usage,
    checkDirection,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from typing import Optional
from landau import landau
import scipy as scipy


def Hit_VoltageDistributionByPixel(
    dataFile: dataAnalysis,
    depth: depthAnalysis,
    clusterWidth: int,
    pathToOutput: str,
    _range: tuple[float, float] = (-5.5, 25.5),
    measuredAttribute: str = "Hit_Voltage",
    saveToPDF: bool = True,
):
    plot = plotClass(
        pathToOutput + f"{dataFile.get_fileName()}/",
        shape=(1, clusterWidth),
        sharex=True,
        sharey=True,
        sizePerPlot=(10, 2),
        hspace=0,
    )
    axs = np.flip(plot.axs)
    hitPositionArray, Indexes = depth.loadOneLength(
        dataFile, clusterWidth, measuredAttribute=measuredAttribute,returnIndexes=True
    )
    hitPositionErrorArray, _ = depth.loadOneLength(
        dataFile, clusterWidth, error=True, measuredAttribute=measuredAttribute
    )
    times = np.zeros(hitPositionArray.shape)
    times = times - 100
    for i,cluster in enumerate(dataFile.get_clusters(excludeCrossTalk=True)[Indexes]):
        values = cluster.getTSs(excludeCrossTalk=True) - np.min(cluster.getTSs(excludeCrossTalk=True))
        values[values > 400] -=1024
        x = cluster.getRows(excludeCrossTalk=True)
        index = (x - np.min(x)).astype(int)
        if checkDirection(values, x, cluster.getRowWidth(excludeCrossTalk=True)):
            values = np.flip(values)
            index = np.flip(index)
        times[index,i] = values
    d = depth.find_d_value(dataFile)
    x = calcDepth(
            d,
            clusterWidth,
            dataFile.get_angle(),
            depthCorrection=True,
            upTwo=False,
        )
    for j in range(clusterWidth):
        values = times[j][times[j] != -100]
        values = values[np.invert(np.isnan(values))]
        hist, binEdges = np.histogram(values, bins=31, range=_range)
        binCentres = (binEdges[:-1] + binEdges[1:]) / 2

        axs[j].step(
            binEdges,
            np.append(hist[0], hist),
            c=plot.colorPalette[3],
            linewidth=1,
            label=f"Pixel {clusterWidth-j}\nEffective Depth {x[j]:.1f} μm",
        )
        axs[j].get_xaxis().set_visible(False)
        plot.set_config(
            axs[j],
            xlim=(float(np.min(binEdges)), float(np.max(binEdges))),
            legend=True,
        )
        axs[j].text(
            0,
            hist[binCentres==0][0],
            f"{hist[binCentres==0][0]}",
            color=plot.textColor,
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        axs[j].text(
            5,
            hist[binCentres==0][0]*2,
            f"{np.sum(hist[binCentres>5])}",
            color=plot.colorPalette[5],
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    axs[1].set_ylim(0,None)
    axs[0].set_xlabel("TS Diff [TS]")
    axs[0].get_xaxis().set_visible(True)
    axs[0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_major_formatter("{x:.0f}")
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[-1].set_xlabel("TS Diff [TS]")
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].xaxis.set_label_position("top")
    axs[-1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[-1].xaxis.set_major_locator(MultipleLocator(5))
    axs[-1].xaxis.set_major_formatter("{x:.0f}")
    axs[-1].xaxis.set_minor_locator(MultipleLocator(1))

    plot.fig.suptitle(f"{clusterWidth} Width TS difference By Pixel")
    if saveToPDF:
        plot.saveToPDF(
            f"VoltageDepth/ByWidth/TS/DiffTSDistributionByPixel_{clusterWidth}"
        )
    else:
        return plot.fig
    
if __name__ == "__main__":
    import configLoader

    config = configLoader.loadConfig()
    dataFiles = initDataFiles(config)
    for dataFile in dataFiles:
        depth = depthAnalysis(
            config["pathToCalcData"],
            maxLine=config["maxLine"],
            maxClusterWidth=config["maxClusterWidth"],
            layers=config["layers"],
            excludeCrossTalk=config["excludeCrossTalk"],
        )
        iList = [3, 5, 8, 11, 13, 15, 18, 20, 22, 24, 25, 27]
        for i in iList:
            Hit_VoltageDistributionByPixel(dataFile,depth,i,config["pathToOutput"])