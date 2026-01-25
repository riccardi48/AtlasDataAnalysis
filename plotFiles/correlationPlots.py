###############################
# Plots in this file:
# 1. RowRow correlation plot
# 2. RowRow correlation plot with only high -> low correlations
# 3. RowRow correlation plot with crosstalk removed
# 4. RowRow correlation plot with crosstalk mapping overlaid
###############################

from plotClass import plotGenerator
import sys

sys.path.append("..")
from dataAnalysis import dataAnalysis, initDataFiles, configLoader
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from typing import Optional
import numpy.typing as npt
from dataAnalysis.handlers._crossTalkFinder import crossTalkFinder
import matplotlib.pyplot as plt


def RowRowCorrelation(
    dataFile: dataAnalysis,
    config: dict,
    recalc: bool = False,
    excludeCrossTalk=False,
    highToLow=False,
    addVoltage=False,
):

    clusters = dataFile.get_clusters(layers=config["layers"])
    dataFile.get_crossTalk()
    RowRow = np.zeros((371, 371))
    print(f"Finding RowRow correlation")
    rows = dataFile.get_base_attr("Row")
    ToT = dataFile.get_base_attr("ToT")
    voltage = dataFile.get_base_attr("Hit_Voltage")
    indexes = rows - np.min(rows)
    for cluster in clusters:
        clusterIndexes = cluster.getIndexes(excludeCrossTalk=excludeCrossTalk)
        for pixel in clusterIndexes:
            if ToT[pixel] > 40 or not highToLow:
                RowRow[
                    indexes[pixel],
                    indexes[clusterIndexes],
                ] += (
                    ToT[clusterIndexes] if addVoltage else 1
                )
    RowRow[np.where(RowRow == 0)] = None
    return RowRow


def plotRowRow(
    dataFile,
    plotGen,
    config,
    path,
    recalc=False,
    excludeCrossTalk=False,
    log=False,
    showFunction=False,
    highToLow=False,
):
    plot = plotGen.newPlot(path, sizePerPlot=(6, 4))
    axs = plot.axs
    RowRow = RowRowCorrelation(
        dataFile, config, recalc=recalc, excludeCrossTalk=excludeCrossTalk, highToLow=highToLow
    )
    extent = (
        0.5,
        371.5,
        0.5,
        371.5,
    )
    norm = None
    if log:
        norm = LogNorm(vmin=1, vmax=np.max(RowRow, where=~np.isnan(RowRow), initial=-1))
    im = axs.imshow(RowRow, origin="lower", aspect="equal", extent=extent, norm=norm)
    plot.set_config(axs, title="RowRow Correlation", xlabel="Row [px]", ylabel="Row [px]")
    axs.xaxis.set_major_locator(MultipleLocator(100))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(20))
    axs.yaxis.set_major_locator(MultipleLocator(100))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(20))
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Frequency", rotation=270, labelpad=15)
    if showFunction:
        tempCrossTalkFinder = crossTalkFinder()
        x = []
        y = []
        for _x, _y in tempCrossTalkFinder.crossTalkFunction().items():
            for i, j in _y:
                x.append(i)
                y.append(j)

        axs.scatter(x, y, c="r", s=1)
    plot.saveToPDF(
        f"RowRowCorrelation"
        + f"{"_cut" if excludeCrossTalk else ""}"
        + f"{"_"+"".join(str(x) for x in config["layers"]) if config["layers"] is not None else ""}"
        + f"{"_log" if log else ""}"
        + f"{"_highToLow" if highToLow else ""}"
        + f"{"_showFunc" if showFunction else ""}"
    )


def plotRowRow3D(
    dataFile,
    plotGen,
    config,
    path,
    recalc=False,
    excludeCrossTalk=False,
    log=False,
    highToLow=False,
):
    plot = plotGen.newPlot(path, sizePerPlot=(12, 12))
    plot.fig = plt.figure()
    axs = plot.fig.add_subplot(111, projection="3d")
    RowRow = RowRowCorrelation(
        dataFile, config, recalc=recalc, excludeCrossTalk=excludeCrossTalk, highToLow=highToLow
    )
    RowRowVoltage = RowRowCorrelation(
        dataFile,
        config,
        recalc=recalc,
        excludeCrossTalk=excludeCrossTalk,
        highToLow=highToLow,
        addVoltage=True,
    )
    RowRowVoltage = RowRowVoltage / RowRow
    coordinates = np.where(RowRow)
    cmap = plt.get_cmap("plasma")
    im = axs.bar3d(
        coordinates[0],
        coordinates[1],
        np.zeros(coordinates[0].size),
        np.ones(coordinates[0].size),
        np.ones(coordinates[0].size),
        RowRow[coordinates],
        color=cmap(RowRowVoltage[coordinates] / np.max(RowRowVoltage)),
    )
    axs.set_ylim(372,0)
    axs.set_xlim(0,372)
    plot.saveToPNG(
        f"RowRowCorrelation_3d"
        + f"{"_cut" if excludeCrossTalk else ""}"
        + f"{"_"+"".join(str(x) for x in config["layers"]) if config["layers"] is not None else ""}"
        + f"{"_log" if log else ""}"
        + f"{"_highToLow" if highToLow else ""}",
        close=False
    )
    plot.saveToPDF(
        f"RowRowCorrelation_3d"
        + f"{"_cut" if excludeCrossTalk else ""}"
        + f"{"_"+"".join(str(x) for x in config["layers"]) if config["layers"] is not None else ""}"
        + f"{"_log" if log else ""}"
        + f"{"_highToLow" if highToLow else ""}"
    )


def runCorrelation(
    dataFiles: dataAnalysis,
    plotGen,
    config,
):
    for i, dataFile in enumerate(dataFiles):
        path = f"Correlation/{dataFile.fileName}/"
        plotRowRow3D(
            dataFile,
            plotGen,
            config,
            path,
            recalc=False,
            excludeCrossTalk=False,
            log=False,
            highToLow=False,
        )
        plotRowRow3D(
            dataFile,
            plotGen,
            config,
            path,
            recalc=False,
            excludeCrossTalk=False,
            log=False,
            highToLow=True,
        )
        plotRowRow(
            dataFile,
            plotGen,
            config,
            path,
            recalc=True,
            excludeCrossTalk=False,
            log=False,
            showFunction=False,
            highToLow=False,
        )
        plotRowRow(
            dataFile,
            plotGen,
            config,
            path,
            recalc=True,
            excludeCrossTalk=False,
            log=False,
            showFunction=False,
            highToLow=True,
        )
        plotRowRow(
            dataFile,
            plotGen,
            config,
            path,
            recalc=True,
            excludeCrossTalk=True,
            log=False,
            showFunction=False,
            highToLow=False,
        )
        plotRowRow(
            dataFile,
            plotGen,
            config,
            path,
            recalc=True,
            excludeCrossTalk=False,
            log=False,
            showFunction=True,
            highToLow=False,
        )


if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
    config["filterDict"] = {"fileName":"4Gev_kit_1"}
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runCorrelation(dataFiles, plotGen, config)
