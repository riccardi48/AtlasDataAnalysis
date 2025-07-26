from plotAnalysis import plotClass, clusterPlotter
from dataAnalysis import dataAnalysis, initDataFiles
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
import configLoader


def Clusters(
    dataFile: dataAnalysis,
    pathToOutput,
    excludeCrossTalk=True,
    z="Hit_Voltages",
    layer=4,
    saveToPDF=True,
):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/", sizePerPlot=(20, 20))
    axs = plot.axs
    plotter = clusterPlotter(dataFile, excludeCrossTalk=excludeCrossTalk)
    clusters = dataFile.get_clusters(layer=layer, excludeCrossTalk=excludeCrossTalk)
    dataFile.init_cluster_voltages()
    firstTS = clusters[1000].getTSs(excludeCrossTalk=excludeCrossTalk)[0]
    # firstTS = dataFile.get_base_attr("ext_TS", excludeCrossTalk=excludeCrossTalk)[50000]
    lastTs = firstTS + 100 / (25 / 1000000)
    TSs = np.array([cluster.getTSs(excludeCrossTalk=excludeCrossTalk)[0] for cluster in clusters])
    clusters = clusters[(TSs <= lastTs) & (TSs >= firstTS)]
    # clusters = dataFile.get_clusters(layer=layer,excludeCrossTalk=excludeCrossTalk)[:40]
    im = plotter.plotClusters(axs, clusters, z=z)
    plot.set_config(axs, title="Clusters", legend=False, xlabel="Row [px]", ylabel="Column [px]")
    axs.xaxis.set_major_locator(MultipleLocator(10))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(2))
    axs.yaxis.set_major_locator(MultipleLocator(5))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1))
    cbar = plt.colorbar(im, ax=axs)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(f"{z}", rotation=270, fontsize="large")
    if saveToPDF:
        plot.saveToPDF(
            "Clusters/Clusters"
            + f"{f"_{layer}" if layer is not None else ""}{"_cut" if excludeCrossTalk else ""}"
        )
    else:
        return plot.fig


config = configLoader.loadConfig()

dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    Clusters(dataFile, config["pathToOutput"], z="TSs", layer=1, excludeCrossTalk=True)
    Clusters(dataFile, config["pathToOutput"], z="TSs", layer=2, excludeCrossTalk=True)
    Clusters(dataFile, config["pathToOutput"], z="TSs", layer=3, excludeCrossTalk=True)
    Clusters(dataFile, config["pathToOutput"], z="TSs", layer=4, excludeCrossTalk=True)
    Clusters(dataFile, config["pathToOutput"], z="Layer", layer=None, excludeCrossTalk=True)
