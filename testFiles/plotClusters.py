import sys

sys.path.append("..")

from plotAnalysis import plotClass, clusterPlotter
from dataAnalysis import dataAnalysis, initDataFiles, configLoader
from matplotlib.ticker import MultipleLocator
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def TStoMS(TS: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    return TS * 25 / 1000000


def MStoTS(Time: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    return np.round(Time * 1000000 / 25).astype(np.int_)

def Clusters(
    dataFile: dataAnalysis,
    pathToOutput,
    excludeCrossTalk=True,
    z="Hit_Voltages",
    layer=4,
    saveToPDF=True,
    name = "",
):
    plot = plotClass(pathToOutput, sizePerPlot=(20, 20))
    axs = plot.axs
    plotter = clusterPlotter(dataFile, excludeCrossTalk=excludeCrossTalk)
    clusters = dataFile.get_clusters(excludeCrossTalk=excludeCrossTalk)
    dataFile.init_cluster_voltages()
    #firstTS = clusters[1000].getEXT_TSs(excludeCrossTalk=excludeCrossTalk)[0]
    # firstTS = dataFile.get_base_attr("ext_TS", excludeCrossTalk=excludeCrossTalk)[50000]
    #lastTS = firstTS + 300 / (25 / 1000000)
    firstTS = 135000
    lastTS = 135300

    #TSs = np.array([cluster.getEXT_TSs(excludeCrossTalk=excludeCrossTalk)[0] for cluster in clusters])
    #TSs = TSs-np.min(TSs)
    TSs,indexes = dataFile.get_cluster_attr("Times", layer=layer, excludeCrossTalk=excludeCrossTalk,returnIndexes=True)
    usedClusters = clusters[indexes[(TSs <= lastTS) & (TSs >= firstTS)]]
    if len(usedClusters) == 0:
        return
    print(len(usedClusters))
    # clusters = dataFile.get_clusters(layer=layer,excludeCrossTalk=excludeCrossTalk)[:40]
    im = plotter.plotClusters(axs, usedClusters, z=z)
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
            f"Clusters/"
            + f"Clusters_"
            + f"{dataFile.fileName}"
            + f"{f"_{layer}" if layer is not None else ""}{"_cut" if excludeCrossTalk else ""}"
            +f"{f"_{name}" if name != "" else ""}"
        )
    else:
        return plot.fig


config = configLoader.loadConfig()
config["pathToOutput"] = "/home/atlas/rballard/AtlasDataAnalysis/output/TimeTests/"
config["filterDict"] = {"telescope": "kit", "angle": 45}
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    Clusters(dataFile, config["pathToOutput"], z="TSs", layer=4, excludeCrossTalk=True,name="TSs")
    Clusters(dataFile, config["pathToOutput"], z="EXT_TSs", layer=4, excludeCrossTalk=True,name="EXT_TSs")
    Clusters(dataFile, config["pathToOutput"], z="Hit_Voltages", layer=4, excludeCrossTalk=True,name="Hit_Voltages")
    Clusters(dataFile, config["pathToOutput"], z="Index", layer=4, excludeCrossTalk=True,name="Index")
