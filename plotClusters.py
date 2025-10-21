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
    clusters,
    z="Hit_Voltages",
    layer=4,
    saveToPDF=True,
    name = "",
):
    plot = plotClass(pathToOutput+"ClusterPlot/", sizePerPlot=(20, 20))
    axs = plot.axs
    plotter = clusterPlotter(dataFile, excludeCrossTalk=True)
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
            f"Clusters_"
            + f"{dataFile.fileName}"
            + f"{f"_{layer}" if layer is not None else ""}"
            +f"{f"_{name}" if name != "" else ""}"
        )
    else:
        return plot.fig

def checkClusterLength(cluster):
    return cluster.getRowWidth(True)>22&cluster.getRowWidth(True)<28

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    layer = 4
    dataFile.init_cluster_voltages()
    clusters = dataFile.get_clusters(excludeCrossTalk=True,layer=layer)
    clusters = clusters[[checkClusterLength(cluster) for cluster in clusters]]
    Clusters(dataFile, config["pathToOutput"],clusters[:50], z="EXT_TSs",layer=layer,name="long_EXT_TSs")
    Clusters(dataFile, config["pathToOutput"],clusters[:50], z="Hit_Voltages",layer=layer,name="long_Hit_Voltages")
    Clusters(dataFile, config["pathToOutput"],clusters[:50], z="Index",layer=layer,name="long_Index")
