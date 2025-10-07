import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
import numpy as np
import numpy.typing as npt
from landau import landau
import scipy

def landauFunc(
    x: npt.NDArray[np.float64],
    x_mpv: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    scaler: npt.NDArray[np.float64],
    threshold: float = 0.16,
) -> npt.NDArray[np.float64]:
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    y[x < threshold] = 0
    return y


config = configLoader.loadConfig()
config["filterDict"] = {"telescope":"kit","fileName":"6Gev_kit_0"}
dataFiles = initDataFiles(config)

for i, dataFile in enumerate(dataFiles):
    plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
    axs = plot.axs
    dataFile.init_cluster_voltages()
    clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=config["layers"][0])])
    clusterCharges = clusterCharges[clusterCharges > 0]
    height, x = np.histogram(clusterCharges, bins=150, range=(0, 30))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(0, 60),
        title=f"Cluster Charge Distribution {dataFile.fileName}",
        xlabel="Charge [V]",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"ClusterCharges_{dataFile.fileName}")
    binCentres = (x[:-1] + x[1:]) / 2
    popt, pcov = scipy.optimize.curve_fit(
            landauFunc,
            binCentres,
            height,
            maxfev=500000,
        )
    averageCharge = popt[0]
    print(averageCharge,np.mean(clusterCharges))

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for i, dataFile in enumerate(dataFiles):
    plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
    axs = plot.axs
    dataFile.init_cluster_voltages()
    clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=config["layers"][0])])
    longClusters = np.array([cluster.getIndexes().size for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=config["layers"][0])])>15
    clusterCharges = clusterCharges[longClusters]
    clusterCharges = clusterCharges[clusterCharges > 0]
    height, x = np.histogram(clusterCharges, bins=150, range=(0, 30))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(0, 30),
        title=f"Cluster Charge Distribution {dataFile.fileName}",
        xlabel="Charge [V]",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"ClusterCharges_Long_{dataFile.fileName}")
    plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
    axs = plot.axs
    angles = np.rad2deg(np.arcsin(averageCharge/clusterCharges))
    angles = angles[~np.isnan(angles) & ~np.isinf(angles) & (angles>=0) & (angles<=90)]
    height, x = np.histogram(90-angles, bins=150, range=(50, 90))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(0, None),
        title=f"Cluster Charge Distribution {dataFile.fileName}",
        xlabel="Angle [degrees]",
        ylabel="Frequency",
        )
    axs.vlines(
        dataFile.angle, 0, axs.get_ylim()[1], colors=plot.textColor, linestyles="dashed"
    )
    plot.saveToPDF(f"ClusterAngles_Long_{dataFile.fileName}")