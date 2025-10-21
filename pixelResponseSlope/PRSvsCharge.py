from PRSFuncs import loadOrCalcPRS,sumOfCauchyFunc,cauchyFunc,plotHist,plotHistRemoved
import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import numpy.typing as npt
from dataAnalysis._fileReader import calcDataFileManager
from landau import landau

def landauFunc(
    x: npt.NDArray[np.float64],
    x_mpv: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    scaler: npt.NDArray[np.float64],
    #threshold: float = 0.16,
) -> npt.NDArray[np.float64]:
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    #y[x < threshold] = 0
    return y

config = configLoader.loadConfig()
config["filterDict"] = {"telescope":"kit","fileName":["4Gev_kit_1"]}
dataFiles = initDataFiles(config)


for i, dataFile in enumerate(dataFiles):
    layer = 4
    dataFile.init_cluster_voltages()
    clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=layer)])
    clusterCharges = clusterCharges[clusterCharges > 0]
    height, x = np.histogram(clusterCharges, bins=1000, range=(0, 20))
    binCentres = (x[:-1] + x[1:]) / 2
    popt, pcov = curve_fit(
            landauFunc,
            binCentres,
            height,
    )
    averageCharge = popt[0]

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    dataFile.init_cluster_voltages()
    clusters,indexes = dataFile.get_clusters(excludeCrossTalk=True,returnIndexes=True,layer=4)
    _range = (-0.1005,0.1005)
    bins = 201
    xlim = (-0.1,0.1)
    PRS = loadOrCalcPRS(dataFile,config)
    indexes = indexes[dataFile.get_cluster_attr("Sizes",excludeCrossTalk=True,layer=4)[0]>0]
    clusters = clusters[dataFile.get_cluster_attr("Sizes",excludeCrossTalk=True,layer=4)[0]>0]
    clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in clusters])
    clusterAngle = 90-np.rad2deg(np.arcsin(averageCharge/clusterCharges))
    PRS = PRS[indexes]
    array, yedges, xedges = np.histogram2d(clusterAngle,PRS,bins=(40,401),range=((70,90),_range))
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/ClusterCharge/")
    axs = plot.axs
    axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    axs.hlines(
        dataFile.angle, axs.get_xlim()[0], axs.get_xlim()[1], colors=plot.textColor, linestyles="dashed"
    )
    plot.set_config(
        axs,
        title=f"PRS vs ClusterAngle from charge {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Angle",
        )
    plot.saveToPDF(f"PRS_Angle_Scatter_{dataFile.fileName}")
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/ClusterCharge/")
    axs = plot.axs
    row_sums = array.sum(axis=1)
    array = array / row_sums[:, np.newaxis]
    axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    axs.hlines(
        dataFile.angle, axs.get_xlim()[0], axs.get_xlim()[1], colors=plot.textColor, linestyles="dashed"
    )
    plot.set_config(
        axs,
        title=f"PRS vs ClusterAngle from charge {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Angle",
        )
    plot.saveToPDF(f"PRS_Angle_Scatter_{dataFile.fileName}_norm")
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/ClusterCharge/")
    axs = plot.axs
    array, yedges, xedges = np.histogram2d(clusterCharges,PRS,bins=(40,401),range=((0,20),_range))
    axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    plot.set_config(
        axs,
        title=f"PRS vs ClusterAngle from charge {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Cluster Charge",
        )
    plot.saveToPDF(f"PRS_Charge_Scatter_{dataFile.fileName}")
