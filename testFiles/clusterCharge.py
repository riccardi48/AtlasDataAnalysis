import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader,printMemUsage
from plotAnalysis import plotClass
import numpy as np
import numpy.typing as npt
from landau import landau,langauss
import scipy

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

def langaussFunc(
    x: npt.NDArray[np.float64],
    x_mpv: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    scaler: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    #threshold: float = 0.16,
) -> npt.NDArray[np.float64]:
    y = langauss.pdf(x, x_mpv, xi,sigma) * scaler
    y = np.reshape(y, np.size(y))
    #y[x < threshold] = 0
    return y

def chiSquared(expected,observed):
    chiArray = (observed-expected)**2/expected
    chiArray[expected<1] = 0
    chiArray[np.isnan(chiArray)|np.isinf(chiArray)] = 0
    print(observed[100:110])
    print(expected[100:110])
    print(chiArray[100:110])
    return np.sum(chiArray)

config = configLoader.loadConfig()
config["filterDict"] = {"telescope":"kit","fileName":["4Gev_kit_1"]}
dataFiles = initDataFiles(config)
for i, dataFile in enumerate(dataFiles):
    for layer in [1,2,3,4]:
        plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
        axs = plot.axs
        dataFile.init_cluster_voltages()
        printMemUsage()
        clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=layer)])
        printMemUsage()
        clusterCharges = clusterCharges[clusterCharges > 0]
        height, x = np.histogram(clusterCharges, bins=1000, range=(0, 20))
        axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
        binCentres = (x[:-1] + x[1:]) / 2
        popt, pcov = scipy.optimize.curve_fit(
                landauFunc,
                binCentres,
                height,
        )
        _x = np.linspace(0,5,1000)
        y = landauFunc(_x, *popt)
        axs.plot(_x,y, color=plot.colorPalette[2], label=f"Fit: $x_{{mpv}}$={popt[0]:.2f} V, $\\xi$={popt[1]:.2f} V\n$\\chi^{2}$={chiSquared(landauFunc(binCentres, *popt),height):.2f}")
        averageCharge = popt[0]
        print(averageCharge,np.mean(clusterCharges))
        print(popt)
        plot.set_config(
            axs,
            ylim=(0, None),
            xlim=(0, 5),
            title=f"Cluster Charge Distribution {dataFile.fileName}\nLayer {layer}",
            xlabel="Charge [V]",
            ylabel="Count",
            legend=True,
            )
        plot.saveToPDF(f"ClusterCharges_{dataFile.fileName}_{layer}")
        print(chiSquared(landauFunc(binCentres[binCentres>0.2], *popt),height[binCentres>0.2]))
        plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
        axs = plot.axs
        printMemUsage()
        clusterCharges = np.array([np.sum(cluster.getToTs(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=layer)])
        printMemUsage()
        clusterCharges = clusterCharges[clusterCharges > 0]
        height, x = np.histogram(clusterCharges, bins=1001, range=(-0.5, 1000.5))
        axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    
        
        binCentres = (x[:-1] + x[1:]) / 2
        popt, pcov = scipy.optimize.curve_fit(
                landauFunc,
                binCentres,
                height,
            )
        _x = np.linspace(0,1000,1000)
        y = landauFunc(_x, *popt)
        axs.plot(_x, y, color=plot.colorPalette[2], label=f"Fit: $x_{{mpv}}$={popt[0]:.2f} V, $\\xi$={popt[1]:.2f} V\n$\\chi^{2}$={chiSquared(landauFunc(binCentres, *popt),height):.2f}")
        plot.set_config(
            axs,
            ylim=(0, None),
            xlim=(0, 600),
            title=f"Cluster Charge Distribution {dataFile.fileName}\nLayer {layer}",
            xlabel="ToT [TS]",
            ylabel="Count",
            legend=True,
            )
        plot.saveToPDF(f"ClusterCharges_ToT_{dataFile.fileName}_{layer}")
        print(chiSquared(landauFunc(binCentres, *popt),height))

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for i, dataFile in enumerate(dataFiles):
    plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
    axs = plot.axs
    dataFile.init_cluster_voltages()
    clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=config["layers"][0])])
    printMemUsage()
    #longClusters = np.array([cluster.getIndexes().size for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=config["layers"][0])])>15
    #clusterCharges = clusterCharges[longClusters]
    clusterCharges = clusterCharges[clusterCharges > 0]
    height, x = np.histogram(clusterCharges, bins=150, range=(0, 30))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(0, 30),
        title=f"Cluster Charge Distribution {dataFile.fileName}",
        xlabel="Charge [V]",
        ylabel="Count",
        )
    plot.saveToPDF(f"ClusterCharges_Long_{dataFile.fileName}")
    plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
    axs = plot.axs
    angles = np.rad2deg(np.arcsin(averageCharge/clusterCharges))
    angles = angles[~np.isnan(angles) & ~np.isinf(angles) & (angles>=0) & (angles<=90)]
    height, x = np.histogram(90-angles, bins=181, range=(0, 90))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(0, 90),
        title=f"Cluster Charge Distribution {dataFile.fileName}",
        xlabel="Angle [degrees]",
        ylabel="Count",
        )
    axs.vlines(
        dataFile.angle, 0, axs.get_ylim()[1], colors=plot.textColor, linestyles="dashed"
    )
    plot.saveToPDF(f"ClusterAngles_{dataFile.fileName}")



    