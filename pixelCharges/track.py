from .plotCluster import clusterPlotter
from .orthClusterCharge import getOrthClusterCharge
from .funcs import (
    angle_with_error_mc,
    characterizeCluster,
    typeDict,
    isTypes,
    calcClusterLength,
    findBestLine,
    findValidLines,
)
import sys

sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import numpy as np
from scipy.stats import linregress
from landau import landau
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
from dataAnalysis._fileReader import calcDataFileManager


def RowsToMicroMeter(rows):
    return rows * 50  # 50 micrometer per row


def MicroMeterToRows(micrometer):
    return micrometer / 50  # 50 micrometer per row


def ColumnsToMicroMeter(columns):
    return columns * 150  # 150 micrometer per column


def MicroMeterToColumns(micrometer):
    return micrometer / 150  # 150 micrometer per column


def line(x, slope, intercept):
    return slope * x + intercept


def angleFromCharge(orthCharge, orthCharge_e, charge, charge_e):
    return angle_with_error_mc(orthCharge, orthCharge_e, charge, charge_e)
    if charge < orthCharge:
        if charge + charge_e > orthCharge:
            charge = orthCharge + 0.0001
        else:
            return np.nan, np.nan
    angle = 90 - np.rad2deg(np.arcsin(orthCharge / charge))
    angle_e = np.rad2deg(
        1
        / np.sqrt((1 - ((orthCharge**2) / (charge**2))))
        * np.sqrt((orthCharge_e**2 / charge**2) + ((orthCharge**2 * charge_e**2) / charge**4))
    )
    return angle, angle_e


def applyTimeWalkCorrection(voltages):
    a = 35
    b = 2000
    return b * np.exp(-a * voltages)

if __name__ == "__main__":

    orthCharge, orthCharge_e = getOrthClusterCharge(layer=4)
    orthCharge2, orthCharge_e2 = 1.87, 0.0039
    # print(orthCharge, orthCharge_e)
    config = configLoader.loadConfig()
    # config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
    config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
    dataFiles = initDataFiles(config)

    for dataFile in dataFiles:
        base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
        dataFile.init_cluster_voltages()
        clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
        i = 0
        for cluster in clusters[20000:30000:5]:
            if i > 100:
                break
            characterizeCluster(cluster)
            if not isTypes("2", [cluster.clusterType]) or cluster.getRowWidth(True) < 25:
                continue
            path = base_path + f"Cluster_{cluster.getIndex()}/"
            Timestamps = cluster.getTSs(True)
            TS = Timestamps - np.min(Timestamps)
            Rows = cluster.getRows(excludeCrossTalk=True)
            Columns = cluster.getColumns(excludeCrossTalk=True)
            #result = linregress(RowsToMicroMeter(Rows), ColumnsToMicroMeter(Columns))
            slope,intercept = findBestLine(cluster,findValidLines(Rows,Columns))
            x = np.linspace(0, 372, 372)
            y = MicroMeterToColumns(line(RowsToMicroMeter(x), slope, intercept))
            angle, angle_e = angleFromCharge(
                orthCharge,
                orthCharge_e,
                cluster.getClusterCharge(True),
                cluster.getClusterChargeError(True),
            )
            string = "".join(
                [value if key in cluster.clusterType else "" for key, value in typeDict.items()]
            )
            if isTypes("369~4", [cluster.clusterType]):
                string += "Perfect Cluster\n"
            label = f"Slope: {slope:.2f}\nAngle: {angle:.2f} ± {angle_e:.2f} Degrees\nCharge: {cluster.getClusterCharge(True):.2f} ± {cluster.getClusterChargeError(True):.2f} V\nLength: {calcClusterLength(cluster.getRows(True),slope,intercept):.2f} µm"

            CP = clusterPlotter(cluster, path, "Cluster Map")
            """
            CP.plot.axs.plot(
                x,
                y,
                color=CP.plot.colorPalette[5],
                label=label,
            )
            CP.plot.axs.text(
                0.01,
                0.99,
                string,
                horizontalalignment="left",
                verticalalignment="top",
                transform=CP.plot.axs.transAxes,
                fontsize="small",
            )
            """
            CP.finishPlot("Voltage", cluster.getHit_Voltages(True), textLabels=True, cmap="hot")

            CP = clusterPlotter(cluster, path, "Relative TS")
            """
            CP.plot.axs.plot(
                x,
                y,
                color=CP.plot.colorPalette[5],
                label=label,
            )
            
            CP.plot.axs.text(
                0.01,
                0.99,
                string,
                horizontalalignment="left",
                verticalalignment="top",
                transform=CP.plot.axs.transAxes,
                fontsize="small",
            )
            """
            CP.finishPlot("Relative TS", TS, cmap="plasma_r")

            plot = plotClass(path)
            axs = plot.axs
            axs.scatter(abs((Rows-np.min(Rows))-np.max((Rows-np.min(Rows)))), TS, color=plot.colorPalette[2], marker="x",label="Cluster TS")
            calcFileManager = calcDataFileManager(config["pathToCalcData"], "TSParams", config["maxLine"])
            calcFileName = calcFileManager.generateFileName(
                attribute=f"{dataFile.fileName}",
            )
            estimate,spread = calcFileManager.loadFile(calcFileName=calcFileName)
            x = np.arange(np.max(Rows-np.min(Rows))+1)
            index = x
            x = x[index<=30]
            index = index[index<=30]
            spread = spread[index]
            estimate = estimate[index]-0.5
            
            axs.plot(x,estimate,color=plot.colorPalette[0],linestyle="dashed",label="Expected TS")
            axs.fill_between(x,estimate-spread,estimate+spread, alpha=0.2,color=plot.colorPalette[0])
            plot.set_config(axs,
                title="Relative TS in cluster",
                xlabel="Relative Row from Seed Pixel",
                ylabel="Relative TS",
                legend=True,
                ylim=(-0.5,np.max(TS)+5),
            )  
            axs.xaxis.set_major_locator(MultipleLocator(5))
            axs.xaxis.set_major_formatter("{x:.0f}")
            axs.xaxis.set_minor_locator(MultipleLocator(1))
            axs.yaxis.set_major_locator(MultipleLocator(5))
            axs.yaxis.set_major_formatter("{x:.0f}")
            axs.yaxis.set_minor_locator(MultipleLocator(1))
            plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Row_vs_RelativeTS")

            print(
                f"Expected Charge 1: {orthCharge/np.sin(np.deg2rad(90-dataFile.angle)):.2f} ± {orthCharge_e/np.sin(np.deg2rad(90-dataFile.angle)):.2f}V"
            )
            input()
            i += 1
