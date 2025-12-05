from plotCluster import plotCluster, clusterPlotter
from orthClusterCharge import getOrthClusterCharge
from funcs import (
    angle_with_error_mc,
    characterizeCluster,
    typeDict,
    isTypes,
    findValidLines,
    findBestLine,
)
import sys

sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import numpy as np
from scipy.stats import linregress
from landau import landau
from plotAnalysis import plotClass


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
    for cluster in clusters[20000:30000]:
        if i > 50:
            break
        path = base_path + f"Cluster_{cluster.getIndex()}/"
        Rows = cluster.getRows(excludeCrossTalk=True)
        Columns = cluster.getColumns(excludeCrossTalk=True)
        Timestamps = cluster.getTSs(True)
        TS = Timestamps - np.min(Timestamps)


        CP = clusterPlotter(cluster, path, "Valid Lines Check")
        lines = findValidLines(Rows,Columns)
        
        for m,c in lines:
            x = np.linspace(0, 372, 372)
            y = MicroMeterToColumns(line(RowsToMicroMeter(x), m, c))
            label = f"Slope: {m:.2f}\nIntercept: {c:.2f}"
            CP.plot.axs.plot(
                x,
                y,
                color=CP.plot.colorPalette[5],
                label=label,
            )
        CP.finishPlot("Relative TS", TS, textLabels=True)
        
        CP = clusterPlotter(cluster, path, "Best Lines Check")
        lines = findValidLines(Rows,Columns)
        if len(lines) != 0:
            m,c = findBestLine(cluster,lines)
            x = np.linspace(0, 372, 372)
            y = MicroMeterToColumns(line(RowsToMicroMeter(x), m, c))
            label = f"Slope: {m:.2f}\nIntercept: {c:.2f}"
            CP.plot.axs.plot(
                x,
                y,
                color=CP.plot.colorPalette[5],
                label=label,
            )
        CP.finishPlot("Relative TS", TS, textLabels=True)

        print(
            f"Expected Charge 1: {orthCharge/np.sin(np.deg2rad(90-dataFile.angle)):.2f} ± {orthCharge_e/np.sin(np.deg2rad(90-dataFile.angle)):.2f}V"
        )
        input()
        i += 1
