from plotCluster import plotCluster, clusterPlotter
from orthClusterCharge import getOrthClusterCharge
from funcs import angle_with_error_mc,isTrack
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
#print(orthCharge, orthCharge_e)
config = configLoader.loadConfig()
# config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
    dataFile.init_cluster_voltages()
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    i = 0
    for cluster in clusters[20000:]:
        if i > 20:
            break
        path = base_path + f"Cluster_{cluster.getIndex()}/"
        Timestamps = cluster.getTSs(True)
        TS = Timestamps - np.min(Timestamps)
        # clusterPlotter(cluster,path,"Relative TS").finishPlot("Relative TS",TS)
        # clusterPlotter(cluster,path,"Voltage").finishPlot("Voltage",cluster.getHit_Voltages(True))
        # clusterPlotter(cluster,path,"Voltage Error").finishPlot("Voltage Error",cluster.getHit_VoltageErrors(True))
        # clusterPlotter(cluster,path,"ToT").finishPlot("ToT",cluster.getToTs(True))
        Rows = cluster.getRows(excludeCrossTalk=True)
        Columns = cluster.getColumns(excludeCrossTalk=True)
        if not isTrack(cluster):
            continue
        result = linregress(RowsToMicroMeter(Rows), ColumnsToMicroMeter(Columns))
        x = np.linspace(0, 372, 372)
        y = MicroMeterToColumns(line(RowsToMicroMeter(x), result.slope, result.intercept))
        CP = clusterPlotter(cluster, path, "Cluster Map")
        angle, angle_e = angleFromCharge(
            orthCharge,
            orthCharge_e,
            cluster.getClusterCharge(True),
            cluster.getClusterChargeError(True),
        )
        #CP.plot.axs.plot(
        #    x,
        #    y,
        #    color=CP.plot.colorPalette[5],
        #    label=f"Slope: {result.slope:.2f}\nAngle: {angle:.2f} ± {angle_e:.2f} Degrees\nLandau CDF: {landau.cdf(cluster.getClusterCharge(True), 13, 1):.2}\nCharge: {cluster.getClusterCharge(True):.2f} ± {cluster.getClusterChargeError(True):.2f} V",
        #)
        CP.finishPlot("Voltage", cluster.getHit_Voltages(True),textLabels=True)
        CP = clusterPlotter(cluster, path, "Relative TS")
        CP.plot.axs.plot(
            x,
            y,
            color=CP.plot.colorPalette[5],
            label=f"Slope: {result.slope:.2f}\nAngle: {angle:.2f} ± {angle_e:.2f} Degrees\nLandau CDF: {landau.cdf(cluster.getClusterCharge(True), 13, 1):.2}\nCharge: {cluster.getClusterCharge(True):.2f} ± {cluster.getClusterChargeError(True):.2f} V",
        )
        CP.finishPlot("Relative TS", TS)
        """
        Timestamps = Timestamps - applyTimeWalkCorrection(cluster.getHit_Voltages(True))
        TS = Timestamps - np.min(Timestamps)
        CP = clusterPlotter(cluster, path, "Relative TS Time walk corrected")
        CP.plot.axs.plot(
            x,
            y,
            color=CP.plot.colorPalette[5],
            label=f"Slope: {result.slope:.2f}\nAngle 1: {angle:.2f} ± {angle_e:.2f} Degrees\nLandau CDF: {landau.cdf(cluster.getClusterCharge(True), 13, 1):.2}\nCharge: {cluster.getClusterCharge(True):.2f} ± {cluster.getClusterChargeError(True):.2f} V",
        )
        CP.finishPlot("Relative TS", TS)
        """
        # print(angleFromCharge(orthCharge,orthCharge_e,cluster.getClusterCharge(True),cluster.getClusterChargeError(True)))
        # print(orthCharge,orthCharge_e,cluster.getClusterCharge(True),cluster.getClusterChargeError(True))
        plot = plotClass(path)
        axs = plot.axs
        axs.scatter(cluster.getRows(True),TS,color=plot.colorPalette[0],marker="x")
        plot.saveToPDF(f"Cluster_{indexes[i]}_Row_vs_RelativeTS")
        
        print(
            f"Expected Charge 1: {orthCharge/np.sin(np.deg2rad(90-dataFile.angle)):.2f} ± {orthCharge_e/np.sin(np.deg2rad(90-dataFile.angle)):.2f}V"
        )
        i += 1
