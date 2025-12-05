from plotCluster import plotCluster, clusterPlotter
from orthClusterCharge import getOrthClusterCharge
from funcs import angle_with_error_mc
import sys
sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import numpy as np
from scipy.stats import linregress
from landau import landau
from matplotlib.ticker import MultipleLocator
from plotAnalysis import plotClass

def isTrack(cluster):
    if np.unique(cluster.getColumns(True)).size > 1:
        return False
    landau_value = landau.cdf(cluster.getClusterCharge(True), 13, 1)
    if landau_value < 0.01 or landau_value > 0.99:
        return False
    return True

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

orthCharge, orthCharge_e = getOrthClusterCharge(layer=4)
orthCharge2, orthCharge_e2 = 1.87, 0.0039
#print(orthCharge, orthCharge_e)
config = configLoader.loadConfig()
# config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/"
    dataFile.init_cluster_voltages()
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    i = 0
    for cluster in clusters[10000:]:
        if i > 20:
            break
        if not isTrack(cluster):
            continue
        path = base_path + f"Cluster_{cluster.getIndex()}/"
        Timestamps = cluster.getTSs(True)
        TS = Timestamps - np.min(Timestamps)
        Rows = cluster.getRows(excludeCrossTalk=True)
        Columns = cluster.getColumns(excludeCrossTalk=True)
        result = linregress(RowsToMicroMeter(Rows), ColumnsToMicroMeter(Columns))
        x = np.linspace(0, 372, 372)
        y = MicroMeterToColumns(line(RowsToMicroMeter(x), result.slope, result.intercept))
        CP = clusterPlotter(cluster, path, "Track Fit")
        angle, angle_e = angle_with_error_mc(
            orthCharge,
            orthCharge_e,
            cluster.getClusterCharge(True),
            cluster.getClusterChargeError(True),
        )
        CP.plot.axs.plot(
            x,
            y,
            color=CP.plot.colorPalette[5],
            label=f"Slope: {result.slope:.2f}\nAngle: {angle:.2f} ± {angle_e:.2f} Degrees\nLandau CDF: {landau.cdf(cluster.getClusterCharge(True), 13, 1):.2}\nCharge: {cluster.getClusterCharge(True):.2f} ± {cluster.getClusterChargeError(True):.2f} V",
        )
        CP.finishPlot("Voltage", cluster.getHit_Voltages(True),textLabels=True)
        CP = clusterPlotter(cluster, path, "Relative TS")
        CP.plot.axs.plot(
            x,
            y,
            color=CP.plot.colorPalette[5],
            label=f"Slope: {result.slope:.2f}\nAngle: {angle:.2f} ± {angle_e:.2f} Degrees\nLandau CDF: {landau.cdf(cluster.getClusterCharge(True), 13, 1):.2}\nCharge: {cluster.getClusterCharge(True):.2f} ± {cluster.getClusterChargeError(True):.2f} V",
        )
        CP.finishPlot("Relative TS", TS)
        i += 1
    print(
        f"Expected Charge 1: {orthCharge/np.sin(np.deg2rad(90-dataFile.angle)):.2f} ± {orthCharge_e/np.sin(np.deg2rad(90-dataFile.angle)):.2f}V"
    )
