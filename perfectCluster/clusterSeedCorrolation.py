import sys
from funcs import (
    findConnectedSections,
    getTemplate,
    isPerfectCluster,
    loadOrCalcMPV,
    filterForTemplate,
    convertRowsForFit,
)

sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader, _memCheck
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm
from pixelCharges.plotCluster import clusterPlotter
from landau import landau


config = configLoader.loadConfig()
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/"
    dataFile.init_cluster_voltages()
    dataList = []
    dataList2 = []
    ToTList1 = []
    ToTList2 = []
    for cluster in tqdm(dataFile.get_perfectClusters(minPval=0.1,layer=4,excludeCrossTalk=True,maxRow=25), desc="Checking Correlation"):
        rows = cluster.getRows(excludeCrossTalk=True)
        ToT = cluster.getToTs(excludeCrossTalk=True)
        if cluster.flipped:
            seedRow = np.max(rows[cluster.section])
        else:
            seedRow = np.min(rows[cluster.section])
        mask = np.zeros(len(rows), dtype=bool)
        mask[cluster.section,] = True
        rowDiffs = rows - seedRow
        if np.any(rowDiffs > 100) or np.any(rowDiffs < -100):    
            path = base_path + f"Clusters/Cluster_{cluster.getIndex()}/"
            CP = clusterPlotter(cluster, path, "ToT")
            column = cluster.getColumns(excludeCrossTalk=True)[0]
            CP.plot.axs.vlines(seedRow,column-10,column+10, color=CP.plot.colorPalette[0], linestyle="--", label="Seed Pixel")
            CP.plot.axs.vlines(seedRow+248-248 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[2], linestyle="--", label="Seed Pixel + 248")
            CP.plot.axs.vlines(seedRow+185-185 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[3], linestyle="--", label="Seed Pixel + 185")
            CP.plot.axs.vlines(seedRow+80-80 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[4], linestyle="--", label="Seed Pixel + 80")
            CP.finishPlot("ToT", cluster.getToTs(True))
            
            CP = clusterPlotter(cluster, path, "ToT_WithCrossTalk")
            column = cluster.getColumns(excludeCrossTalk=False)[0]
            CP.plot.axs.vlines(seedRow,column-10,column+10, color=CP.plot.colorPalette[0], linestyle="--", label="Seed Pixel")
            CP.plot.axs.vlines(seedRow+248-248 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[2], linestyle="--", label="Seed Pixel + 248")
            CP.plot.axs.vlines(seedRow+185-185 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[3], linestyle="--", label="Seed Pixel + 185")
            CP.plot.axs.vlines(seedRow+80-80 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[4], linestyle="--", label="Seed Pixel + 80")
            CP.finishPlot("ToT", cluster.getToTs(False),excludeCrossTalk=False)

            CP = clusterPlotter(cluster, path, "Hit_Voltages_WithCrossTalk")
            column = cluster.getColumns(excludeCrossTalk=False)[0]
            CP.plot.axs.vlines(seedRow,column-10,column+10, color=CP.plot.colorPalette[0], linestyle="--", label="Seed Pixel")
            CP.plot.axs.vlines(seedRow+248-248 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[2], linestyle="--", label="Seed Pixel + 248")
            CP.plot.axs.vlines(seedRow+185-185 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[3], linestyle="--", label="Seed Pixel + 185")
            CP.plot.axs.vlines(seedRow+80-80 * 2 * np.any(rowDiffs < -100),column-10,column+10, color=CP.plot.colorPalette[4], linestyle="--", label="Seed Pixel + 80")
            CP.finishPlot("Voltage", cluster.getHit_Voltages(False),excludeCrossTalk=False)
            input()
    plot = plotClass(base_path + "SeedCorrelation/")
    axs = plot.axs
    height, x = np.histogram(dataList, bins=372*2, range=(-372,372))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    plot.set_config(
        axs,
        title="Pixel Row Correlation to Seed Pixel",
        xlabel="Row difference from seed pixel",
        ylabel="Counts",
    )
    plot.saveToPDF("Distance_From_Seed")

    plot = plotClass(base_path + "SeedCorrelation/")
    axs = plot.axs
    height, x = np.histogram(dataList2, bins=372*2, range=(-372,372))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    plot.set_config(
        axs,
        title="Pixel Row Correlation to Seed Pixel",
        xlabel="Row difference from seed pixel",
        ylabel="Counts",
    )
    plot.saveToPDF("Distance_From_Seed_NonSection")
    


    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(20,10))
    axs = plot.axs
    array, yedges, xedges = np.histogram2d(
        ToTList1,
        dataList,
        bins=(128, 372),
        range=[[0, 256], [-372, 372]],
    )
    axs.imshow(
        array, aspect="auto", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plot.set_config(
        axs,
        title="Distance From Seed Pixel vs ToT (Section Pixels)",
        xlabel="Distance From Seed Pixel",
        ylabel="ToT",
    )
    plot.saveToPDF(f"Distance_From_Seed_ToT")

    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(20,10))
    axs = plot.axs
    array, yedges, xedges = np.histogram2d(
        ToTList2,
        dataList2,
        bins=(128, 372),
        range=[[0, 256], [-372, 372]],
    )
    axs.imshow(
        array, aspect="auto", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plot.set_config(
        axs,
        title="Distance From Seed Pixel vs ToT (None Section Pixels)",
        xlabel="Distance From Seed Pixel",
        ylabel="ToT",
    )
    plot.saveToPDF(f"Distance_From_Seed_NonSection_ToT")