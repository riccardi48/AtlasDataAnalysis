###############################
# Plots in this file:
# 1. Histogram of Cluster Charge, with comparison to perfect clusters
# 2. Same as above with smaller y axis
###############################
import sys
from plotClass import plotGenerator
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from dataAnalysis.handlers._genericClusterFuncs import isFlat

def runCharge(dataFiles,plotGen,config):

    for dataFile in dataFiles:
        dataFile.init_cluster_voltages()
        layer = 4
        path = f"PerfectClusters/{dataFile.fileName}/"
        clusters = dataFile.get_clusters(layer=4,excludeCrossTalk = True)
        clusterCharges = [cluster.getClusterCharge(True) for cluster in clusters if isFlat(cluster)]
        clusters = dataFile.get_perfectClusters(layer=4,excludeCrossTalk = True)
        clusterChargesPerfect = [cluster.getClusterCharge(True) for cluster in clusters]
        plot = plotGen.newPlot(path)
        height, x = np.histogram(clusterCharges, bins=200, range=(0, 30))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[1], baseline=None, label="All Cluster Charge"
        )
        height, x = np.histogram(clusterChargesPerfect, bins=200, range=(0, 30))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[0], baseline=None, label="Perfect Cluster Charge"
        )
        plot.set_config(plot.axs,
            title="Cluster Charge",
            xlabel="Voltage [V]",
            ylabel="Count",
            ylim = (0,None),
            xlim = (0,None),
            xticks=[2,1],
            legend=True,
            )
        plot.saveToPDF("ClusterCharge") 
        plot = plotGen.newPlot(path)
        height, x = np.histogram(clusterCharges, bins=200, range=(0, 30))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[1], baseline=None, label="All Cluster Charge"
        )
        height, x = np.histogram(clusterChargesPerfect, bins=200, range=(0, 30))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[0], baseline=None, label="Perfect Cluster Charge"
        )
        plot.set_config(plot.axs,
            title="Cluster Charge",
            xlabel="Voltage [V]",
            ylabel="Count",
            ylim = (0,np.max(height)*4),
            xlim = (0,None),
            xticks=[2,1],
            legend=True,
            )
        plot.saveToPDF("ClusterCharge_Perfect") 



        clusters = dataFile.get_clusters(layer=4,excludeCrossTalk = True)
        RowWidths = [cluster.getRowWidth(True) for cluster in clusters if isFlat(cluster)]
        clusters = dataFile.get_perfectClusters(layer=4,excludeCrossTalk = True)
        RowWidthsPerfect = [cluster.getRowWidth(True) for cluster in clusters]
        plot = plotGen.newPlot(path)
        height, x = np.histogram(RowWidths, bins=50, range=(0.5, 50.5))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[1], baseline=None, label="All Clusters Row Width"
        )
        height, x = np.histogram(RowWidthsPerfect, bins=50, range=(0.5, 50.5))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[0], baseline=None, label="Perfect Clusters Row Width"
        )
        plot.set_config(plot.axs,
            title="Row Width",
            xlabel="Row Width",
            ylabel="Count",
            ylim = (0,None),
            xlim = (0,None),
            xticks=[2,1],
            legend=True,
            )
        plot.saveToPDF("RowWidth") 
        plot = plotGen.newPlot(path)
        height, x = np.histogram(RowWidths, bins=50, range=(0.5, 50.5))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[1], baseline=None, label="All Clusters Row Width"
        )
        height, x = np.histogram(RowWidthsPerfect, bins=50, range=(0.5, 50.5))
        plot.axs.stairs(
            height, x, color=plot.colorPalette[0], baseline=None, label="Perfect Clusters Row Width"
        )
        plot.set_config(plot.axs,
            title="Row Width",
            xlabel="Row Width",
            ylabel="Count",
            xlim = (0,None),
            ylim = (0,np.max(height)*4),
            xticks=[5,1],
            legend=True,
            )
        plot.saveToPDF("RowWidth_Perfect") 

if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runCharge(dataFiles,plotGen,config)
