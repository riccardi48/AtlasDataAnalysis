import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
import numpy as np

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for i, dataFile in enumerate(dataFiles):
    plot = plotClass(config["pathToOutput"] + "ClusterCharges/")
    axs = plot.axs
    dataFile.init_cluster_voltages()
    clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=config["layers"][0])])
    clusterCharges = clusterCharges[clusterCharges > 0]
    height, x = np.histogram(clusterCharges, bins=150, range=(0, 60))
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