import sys

sys.path.append("..")

from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for i, dataFile in enumerate(dataFiles):
    plot = plotClass(config["pathToOutput"] + "TimeTests/")
    axs = plot.axs
    times4,indexes = dataFile.get_cluster_attr("Times", layer=4, excludeCrossTalk=True,returnIndexes=True)
    minTime = 135000
    maxTime = 135300
    range = (minTime, maxTime)
    bins = int(np.ptp(range) / 1)
    height, x = np.histogram(times4, bins=bins, range=range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1], label=f"{dataFile.fileName}")
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=range,
        title="Clusters Count Over Time",
        legend=False,
        xlabel="Time [ms]",
        ylabel="Frequency [hz]",
    )
    plot.saveToPDF(f"ClusterTimes_{dataFile.fileName}")
    plot = plotClass(config["pathToOutput"] + "TimeTests/")
    axs = plot.axs
    timesDiff4 = np.diff(times4)
    minTime = 135000
    maxTime = 135300
    range = (minTime, maxTime)
    axs.scatter(
        times4[1:],
        timesDiff4,
        s=2,
        marker="x",
    )
    axs.hlines(1/300*1000, minTime, maxTime, colors=plot.textColor, linestyles="dashed")
    axs.text(
            maxTime + (maxTime - minTime) * 0.01,
            1/300*1000,
            "300 Hz",
            color=plot.textColor,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )
    axs.hlines(1/30*1000, minTime, maxTime, colors=plot.textColor, linestyles="dashed")
    axs.text(
            maxTime + (maxTime - minTime) * 0.01,
            1/30*1000,
            "30 Hz",
            color=plot.textColor,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )
    axs.set_yscale("log")
    plot.set_config(
        axs,
        ylim=(0.001, 1000),
        xlim=range,
        title="Clusters Time Difference",
        legend=False,
        xlabel="Cluster Index",
        ylabel="Time Difference [ms]",
    )
    plot.saveToPDF(f"ClusterTimeDifferences_{dataFile.fileName}")

    #clusters = dataFile.get_clusters(excludeCrossTalk = True)[indexes[(times4>minTime) & (times4<maxTime)]]
    #df = dataFile.get_dataFrame()
    #print(len(clusters))
    #for cluster in clusters:
    #    print(df.iloc[cluster.indexes])