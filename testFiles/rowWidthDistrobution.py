import sys

sys.path.append("..")

from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np
from matplotlib.ticker import MultipleLocator

config = configLoader.loadConfig()
config["filterDict"]["voltage"] = 48
dataFiles = initDataFiles(config)

plot = plotClass(config["pathToOutput"] + "rowWidths/")
axs = plot.axs
bins = 11
range = (-0.5, 10.5)
minTime = 400000
maxTime = 600000
for i, dataFile in enumerate(dataFiles):
    rowsWidths, _ = dataFile.get_cluster_attr(
        "ColumnWidths", layer=4, excludeCrossTalk=True
    )
    times, _ = dataFile.get_cluster_attr(
        "Times", layer=4, excludeCrossTalk=True
    )
    rowsWidths = rowsWidths[(times > minTime) & (times < maxTime)]
    height, x = np.histogram(rowsWidths, bins=bins, range=range)
    axs.stairs(
        height,
        x,
        baseline=None,
        color=plot.colorPalette[i],
        label=f"{dataFile.fileName}",
    )
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=range,
    title="Column Width Distribution",
    legend=True,
    xlabel="Column Width [px]",
    ylabel="Frequency",
)
axs.xaxis.set_major_locator(MultipleLocator(5))
axs.xaxis.set_major_formatter("{x:.0f}")
axs.xaxis.set_minor_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(5000))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(1000))
plot.saveToPDF(f"Comparison_ColumnWidthDistribution")
