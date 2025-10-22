import sys

sys.path.append("..")

from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np
from matplotlib.ticker import MultipleLocator

config = configLoader.loadConfig()
config["filterDict"]["voltage"] = 48
config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}

dataFiles = initDataFiles(config)

plot1 = plotClass(config["pathToOutput"] + "widths/")
axs1 = plot1.axs
plot2 = plotClass(config["pathToOutput"] + "widths/")
axs2 = plot2.axs
range1 = (-0.5, 50.5)
range2 = (-0.5, 20.5)
bins1 = int(np.ptp(range1)-1)
bins2 = int(np.ptp(range2)-1)
minTime = 400000
maxTime = 600000
for i, dataFile in enumerate(dataFiles):
    columnWidths, _ = dataFile.get_cluster_attr(
        "ColumnWidths", layer=4, excludeCrossTalk=True
    )

    rowsWidths, _ = dataFile.get_cluster_attr(
        "RowWidths", layer=4, excludeCrossTalk=True
    )

    times, _ = dataFile.get_cluster_attr(
        "Times", layer=4, excludeCrossTalk=True
    )
    rowsWidths = rowsWidths[(times > minTime) & (times < maxTime)]
    height, x = np.histogram(rowsWidths, bins=bins1, range=range1)
    axs1.stairs(
        height,
        x,
        baseline=None,
        color=plot1.colorPalette[i],
        label=f"{dataFile.fileName}",
    )
    columnWidths = columnWidths[(times > minTime) & (times < maxTime)]
    height, x = np.histogram(columnWidths, bins=bins2, range=range2)
    axs2.stairs(
        height,
        x,
        baseline=None,
        color=plot2.colorPalette[i],
        label=f"{dataFile.fileName}",
    )
plot1.set_config(
    axs1,
    ylim=(0, None),
    xlim=range1,
    title="Row Width Distribution",
    legend=True,
    xlabel="Row Width [px]",
    ylabel="Frequency",
)
axs1.xaxis.set_major_locator(MultipleLocator(5))
axs1.xaxis.set_major_formatter("{x:.0f}")
axs1.xaxis.set_minor_locator(MultipleLocator(1))
axs1.yaxis.set_major_locator(MultipleLocator(5000))
axs1.yaxis.set_major_formatter("{x:.0f}")
axs1.yaxis.set_minor_locator(MultipleLocator(1000))
plot1.saveToPDF(f"Comparison_RowWidthDistribution")

plot2.set_config(
    axs2,
    ylim=(0, None),
    xlim=range2,
    title="Column Width Distribution",
    legend=True,
    xlabel="Column Width [px]",
    ylabel="Frequency",
)
axs2.xaxis.set_major_locator(MultipleLocator(5))
axs2.xaxis.set_major_formatter("{x:.0f}")
axs2.xaxis.set_minor_locator(MultipleLocator(1))
axs2.yaxis.set_major_locator(MultipleLocator(5000))
axs2.yaxis.set_major_formatter("{x:.0f}")
axs2.yaxis.set_minor_locator(MultipleLocator(1000))
plot2.saveToPDF(f"Comparison_ColumnWidthDistribution")
