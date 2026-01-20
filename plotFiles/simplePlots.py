###############################
# Plots in this file:
# 1. Histogram of Row frequency with comparison when crosstalk removed
# 2. Same as above with Columns
# 3. Same as above with ToT
###############################
import sys
from plotClass import plotGenerator
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader

def attributeHistogram(
    ax, values, bins=128, _range=(0, 256), xlabel="", xticks=None, yticks=None, color=None
):
    height, x = np.histogram(values, bins=bins, range=_range)
    ax.stairs(height, x, color=color, baseline=None)
    ax.set_config(
        ax,
        xlabel=xlabel,
        ylabel="Counts",
        legend=False,
        xticks=xticks,
        yticks=yticks,
    )

def runSimple(dataFiles,plotGen,config):
    layer = config["layers"][0]
    for dataFile in dataFiles:
        path = f"Simple_Plots/{dataFile.fileName}/"
        rowPlot = plotGen.newPlot(path)
        height, x = np.histogram(
            dataFile.get_base_attr("Row", layers = config["layers"]), bins=371, range=(0.5, 371.5)
        )
        rowPlot.axs.stairs(height, x, color=rowPlot.colorPalette[0], baseline=None, label="Raw")
        height, x = np.histogram(
            dataFile.get_base_attr("Row", layers = config["layers"], excludeCrossTalk=True),
            bins=371,
            range=(0.5, 371.5),
        )
        rowPlot.axs.stairs(
            height, x, color=rowPlot.colorPalette[1], baseline=None, label="CrossTalk Removed"
        )
        rowPlot.set_config(
            rowPlot.axs,
            title=f"Rows on Layer {layer}",
            xlabel="Rows",
            ylabel="Counts",
            legend=True,
            xticks=[50, 10],
            yticks=[2000, 200],
            ylim=(0, None),
            xlim=(1, 371),
        )
        rowPlot.saveToPDF("Rows")

        columnPlot = plotGen.newPlot(path)
        height, x = np.histogram(
            dataFile.get_base_attr("Column", layers = config["layers"]), bins=131, range=(0.5, 131.5)
        )
        columnPlot.axs.stairs(height, x, color=columnPlot.colorPalette[0], baseline=None, label="Raw")
        height, x = np.histogram(
            dataFile.get_base_attr("Column", layers = config["layers"], excludeCrossTalk=True),
            bins=131,
            range=(0.5, 131.5),
        )
        columnPlot.axs.stairs(
            height, x, color=columnPlot.colorPalette[1], baseline=None, label="CrossTalk Removed"
        )
        columnPlot.set_config(
            columnPlot.axs,
            title=f"Columns on Layer {layer}",
            xlabel="Columns",
            ylabel="Counts",
            legend=True,
            xticks=[20, 5],
            yticks=[5000, 1000],
            ylim=(0, None),
            xlim=(1, 131),
        )
        columnPlot.saveToPDF("Columns")

        ToTPlot = plotGen.newPlot(path)
        height, x = np.histogram(
            dataFile.get_base_attr("ToT", layers = config["layers"]), bins=256, range=(-0.5, 255.5)
        )
        ToTPlot.axs.stairs(height, x, color=ToTPlot.colorPalette[0], baseline=None, label="Raw")
        height, x = np.histogram(
            dataFile.get_base_attr("ToT", layers = config["layers"], excludeCrossTalk=True),
            bins=256,
            range=(-0.5, 255.5),
        )
        ToTPlot.axs.stairs(
            height, x, color=ToTPlot.colorPalette[1], baseline=None, label="CrossTalk Removed"
        )
        ToTPlot.set_config(
            ToTPlot.axs,
            title=f"ToT on Layer {layer}",
            xlabel="ToT [TS]",
            ylabel="Counts",
            legend=True,
            xticks=[32, 8],
            yticks=[50000, 10000],
            ylim=(0, None),
            xlim=(0, 255),
        )
        ToTPlot.saveToPDF("ToT")

if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runSimple(dataFiles,plotGen,config)

