import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
import numpy as np

config = configLoader.loadConfig()
#config["filterDict"] = {"telescope":"kit","fileName":"angle6_4Gev_kit_2"}
config["filterDict"] = {"telescope":"kit"}
dataFiles = initDataFiles(config)

def linearLine(x,m,c):
    return m*x + c

def getPixelResponseSlope(rows,chargeCollected):
    rows = rows[chargeCollected>0]
    chargeCollected = chargeCollected[chargeCollected>0]
    if len(rows) <= 1 or np.unique(rows).size <= 1:
        return np.nan
    relativeRows = rows-np.min(rows)
    popt,pcov = curve_fit(linearLine,relativeRows,chargeCollected)
    return popt[0]

for dataFile in dataFiles:
    dataFile.init_cluster_voltages()
    for _range in ((-5,5),(-20,20)):
        plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/")
        axs = plot.axs
        PRS = np.array([getPixelResponseSlope(cluster.getRows(excludeCrossTalk=True),cluster.getToTs(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)])
        PRS = PRS[~np.isnan(PRS)]
        height, x = np.histogram(PRS, bins=210, range=_range)
        axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
        plot.set_config(
            axs,
            ylim=(0, None),
            xlim=_range,
            title=f"Pixel Response Slope Histogram {dataFile.fileName} {_range}",
            xlabel="Pixel Response Slope",
            ylabel="Frequency",
            )
        plot.saveToPDF(f"PixelResponseSlope_{dataFile.fileName}_{_range}")

"""
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/ScatterTest/")
    axs = plot.axs
    PRS = np.array([getPixelResponseSlope(cluster.getRows(excludeCrossTalk=True),cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)])
    Columns = np.array([np.mean(cluster.getColumns(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)])
    Columns = Columns[~np.isnan(PRS)]
    PRS = PRS[~np.isnan(PRS)]
    _range = (-0.1,0.1)
    axs.hist2d(Columns,PRS,bins=(135,210),range=((0,135),_range),cmin=1)
    plot.set_config(
        axs,
        ylim=_range,
        xlim=(0, None),
        title=f"Pixel Response Slope Histogram {dataFile.fileName}",
        xlabel="Column",
        ylabel="Pixel Response Slope",
        )
    plot.saveToPDF(f"PixelResponseSlope_Scatter_{dataFile.fileName}")
 """