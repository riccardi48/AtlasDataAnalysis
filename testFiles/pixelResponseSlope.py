import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
import numpy as np

config = configLoader.loadConfig()
#config["filterDict"] = {"telescope":"kit","fileName":"angle6_4Gev_kit_2"}
dataFiles = initDataFiles(config)

def linearLine(x,m,c):
    return m*x + c

def getPixelResponseSlope(rows,chargeCollected):
    rows = rows[chargeCollected>0]
    chargeCollected = chargeCollected[chargeCollected>0]
    if len(rows) <= 15 or np.unique(rows).size <= 10:
        return np.nan
    relativeRows = rows-np.min(rows)
    popt,pcov = curve_fit(linearLine,relativeRows,chargeCollected)
    return popt[0]

for dataFile in dataFiles:
    for _range in ((-1,1),(-0.1,0.1),(-0.05,0.05),(-0.01,0.01),(-0.002,0.002)):
        plot = plotClass(config["pathToOutput"] + "PixelResponseSLope/")
        axs = plot.axs
        dataFile.init_cluster_voltages()
        PRS = np.array([getPixelResponseSlope(cluster.getRows(excludeCrossTalk=True),cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)])
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
        plot.saveToPDF(f"PixelResponseSlope_Long_{dataFile.fileName}_{_range}")
