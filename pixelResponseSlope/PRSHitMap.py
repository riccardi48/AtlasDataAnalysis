from PRSFuncs import loadOrCalcPRS,sumOfCauchyFunc,cauchyFunc,plotHist,plotHistRemoved
import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import numpy.typing as npt
from dataAnalysis._fileReader import calcDataFileManager
from landau import landau
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

config = configLoader.loadConfig()
#config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    dataFile.init_cluster_voltages()
    clusters,indexes = dataFile.get_clusters(excludeCrossTalk=True,returnIndexes=True,layer=4)
    PRS = loadOrCalcPRS(dataFile,config)
    indexes = indexes[dataFile.get_cluster_attr("Sizes",excludeCrossTalk=True,layer=4)[0]>0]
    clusters = clusters[dataFile.get_cluster_attr("Sizes",excludeCrossTalk=True,layer=4)[0]>0]
    PRS = PRS[indexes]
    array = np.zeros((132,372))
    print(len(PRS),len(clusters))
    for i,cluster in enumerate(clusters):
        if PRS[i] > 0.015 and PRS[i] < 0.050:
            pixels = cluster.getRows(True)<np.max(cluster.getRows(True))-10
            rows = cluster.getRows(True)[pixels]
            columns = cluster.getColumns(True)[pixels]
            array[columns,rows] += 1
    
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/HeatMap/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(array, aspect=3,origin="lower")
    plot.set_config(
        axs,
        title=f"PRS Heatmap {dataFile.fileName}\nmax:{np.max(array)}",
        xlabel="Rows",
        ylabel="Columns",
        )
    plot.saveToPDF(f"{dataFile.fileName}_HeatMap")
    arrayAll = np.zeros((132,372))
    for i,cluster in enumerate(clusters):
        rows = cluster.getRows(True)
        columns = cluster.getColumns(True)
        arrayAll[columns,rows] += 1
    
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/HeatMap/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(arrayAll, aspect=3,origin="lower")
    plot.set_config(
        axs,
        title=f"PRS Heatmap {dataFile.fileName}\nmax:{np.max(arrayAll)}",
        xlabel="Rows",
        ylabel="Columns",
        )
    plot.saveToPDF(f"{dataFile.fileName}_HeatMap_Unchanged")

    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/HeatMap/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(array/arrayAll, aspect=3,origin="lower")
    plot.set_config(
        axs,
        title=f"PRS Heatmap {dataFile.fileName}\nmax:{np.max(array/arrayAll)}",
        xlabel="Rows",
        ylabel="Columns",
        )
    plot.saveToPDF(f"{dataFile.fileName}_HeatMap_Ratio")


