from PRSFuncs import loadOrCalcPRS,sumOfCauchyFunc,cauchyFunc,plotHist,plotHistRemoved
import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
from dataAnalysis._fileReader import calcDataFileManager

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    _range = (-0.2005,0.2005)
    bins = 401
    xlim = (-0.1,0.1)
    PRS = loadOrCalcPRS(dataFile,config)
    clusters,indexes = dataFile.get_clusters(excludeCrossTalk=True,returnIndexes=True,layer=4)
    filter = (dataFile.get_cluster_attr("Sizes",excludeCrossTalk=True,layer=4)[0]>0)&(dataFile.get_cluster_attr("ColumnWidths",excludeCrossTalk=True,layer=4)[0]<=2)
    indexes = indexes[filter]
    clusters = clusters[filter]
    minRows = np.array([np.min(cluster.getRows(True)) for cluster in clusters])
    maxRows = np.array([np.max(cluster.getRows(True)) for cluster in clusters])
    height, x = np.histogram(PRS[indexes], bins=bins, range=_range)
    plotHist(dataFile,config["pathToOutput"] + "PixelResponseSlope/",height,x,_range,xlim)
    binCentres = (x[:-1] + x[1:]) / 2
    popt, pcov = curve_fit(
            sumOfCauchyFunc,
            binCentres[binCentres<0],
            height[binCentres<0],
            maxfev=10000,
       )
    height = height - sumOfCauchyFunc(binCentres, *popt)
    height[binCentres==0] = 0
    plotHistRemoved(dataFile,config["pathToOutput"] + "PixelResponseSlope/removed/",height,x,xlim)
