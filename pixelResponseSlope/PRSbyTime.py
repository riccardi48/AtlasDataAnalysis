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
config["filterDict"] = {"telescope":"kit","fileName":["4Gev_kit_1"]}
dataFiles = initDataFiles(config)


config = configLoader.loadConfig()
#config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    dataFile.init_cluster_voltages()
    clusters,indexes = dataFile.get_clusters(excludeCrossTalk=True,returnIndexes=True,layer=4)
    _range = (-0.10125,0.10125)
    bins = 201
    PRS = loadOrCalcPRS(dataFile,config)
    filter = (dataFile.get_cluster_attr("Sizes",excludeCrossTalk=True,layer=4)[0]>0)&(dataFile.get_cluster_attr("RowWidths",excludeCrossTalk=True,layer=4)[0]>15)
    indexes = indexes[filter]
    clusters = clusters[filter]
    clusterTimes = dataFile.get_cluster_attr("Times",excludeCrossTalk=True)[0][indexes]
    PRS = PRS[indexes]
    
    array, yedges, xedges = np.histogram2d(clusterTimes,PRS,bins=(300,81),range=((0,300000),_range))
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/PRSvsTimes/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    plot.set_config(
        axs,
        title=f"PRS vs Cluster Row Width {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Cluster Row Width",
        )
    plot.saveToPDF(f"{dataFile.fileName}_PRS_Times_Scatter")
