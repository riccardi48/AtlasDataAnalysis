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
    _range = (-0.40125,0.40125)
    bins = 201
    xlim = (-0.1,0.1)
    PRS = loadOrCalcPRS(dataFile,config)
    filter = (dataFile.get_cluster_attr("Sizes",excludeCrossTalk=True,layer=4)[0]>0)&(dataFile.get_cluster_attr("ColumnWidths",excludeCrossTalk=True,layer=4)[0]<=2)
    indexes = indexes[filter]
    clusters = clusters[filter]
    clusterRowWidths = np.array([cluster.getRowWidth(excludeCrossTalk=True) for cluster in clusters])
    PRS = PRS[indexes]
    
    array, yedges, xedges = np.histogram2d(clusterRowWidths,PRS,bins=(60,321),range=((0.5,60.5),_range))
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/PRSvsRowWidth/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    plot.set_config(
        axs,
        title=f"PRS vs Cluster Row Width {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Cluster Row Width",
        )
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Frequency", rotation=270, labelpad=15)
    plot.saveToPDF(f"{dataFile.fileName}_PRS_RowWidth_Scatter")
    blank = np.zeros(array.shape,dtype=float)
    for i,slice in enumerate(array):
        binCentres = (xedges[:-1] + xedges[1:]) / 2
        popt, pcov = curve_fit(
            cauchyFunc,
            binCentres[binCentres<=0],
            slice[binCentres<=0],
            maxfev=10000,
       )
        slice = slice - cauchyFunc(binCentres, *popt)
        slice[binCentres==0] = 0
        array[i] = slice
        blank[i] = cauchyFunc(binCentres, *popt)
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/PRSvsRowWidth/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],vmin=0)
    plot.set_config(
        axs,
        title=f"PRS vs Cluster Row Width {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Cluster Row Width",
        )
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Frequency", rotation=270, labelpad=15)
    plot.saveToPDF(f"{dataFile.fileName}_PRS_RowWidth_Scatter_CauchyRemoved")
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/PRSvsRowWidth/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(blank,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    plot.set_config(
        axs,
        title=f"PRS vs Cluster Row Width {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Cluster Row Width",
        )
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Frequency", rotation=270, labelpad=15)
    plot.saveToPDF(f"{dataFile.fileName}_PRS_RowWidth_Scatter_Cauchy")

    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/PRSvsRowWidth/")
    axs = plot.axs
    axs.stairs(np.sum(array[:,binCentres>0],axis=1), yedges, baseline=None, color=plot.colorPalette[0])
    plot.set_config(
        axs,
        title=f"PRS vs Cluster Row Width {dataFile.fileName}",
        xlabel="Cluster Row Width",
        ylabel="Count",
        ylim=(0,np.max(np.sum(array[:,binCentres>0],axis=1))*1.05),
        )
    plot.saveToPDF(f"{dataFile.fileName}_ClusterRowWidth_Removed")

    clusterColumnWidths = np.array([cluster.getColumnWidth(excludeCrossTalk=True) for cluster in clusters])
    array, yedges, xedges = np.histogram2d(clusterColumnWidths,PRS,bins=(11,41),range=((-0.5,10.5),(-0.1025,0.1025)))
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/PRSvsRowWidth/",sizePerPlot=(12,12))
    axs = plot.axs
    im = axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    plot.set_config(
        axs,
        title=f"PRS vs Cluster Column Width {dataFile.fileName}",
        xlabel="PRS",
        ylabel="Cluster Column Width",
        )
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Frequency", rotation=270, labelpad=15)
    plot.saveToPDF(f"{dataFile.fileName}_PRS_ColumnWidth_Scatter")

