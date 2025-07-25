from plotAnalysis import plotClass, correlationPlotter
from dataAnalysis import dataAnalysis, crossTalkFinder, filterDataFiles
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
import numpy as np
import configLoader

def RowRowCorrelation(dataFile : dataAnalysis,pathToOutput,pathToCalcData ,layers:list[int] = [1,2,3,4],excludeCrossTalk=True,recalc:bool=False,log=True):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/",sizePerPlot=(8, 8))
    axs = plot.axs
    
    rowRowPlotter = correlationPlotter(pathToCalcData,layers=layers,excludeCrossTalk=excludeCrossTalk)
    RowRow = rowRowPlotter.RowRowCorrelation(dataFile,recalc=recalc)
    extent = (
        0.5,
        371.5,
        0.5,
        371.5,
    )
    norm = None
    if log:
        norm = LogNorm(vmin=1, vmax=np.max(RowRow, where=~np.isnan(RowRow), initial=-1))
    im = axs.imshow(RowRow, origin="lower", aspect="equal", extent=extent, norm=norm)
    plot.set_config(axs, title="RowRow correlation", xlabel="Row [px]", ylabel="Row [px]")
    axs.xaxis.set_major_locator(MultipleLocator(30))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(10))
    axs.yaxis.set_major_locator(MultipleLocator(30))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(10))
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Frequency", rotation=270, labelpad=15)
    tempCrossTalkFinder = crossTalkFinder()
    x = []
    y = []
    for _x,_y in tempCrossTalkFinder.crossTalkFunction(None,returnDict=True).items():
        for i,j in _y:
            x.append(i)
            y.append(j)

    axs.scatter(x, y, c="r", s=1)
    plot.saveToOutput(f"RowRowCorrelation{"_cut" if excludeCrossTalk else ""}{"_"+"".join(str(x) for x in layers) if layers is not None else ""}")

config = configLoader.loadConfig()

files = glob(f"{config["pathToData"]}{config["fileFormate"]}")
allDataFiles = [dataAnalysis(pathToDataFile, config["pathToCalcData"], maxLine=config["maxLine"]) for pathToDataFile in files]
dataFiles = filterDataFiles(
    allDataFiles,
    filterDict=config["filterDict"],
)
for dataFile in dataFiles:
    #dataFile.get_base_attr("Row",excludeCrossTalk = True)
    RowRowCorrelation(dataFile,config["pathToOutput"],config["pathToCalcData"],layers=[4] ,excludeCrossTalk=False,recalc=False,log=True)
    RowRowCorrelation(dataFile,config["pathToOutput"],config["pathToCalcData"],layers=[4] ,excludeCrossTalk=True,recalc=False,log=True)
    clusters = dataFile.get_clusters(layers=[4])
    for cluster in clusters:
        print(dataFile.get_dataFrame().iloc[cluster.getIndexes()])
        print(cluster.crossTalk)
        print(cluster.getToTs())
        print(cluster.getIndexes())
        input("Press enter to continue...")