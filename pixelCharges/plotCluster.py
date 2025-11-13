import sys
sys.path.append("..")
from plotAnalysis import plotClass
import numpy as np
from dataAnalysis import clusterClass
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class displayClass():
    def __init__(self, x, y, buffer: int = 5, cmap: str = "plasma"):
        self.buffer = buffer
        self.cmap = cmap
        x_range = int(np.max(x) - np.min(x))
        y_range = int(np.max(y) - np.min(y))
        self.display = np.zeros((y_range + buffer, x_range + buffer))
        self.display[self.display == 0] = np.nan
        self.extent = np.array(
            [
                np.min(x) - buffer / 2,
                np.max(x) + buffer / 2,
                np.min(y) - buffer / 2,
                np.max(y) + buffer / 2,
            ]
            )
        self.xlim = (self.extent[0], self.extent[1])
        self.ylim = (self.extent[2], self.extent[3])
    def addToDisplay(self,x,y,value):
        for i in range(len(x)):
            self.display[
                y[i] - np.min(y) + int((self.buffer - 1) / 2),
                x[i] - np.min(x) + int((self.buffer - 1) / 2),
            ] = value[i]
    def showDisplay(self,ax,):
        im = ax.imshow(
            self.display, cmap=self.cmap, extent=self.extent, aspect=3, origin="lower", vmin = 0
        )
        return im 
def plotCluster(plot: plotClass,cluster: clusterClass, values,name,colorbarName,textLabels=False):
    x = cluster.getRows(excludeCrossTalk=True)
    y = cluster.getColumns(excludeCrossTalk=True)
    display = displayClass(x,y)
    display.addToDisplay(x,y,values)
    im = display.showDisplay(plot.axs)
    #divider = make_axes_locatable(axs)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    #cbar.set_label(name, rotation=270, labelpad=15)
    plt.colorbar(im, ax=plot.axs, label=colorbarName)
    plot.set_config(
        plot.axs,
        title=f"Cluster {name} Distribution",
        xlabel="Pixel Row",
        ylabel="Pixel Column",
        xlim=display.xlim,
        ylim=display.ylim,
        #legend=True,
    )
    if np.ptp(x) > 20:
        plot.axs.xaxis.set_major_locator(MultipleLocator(5))
        plot.axs.xaxis.set_major_formatter("{x:.0f}")
        plot.axs.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        plot.axs.xaxis.set_major_locator(MultipleLocator(1))
        plot.axs.xaxis.set_major_formatter("{x:.0f}")
        plot.axs.xaxis.set_minor_locator(MultipleLocator(1))
    plot.axs.yaxis.set_major_locator(MultipleLocator(1))
    plot.axs.yaxis.set_major_formatter("{x:.0f}")
    plot.axs.yaxis.set_minor_locator(MultipleLocator(1))
    if textLabels:
        for i in range(len(x)):
            plot.axs.text(
                x[i],
                y[i]-0.5-i%2*0.15,
                f"{values[i]:.2f}",
                color=plot.colorPalette[2],
                fontsize=6,
                ha="center",
                va="top",
            )


class clusterPlotter():
    def __init__(self,cluster: clusterClass,path,name):
        self.cluster = cluster
        self.plot = plotClass(path,sizePerPlot=(10,5))
        self.name = name
    def finishPlot(self,colorbarName, values, textLabels=False):
        plotCluster(self.plot,self.cluster, values,self.name,colorbarName, textLabels=textLabels)
        self.plot.saveToPDF(f"{self.cluster.getIndex()}_{self.name.replace(' ','_')}")
        