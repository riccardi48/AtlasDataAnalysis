import sys
sys.path.append("..")
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib.colors import LogNorm

def timeSectionPlots(config,path,n,dataFile,timesMod,widthToBeUsed,chargeToBeUsed,highFreq,cutIndex):
    highPeriod = 1 / highFreq
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs

    height, x = np.histogram(timesMod, bins=100)
    axs.stairs(
        height, x, baseline=None, color=plot.colorPalette[1], label=f"{highFreq*1000:.2f} Hz"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, highPeriod),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_{n}")
    height2, x2 = np.histogram(timesMod[np.invert(cutIndex)], bins=100)
    height1, x1 = np.histogram(timesMod[cutIndex], bins=100)
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    axs.stairs(
        height1, x1, baseline=None, color=plot.colorPalette[0], label=f"High"
    )
    axs.stairs(
        height2, x2, baseline=None, color=plot.colorPalette[1], label=f"Low"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, highPeriod),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_{n}_Widths")
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    axs.stairs(
        height1/np.sum(height1), x1, baseline=None, color=plot.colorPalette[0], label=f"High"
    )
    axs.stairs(
        height2/np.sum(height2), x2, baseline=None, color=plot.colorPalette[1], label=f"Low"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, highPeriod),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_{n}_Widths_norm")

    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    axs.stairs(
        np.diff(height1), x1[1:], baseline=None, color=plot.colorPalette[0], label=f"High"
    )
    axs.stairs(
        np.diff(height2), x2[1:], baseline=None, color=plot.colorPalette[1], label=f"Low"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, highPeriod),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_{n}_WidthsDiff")
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    axs.stairs(
        np.diff(height1)/np.sum(height1), x1[1:], baseline=None, color=plot.colorPalette[0], label=f"High"
    )
    axs.stairs(
        np.diff(height2)/np.sum(height2), x2[1:], baseline=None, color=plot.colorPalette[1], label=f"Low"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, highPeriod),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_{n}_WidthsDiff_norm")

    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    array, xedges, yedges = np.histogram2d(
        timesMod, widthToBeUsed, (100, 30), ((np.min(timesMod), np.max(timesMod)), (0.5, 30.5))
    )
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    array = array.transpose()
    # array = array / np.sum(array,axis=1)[:, np.newaxis]
    im = axs.imshow(
        array,
        aspect=(xedges[-1] - xedges[0]) / (yedges[-1] - yedges[0]),
        origin="lower",
        extent=extent,
        norm=LogNorm(vmin=1, vmax=np.max(array)),
    )
    plot.set_config(axs, xlabel="Times", ylabel="Widths")
    plot.saveToPDF(f"TimeGroups_high_{n}_hist2d_width")

