###############################
# Plots in this file:
# 1. Scatter Plot of MPV vs Relative Row
# 2. Voltages
###############################

import sys
from functions.mpvFuncs import mpvData,histogramHit_Voltage_Errors,landauFunc,landauBinned
from plotClass import plotGenerator
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from matplotlib.ticker import MultipleLocator

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),
                frameon=False,)


def plotVoltagePlots(data,path,plotGen,rangeOfRows = (0,30),xlim=(0,2)):
    rangeToPlot = np.arange(rangeOfRows[0],rangeOfRows[1]+1)
    rangeToPlot = rangeToPlot[np.isin(rangeToPlot,list(data.fittings.keys()))]
    plot = plotGen.newPlot(
        path,
        shape=(1, rangeToPlot.size),
        sharex=True,
        sizePerPlot=(5,2),
        hspace=0,
    )
    axs = plot.axs
    for i,j in enumerate(rangeToPlot):
        values = np.array(data.histLists[j])
        valuesErrors = np.array(data.histErrorsLists[j])
        hist, binEdges, binCentres = histogramHit_Voltage_Errors(
            values, valuesErrors, _range=data._range
        )
        axs[i].stairs(
            hist / (binEdges[1:] - binEdges[:-1]),
            binEdges,
            color=plot.colorPalette[3],
            baseline=None,
            label=f"Row {j}",  # Landau Area: {landau.cdf(_range[1], x_mpv, xi) * scale:.0f}",
        )
        x = np.linspace(0, np.max(binEdges) + 0.1, 1000)
        x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = data.fittings[j]
        y = landauFunc(x, x_mpv, xi, scale)
        axs[i].plot(
            x,
            y,
            c=plot.colorPalette[0],
            label=f"{"Mpv": <7}: {x_mpv:.3f} ± {x_mpv_e:.3f}\n{"Width": <7}: {xi:.3f} ± {xi_e:.3f}\n{"Scale": <7}: {scale:.3f} ± {scale_e:.3f}",
        )
        axs[i].errorbar(
            x[np.argmax(y)],
            y[np.argmax(y)],
            xerr=[x_mpv_e],
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
        if not np.isnan(data.constrainedFittings[j][0]):
            x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = data.constrainedFittings[j]
            y = landauFunc(x, x_mpv, xi, scale)
            axs[i].plot(
                x,
                y,
                c=plot.colorPalette[2],
                label=f"{"Mpv": <7}: {x_mpv:.3f} ± {x_mpv_e:.3f}\n{"Width": <7}: {xi:.3f} ± {xi_e:.3f}\n{"Scale": <7}: {scale:.3f} ± {scale_e:.3f}",
            )
            axs[i].errorbar(
                x[np.argmax(y)],
                y[np.argmax(y)],
                xerr=[x_mpv_e],
                fmt="none",
                color=plot.colorPalette[2],
                elinewidth=1,
                capsize=3,
            )
        axs[i].get_xaxis().set_visible(False)
        plot.set_config(
            axs[i],
            ylim=(0, np.max(y) * 1.1),
            xlim=xlim,
            legend=True,
        )
        axs[i].vlines(
                    0.161,
                    0,
                    axs[i].get_ylim()[1],
                    linestyle="--",
                    color=plot.colorPalette[5],
                    label="Threshold",
                    )
        axs[i].xaxis.set_major_formatter("{x:.1e}")
    axs[-1].set_xlabel("Charge Collected [V]")
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    axs[-1].xaxis.set_major_locator(MultipleLocator(0.2))
    axs[-1].xaxis.set_major_formatter("{x:.1f}")
    axs[-1].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].set_xlabel("Charge Collected [V]")
    axs[0].get_xaxis().set_visible(True)
    axs[0].xaxis.set_label_position("top")
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].xaxis.set_major_locator(MultipleLocator(0.2))
    axs[0].xaxis.set_major_formatter("{x:.1f}")
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    plot.fig.suptitle(f"Cluster Charge Distribution By Relative Row")
    plot.saveToPDF(f"Voltages{"" if rangeOfRows==(0,30) else f"_{rangeOfRows[0]}_{rangeOfRows[1]}"}")



def plotMPV(data,path,plotGen):
    plot = plotGen.newPlot(
        path,
        sizePerPlot=(6,4),
    )
    axs = plot.axs
    rows = list(data.fittings.keys())
    for i in rows:
        x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = data.fittings[i]
        axs.scatter(
            i, x_mpv, color=plot.colorPalette[0], marker="x", label="Unconstrained"
        )
        axs.errorbar(
            i,
            x_mpv,
            yerr=x_mpv_e,
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
        if not np.isnan(data.constrainedFittings[i][0]):
            x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = data.constrainedFittings[i]
            axs.scatter(
                i, x_mpv, color=plot.colorPalette[2], marker="x", label="Constrained"
            )
            axs.errorbar(
                i,
                x_mpv,
                yerr=x_mpv_e,
                fmt="none",
                color=plot.colorPalette[2],
                elinewidth=1,
                capsize=3,
            )
    axs.hlines(
        0.161,
        -1,
        rows[-1]+1,
        linestyle="--",
        color=plot.colorPalette[5],
        label="Threshold",
    )
    plot.set_config(
        axs,
        title="MPV per Relative Row in Clusters",
        xlabel="Relative Row [Px]",
        ylabel="MPV [V]",
        xlim=(-1, rows[-1]+1),
        ylim=(0, None),
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(0.05))
    axs.yaxis.set_major_formatter("{x:.2f}")
    axs.yaxis.set_minor_locator(MultipleLocator(0.01))
    axs.grid(True)
    legend_without_duplicate_labels(axs)
    plot.saveToPDF(f"MPV")



def runMPV(dataFiles,plotGen,config):
    for dataFile in dataFiles:
        dataFile.get_crossTalk()
        dataFile.init_cluster_voltages()
        data = mpvData(dataFile)
        data.calcFittings()
        path = f"MPV/{dataFile.fileName}/"
        plotVoltagePlots(data,path,plotGen)
        plotVoltagePlots(data,path,plotGen,rangeOfRows=(2,5),xlim=(0,2))
        plotVoltagePlots(data,path,plotGen,rangeOfRows=(23,26),xlim=(0,1))
        plotMPV(data,path,plotGen)

if __name__ == "__main__":
    config = configLoader.loadConfig("config.json")
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runMPV(dataFiles,plotGen,config)