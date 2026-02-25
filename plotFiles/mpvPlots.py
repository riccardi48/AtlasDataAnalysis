###############################
# Plots in this file:
# 1. Scatter Plot of MPV vs Relative Row
# 2. Voltages
###############################

import sys
from functions.mpvFuncs import mpvData,histogramHit_Voltage_Errors,landauFunc,fitVoltageDepth,chargeCollectionEfficiencyFunc,depletionWidthFunc
from functions.genericFuncs import getColor
from plotClass import plotGenerator
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

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


def fitAndPlotCCE(
    ax,
    plot,
    x,
    y,
    yerr,
    GeV: int = 6,
    textHeight = 0.98,
    text = "",
) -> None:
    popt, pcov = fitVoltageDepth(x, y, yerr, GeV=GeV)
    (V_0, t_epi, edl) = popt
    (V_0_e, t_epi_e, edl_e) = np.sqrt(np.diag(pcov))
    _x = np.linspace(0, 90, 1000)
    _y = chargeCollectionEfficiencyFunc(_x, *popt, GeV=GeV)
    ax.plot(
        _x,
        _y,
        color=plot.colorPalette[0],
        linestyle="dashed",
    )

    possibleLines = np.random.default_rng().multivariate_normal(popt, pcov, 1000)
    ylist = [
        chargeCollectionEfficiencyFunc(_x, param[0], param[1], param[2], GeV=GeV)
        for param in possibleLines
    ]
    ylist = [
        chargeCollectionEfficiencyFunc(_x, _V_0, _t_epi, _edl, GeV=GeV)
        for _V_0 in [V_0 - V_0_e, V_0 + V_0_e]
        for _t_epi in [t_epi - t_epi_e, t_epi + t_epi_e]
        for _edl in [edl - edl_e, edl + edl_e]
    ]
    ax.fill_between(
        _x, np.min(ylist, axis=0), np.max(ylist, axis=0), color=plot.colorPalette[6], zorder=-1
    )
    ax.text(
        0.98,
        textHeight,
        f"{text}"
        + f"\nV_0   :{V_0:.5f} ± {V_0_e:.5f}"
        + f"\nt_epi :{t_epi:.3f} ± {t_epi_e:.3f}"
        + f"\nedl    :{edl:.3f} ± {edl_e:.3f}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    return popt,pcov

def plotMPV(data,dataFile,path,plotGen):
    plot = plotGen.newPlot(
        path,
        sizePerPlot=(6,4),
    )
    axs = plot.axs
    rows = list(data.fittings.keys())
    for i in rows:
        x = i * 50 / np.tan(np.deg2rad(dataFile.angle))
        x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = data.fittings[i]
        axs.scatter(
            x, x_mpv, color=plot.colorPalette[0], marker="x", label="Unconstrained"
        )
        axs.errorbar(
            x,
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
                x, x_mpv, color=plot.colorPalette[2], marker="x", label="Constrained"
            )
            axs.errorbar(
                x,
                x_mpv,
                yerr=x_mpv_e,
                fmt="none",
                color=plot.colorPalette[2],
                elinewidth=1,
                capsize=3,
            )
    x = np.array(list(data.fittings.keys())) * 50 / np.tan(np.deg2rad(dataFile.angle))
    y = np.array([data.fittings[i][0] for i in data.fittings])
    y_e = np.array([data.fittings[i][3] for i in data.fittings])
    firstLow = x[np.where(y<=0.17)[0][0]]
    print(firstLow)
    popt1,pcov1 = fitAndPlotCCE(
        axs,
        plot,
        x[x < firstLow],
        y[x < firstLow],
        y_e[x < firstLow],
        text = f"Unconstrained Fit",
    )
    index = np.where([np.invert(np.isnan(data.constrainedFittings[i][0])) for i in data.constrainedFittings])
    x[index] = np.array(list(data.constrainedFittings.keys()))[index] * 50 / np.tan(np.deg2rad(dataFile.angle))
    y[index] = np.array([data.constrainedFittings[i][0] for i in data.constrainedFittings])[index]
    y_e[index] = np.array([data.constrainedFittings[i][3] for i in data.constrainedFittings])[index]
    firstLow = x[np.where(y<=0.13)[0][0]]
    print(firstLow)
    popt2,pcov2 = fitAndPlotCCE(
        axs,
        plot,
        x[x < firstLow],
        y[x < firstLow],
        y_e[x < firstLow],
        textHeight=0.75,
        text = f"Constrained Fit",
    )
    plot.set_config(
        axs,
        title="MPV per Relative Row in Clusters",
        xlabel="Depth [µm]",
        ylabel="MPV [V]",
        xlim=(-1, axs.get_xlim()[1]),
        ylim=(0, np.max(np.array([data.fittings[i][0] for i in data.fittings])) * 1.2),
    )
    axs.hlines(
        0.161,
        axs.get_xlim()[0],
        axs.get_xlim()[1],
        linestyle="--",
        color=plot.colorPalette[5],
        label="Threshold",
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
    return popt1, pcov1, popt2, pcov2

def plotDepletionWidth(dataFiles,path,plotGen,fittings):
    plot = plotGen.newPlot("Combined/",sizePerPlot=(6,4))
    for dataFile in dataFiles:
        plot.axs.scatter(dataFile.voltage, fittings[dataFile.fileName][0][1], color=getColor(dataFile), marker="x", s=15)
        plot.axs.errorbar(
            dataFile.voltage,
            fittings[dataFile.fileName][0][1],
            yerr=fittings[dataFile.fileName][1][1],
            fmt="none",
            color=getColor(dataFile),
            elinewidth=0.5,
            capsize=1,
        )

    func = lambda V, a, b, c: depletionWidthFunc(V, a, b, c)
    Vs = [dataFile.voltage for dataFile in dataFiles]
    t_epis = [fittings[dataFile.fileName][0][1] for dataFile in dataFiles]
    t_epis_e = [fittings[dataFile.fileName][1][1] for dataFile in dataFiles]
    initial_guess = (30, 1, 10)
    bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))
    popt, pcov = curve_fit(
        func,
        Vs,
        t_epis,
        p0=initial_guess,
        bounds=bounds,
        sigma=t_epis_e,
        absolute_sigma=False,
        maxfev=10000000,
    )
    (a, b, c) = popt
    (a_e, b_e, c_e) = np.sqrt(np.diag(pcov))

    x = np.linspace(0, 50, 1000)
    y = depletionWidthFunc(x, a, b, c)
    plot.axs.plot(
        x,
        y,
        color=plot.colorPalette[1],
        linestyle="dashed",
        label=f"a : {a:.5f} ± {a_e:.5f} µm/√V\n∆V_bi : {b:.5f} ± {b_e:.5f} V\nc : {c:.5f} ± {c_e:.5f} µm",
    )
    plot.set_config(
        plot.axs,
        ylim=(0, 45),
        xlim=(-2, 50),
        title="Depletion Depth Vs Bias Voltage",
        xlabel="Bias Voltage [V]",
        ylabel="Depletion Depth [μm]",
        legend=True,
        xticks=[5,1],
        yticks=[5,1],
    )
    plot.axs.grid(True)
    plot.saveToPDF(
        f"Depletion_Width_Vs_Bias_Voltage"
    )

def plotEDL(dataFiles,path,plotGen,fittings):
    plot = plotGen.newPlot("Combined/",sizePerPlot=(6,4))
    for dataFile in dataFiles:
        plot.axs.scatter(dataFile.voltage, fittings[dataFile.fileName][0][2], color=getColor(dataFile), marker="x", s=15)
        plot.axs.errorbar(
            dataFile.voltage,
            fittings[dataFile.fileName][0][2],
            yerr=fittings[dataFile.fileName][1][2],
            fmt="none",
            color=getColor(dataFile),
            elinewidth=0.5,
            capsize=1,
        )
    plot.set_config(
        plot.axs,
        ylim=(20, 55),
        xlim=(-2, 50),
        title="Electron Diffusion Length Vs Bias Voltage",
        xlabel="Bias Voltage [V]",
        ylabel="Electron Diffusion Length [μm]",
        legend=True,
        xticks=[5,1],
        yticks=[5,1],
    )
    plot.axs.grid(True)
    plot.saveToPDF(
        f"EDL_Vs_Bias_Voltage"
    )

def plotMaxVoltage(dataFiles,path,plotGen,fittings):
    plot = plotGen.newPlot("Combined/",sizePerPlot=(6,4))
    for dataFile in dataFiles:
        plot.axs.scatter(dataFile.voltage, fittings[dataFile.fileName][0][0]/fittings["angle6_6Gev_kit_4"][0][0], color=getColor(dataFile), marker="x", s=15)
        plot.axs.errorbar(
            dataFile.voltage,
            fittings[dataFile.fileName][0][0]/fittings["angle6_6Gev_kit_4"][0][0],
            yerr=fittings[dataFile.fileName][1][0]/fittings["angle6_6Gev_kit_4"][0][0],
            fmt="none",
            color=getColor(dataFile),
            elinewidth=0.5,
            capsize=1,
        )
    plot.set_config(
        plot.axs,
        ylim=(0.8, 1.01),
        xlim=(-2, 50),
        title="Max Voltage Vs Bias Voltage",
        xlabel="Bias Voltage [V]",
        ylabel="Max Voltage Percent [%]",
        legend=True,
        xticks=[5,1],
        yticks=[0.02,0.01],
        yticksSig=2,
    )
    plot.axs.grid(True)
    plot.saveToPDF(
        f"Max_Voltage_Vs_Bias_Voltage"
    )


def runMPV(dataFiles,plotGen,config):
    mpvPlot1 = plotGen.newPlot("Combined/",sizePerPlot=(6,4))
    mpvPlot2 = plotGen.newPlot("Combined/",sizePerPlot=(6,4))
    fittings = {}
    constrainedFittings = {}
    for dataFile in dataFiles:
        data = mpvData(dataFile)
        data.initFittings(config)
        data.calcExpectedDepth()
        path = f"MPV/{dataFile.fileName}/"
        mpvs1 = np.array([data.fittings[i][0] for i in data.fittings])
        mpvs_e1 = np.array([data.fittings[i][3] for i in data.fittings])
        mpvs2 = np.array([data.constrainedFittings[i][0] if not np.isnan(data.constrainedFittings[i][0]) else data.fittings[i][0] for i in data.constrainedFittings])
        mpvs_e2 = np.array([data.constrainedFittings[i][3] if not np.isnan(data.constrainedFittings[i][3]) else data.fittings[i][3] for i in data.constrainedFittings])

        #plotVoltagePlots(data,path,plotGen)
        #plotVoltagePlots(data,path,plotGen,rangeOfRows=(2,5),xlim=(0,2))
        maxLength = np.where(np.invert(np.isnan(mpvs1)))[-1][-1]
        print(maxLength)
        #plotVoltagePlots(data,path,plotGen,rangeOfRows=(maxLength-6,maxLength-3),xlim=(0,1))
        popt1, pcov1, popt2, pcov2 = plotMPV(data,dataFile,path,plotGen)
        fittings[dataFile.fileName] = np.array([popt1,np.sqrt(np.diag(pcov1))])
        constrainedFittings[dataFile.fileName] = np.array([popt2,np.sqrt(np.diag(pcov2))])
        
        mpvPlot1.axs.plot(
            np.arange(mpvs1.size),
            mpvs1,
            color=getColor(dataFile),
            label=f"{dataFile.fileName[7:]}",
        )
        mpvPlot1.axs.fill_between(
            np.arange(mpvs1.size),
            mpvs1+mpvs_e1,
            mpvs1-mpvs_e1,
            alpha=0.2,
            color=getColor(dataFile),
        )

        mpvPlot2.axs.plot(
            np.arange(mpvs2.size),
            mpvs2,
            color=getColor(dataFile),
            label=f"{dataFile.fileName[7:]}",
        )
        mpvPlot2.axs.fill_between(
            np.arange(mpvs2.size),
            mpvs2+mpvs_e2,
            mpvs2-mpvs_e2,
            alpha=0.2,
            color=getColor(dataFile),
        )
    mpvPlot1.set_config(
        mpvPlot1.axs,
        legend=True,
        title="MPV per Relative Row in Clusters",
        xlabel="Relative Row [Px]",
        ylabel="MPV [V]",
        xlim=(-1, data.maxWidth),
        ylim=(0, mpvPlot1.axs.get_ylim()[1]*1.2),
    )
    mpvPlot1.axs.grid(True)
    mpvPlot1.saveToPDF(f"MPV")
    mpvPlot2.set_config(
        mpvPlot2.axs,
        legend=True,
        title="MPV per Relative Row in Clusters",
        xlabel="Relative Row [Px]",
        ylabel="MPV [V]",
        xlim=(-1, data.maxWidth),
        ylim=(0, mpvPlot2.axs.get_ylim()[1]*1.2),
    )
    mpvPlot2.axs.grid(True)
    mpvPlot2.saveToPDF(f"MPV_Constrained")
    plotDepletionWidth(dataFiles,path,plotGen,fittings)
    plotEDL(dataFiles,path,plotGen,fittings)
    plotMaxVoltage(dataFiles,path,plotGen,fittings)


if __name__ == "__main__":
    config = configLoader.loadConfig()
    #config["filterDict"] = {"fileName":["angle6_6Gev_kit_4","angle6_6Gev_kitHV30_kit_5","angle6_6Gev_kitHV20_kit_6","angle6_6Gev_kitHV15_kit_7"]}
    #config["filterDict"] = {"fileName":["angle6_6Gev_kitHV20_kit_6","angle6_6Gev_kitHV15_kit_7"]}
    dataFiles = initDataFiles(config)
    plotGen = plotGenerator(config["pathToOutput"])
    runMPV(dataFiles,plotGen,config)