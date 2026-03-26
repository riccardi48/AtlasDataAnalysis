###############################
# Plots in this file:
# 1. Scatter Plot of MPV vs Relative Row
# 2. Voltages
###############################

import sys
from functions.mpvFuncs import mpvData,histogramHit_Voltage_Errors,landauFunc,fitVoltageDepth,chargeCollectionEfficiencyFunc,depletionWidthFunc
from functions.genericFuncs import getColor,getName
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
                frameon=False,loc="lower left")


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
    return popt,pcov
    _x = np.linspace(0, 90, 1000)
    _y = chargeCollectionEfficiencyFunc(_x, *popt, GeV=GeV)
    ax.plot(
        _x,
        _y,
        color=plot.colorPalette[2],
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
    #ax.fill_between(
    #    _x, np.min(ylist, axis=0), np.max(ylist, axis=0), color=plot.colorPalette[6], zorder=-1
    #)
    ax.text(
        0.98,
        textHeight,
        f"{text}"
        + f"\n$V_0$ :{V_0:.3f} $\\pm$ {V_0_e:.4f} $V$"
        + f"\n$w_0$ :{t_epi:.3f} $\\pm$ {t_epi_e:.3f} $\\mu m$"
        + f"\n$L_n$ :{edl:.3f} $\\pm$ {edl_e:.3f} $\\mu m$",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        size="x-small"
    )
    return popt,pcov

def plotMPV(data,dataFile,path,plotGen):
    plot = plotGen.newPlot(
        path,
        sizePerPlot=(3.4,2.6),
        rect=(0.08,0.09,0.995,0.995),
    )
    axs = plot.axs
    rows = list(data.fittings.keys())
    for i in rows:
        x = i * 50 / np.tan(np.deg2rad(dataFile.angle))
        x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = data.fittings[i]
        axs.scatter(
            x, x_mpv, color=plot.colorPalette[0], marker="^", label="MPVs",s=6,zorder=40,
        )
        axs.errorbar(
            x,
            x_mpv,
            yerr=x_mpv_e,
            fmt="none",
            ls='',
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=0,
            zorder=40,
        )
        if not np.isnan(data.constrainedFittings[i][0]):
            x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = data.constrainedFittings[i]
            axs.scatter(
                x, x_mpv, color=plot.colorPalette[2], marker="s", label="Constrained",s=6,zorder=30,
            )
            axs.errorbar(
                x,
                x_mpv,
                yerr=x_mpv_e,
                fmt="none",
                ls='',
                color=plot.colorPalette[2],
                elinewidth=1,
                capsize=0,
                zorder=30,
            )
    x = np.array(list(data.fittings.keys())) * 50 / np.tan(np.deg2rad(dataFile.angle))
    y = np.array([data.fittings[i][0] for i in data.fittings])
    y_e = np.array([data.fittings[i][3] for i in data.fittings])
    firstLow = x[1:][np.where(y[1:]<=0.2)[0][0]]
    print(firstLow)
    popt1,pcov1 = fitAndPlotCCE(
        axs,
        plot,
        x[x < firstLow],
        y[x < firstLow],
        y_e[x < firstLow],
    )
    (V_0, t_epi, edl) = popt1
    (V_0_e, t_epi_e, edl_e) = np.sqrt(np.diag(pcov1))
    _x = np.linspace(0, 90, 1000)
    _y = chargeCollectionEfficiencyFunc(_x, *popt1)
    plot.axs.plot(
        _x,
        _y,
        color=plot.colorPalette[6],
        linestyle="dashed",
        zorder=20,
    )
    plot.axs.text(
        0.98,
        0.98,
        f"Fitting"
        + f"\n$V_0$ :{V_0:.3f} $\\pm$ {V_0_e:.4f} $V$"
        + f"\n$w_0$ :{t_epi:.3f} $\\pm$ {t_epi_e:.3f} $\\mu m$"
        + f"\n$L_n$ :{edl:.3f} $\\pm$ {edl_e:.3f} $\\mu m$",
        horizontalalignment="right",
        verticalalignment="top",
        transform=plot.axs.transAxes,
        size="x-small"
    )
    index = np.where([np.invert(np.isnan(data.constrainedFittings[i][0])) for i in data.constrainedFittings])
    x[index] = np.array(list(data.constrainedFittings.keys()))[index] * 50 / np.tan(np.deg2rad(dataFile.angle))
    y[index] = np.array([data.constrainedFittings[i][0] for i in data.constrainedFittings])[index]
    y_e[index] = np.array([data.constrainedFittings[i][3] for i in data.constrainedFittings])[index]
    firstLow = x[1:][np.where(y[1:]<=0.1)[0][0]]
    print(firstLow)
    popt2,pcov2 = fitAndPlotCCE(
        axs,
        plot,
        x[x < firstLow],
        y[x < firstLow],
        y_e[x < firstLow],
    )
    (V_0, t_epi, edl) = popt2
    (V_0_e, t_epi_e, edl_e) = np.sqrt(np.diag(pcov2))
    _x = np.linspace(0, 90, 1000)
    _y = chargeCollectionEfficiencyFunc(_x, *popt2)
    plot.axs.plot(
        _x,
        _y,
        color=plot.colorPalette[8],
        linestyle="dashed",
        zorder=10,
    )
    plot.axs.text(
        0.98,
        0.75,
        f"Constrained Fit"
        + f"\n$V_0$ :{V_0:.3f} $\\pm$ {V_0_e:.4f} $V$"
        + f"\n$w_0$ :{t_epi:.3f} $\\pm$ {t_epi_e:.3f} $\\mu m$"
        + f"\n$L_n$ :{edl:.3f} $\\pm$ {edl_e:.3f} $\\mu m$",
        horizontalalignment="right",
        verticalalignment="top",
        transform=plot.axs.transAxes,
        size="x-small"
    )
    plot.set_config(
        axs,
        #title=f"MPV per Relative Row in Clusters - {getName(dataFile)}",
        xlabel=r"Depth $[\mu\, m]$",
        ylabel=r"MPV $[V]$",
        xlim=(-1, axs.get_xlim()[1]),
        ylim=(0, np.max(np.array([data.fittings[i][0] for i in data.fittings])) * 1.2),
        xticks = [10,2],
        yticks = [0.1,0.02],
        yticksSig = 1,
        grid=True,
    )
    axs.hlines(
        0.161,
        axs.get_xlim()[0],
        axs.get_xlim()[1],
        linestyle="-",
        color=plot.colorPalette[5],
        label="Threshold",
        zorder=2,
    )

    legend_without_duplicate_labels(axs)
    plot.saveToPDF(f"MPV")
    return popt1, pcov1, popt2, pcov2

def plotDepletionWidth(dataFiles,path,plotGen,fittings):
    plot = plotGen.newPlot(path,
        sizePerPlot=(3.4,2.6),
        rect=(0.08,0.1,0.995,0.995),
    )
    for dataFile in dataFiles:
        plot.axs.scatter(dataFile.voltage**0.5, fittings[dataFile.fileName][0][1], color=getColor(dataFile), marker="x", s=15)
        plot.axs.errorbar(
            dataFile.voltage**0.5,
            fittings[dataFile.fileName][0][1],
            yerr=fittings[dataFile.fileName][1][1],
            fmt="none",
            color=getColor(dataFile),
            elinewidth=0.5,
            capsize=1,
        )

    func = lambda V, a, b, c: depletionWidthFunc(V, a, b, c)
    func = lambda V,m,c : m*V+c
    Vs = np.array([dataFile.voltage for dataFile in dataFiles])**0.5
    t_epis = np.array([fittings[dataFile.fileName][0][1] for dataFile in dataFiles])
    t_epis_e = np.array([fittings[dataFile.fileName][1][1] for dataFile in dataFiles])
    #initial_guess = (30, 1, 10)
    #bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))
    popt, pcov = curve_fit(
        func,
        Vs[Vs!=0],
        t_epis[Vs!=0],
        sigma=t_epis_e[Vs!=0],
        absolute_sigma=False,
    )
    (m,c) = popt
    (m_e,c_e) = np.sqrt(np.diag(pcov))

    x = np.linspace(0, np.max(Vs)*1.2, 1000)
    y = func(x, *popt)
    plot.axs.plot(
        x,
        y,
        color=plot.colorPalette[1],
        linestyle="dashed",
        #label=f"a : {a:.5f} ± {a_e:.5f} µm/√V\n∆V_bi : {b:.5f} ± {b_e:.5f} V\nc : {c:.5f} ± {c_e:.5f} µm",
        label=f"m : {m:.2f} $\\pm$ {m_e:.2f} \nc : {c:.2f} $\\pm$ {c_e:.2f}",
    )
    plot.set_config(
        plot.axs,
        ylim=(0, 45),
        xlim=(-0.5, None),
        #title="Depletion Depth Vs Bias Voltage",
        xlabel=r"Bias Voltage $[\sqrt{V}]$",
        ylabel=r"Depletion Depth $[\mu m]$",
        legend=True,
        xticks=[1,0.2],
        yticks=[5,1],
        grid=True,
    )
    plot.saveToPDF(
        f"Depletion_Width_Vs_Bias_Voltage"
    )

def plotEDL(dataFiles,path,plotGen,fittings):
    plot = plotGen.newPlot(path,
        sizePerPlot=(3.4,2.6),
        rect=(0.08,0.09,0.995,0.995),
    )
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
        ylim=(0, 40),
        xlim=(-2, 50),
        #title="Electron Diffusion Length Vs Bias Voltage",
        xlabel=r"Bias Voltage $[V]$",
        ylabel=r"Electron Diffusion Length $[\mu\, m]$",
        legend=True,
        xticks=[5,1],
        yticks=[5,1],
        grid=True,
    )
    plot.saveToPDF(
        f"EDL_Vs_Bias_Voltage"
    )

def plotMaxVoltage(dataFiles,path,plotGen,fittings):
    plot = plotGen.newPlot(path,
        sizePerPlot=(3.4,2.6),
        rect=(0.08,0.09,0.995,0.995),
    )
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
        #title="Max Voltage Vs Bias Voltage",
        xlabel=r"Bias Voltage $[V]$",
        ylabel=r"Max Voltage Percent $[\%]$",
        legend=True,
        xticks=[5,1],
        yticks=[0.02,0.01],
        yticksSig=2,
        grid=True,
    )
    plot.saveToPDF(
        f"Max_Voltage_Vs_Bias_Voltage"
    )


def runMPV(dataFiles,plotGen,config):
    mpvPlot1 = plotGen.newPlot("Combined/",
        sizePerPlot=(3.4,2.6),
        rect=(0.08,0.09,0.995,0.995),
    )
    mpvPlot2 = plotGen.newPlot("Combined/",
        sizePerPlot=(3.4,2.6),
        rect=(0.08,0.09,0.995,0.995),
    )
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

        plotVoltagePlots(data,path,plotGen)
        #plotVoltagePlots(data,path,plotGen,rangeOfRows=(2,5),xlim=(0,2))
        maxLength = np.where(np.invert(np.isnan(mpvs1)))[-1][-1]
        print(maxLength)
        #plotVoltagePlots(data,path,plotGen,rangeOfRows=(maxLength-6,maxLength-3),xlim=(0,1))
        popt1, pcov1, popt2, pcov2 = plotMPV(data,dataFile,path,plotGen)
        fittings[dataFile.fileName] = np.array([popt1,np.sqrt(np.diag(pcov1))])
        constrainedFittings[dataFile.fileName] = np.array([popt2,np.sqrt(np.diag(pcov2))])
        firstLow = np.where(mpvs1[1:]<=0.2)[0][0]+1
        mpvPlot1.axs.plot(
            np.arange(mpvs1.size)[:firstLow],
            mpvs1[:firstLow],
            color=getColor(dataFile),
            label=f"{getName(dataFile)}",
        )
        mpvPlot1.axs.fill_between(
            np.arange(mpvs1.size)[:firstLow],
            mpvs1[:firstLow]+mpvs_e1[:firstLow],
            mpvs1[:firstLow]-mpvs_e1[:firstLow],
            alpha=0.2,
            color=getColor(dataFile),
        )
        mpvPlot2.axs.plot(
            np.arange(mpvs2.size),
            mpvs2,
            color=getColor(dataFile),
            label=f"{getName(dataFile)}",
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
        #title="MPV per Relative Row in Clusters",
        xlabel="Relative Row [Px]",
        ylabel="MPV [V]",
        xlim=(-1, data.maxWidth),
        ylim=(0, mpvPlot1.axs.get_ylim()[1]*1.2),
        xticks = [10,2],
        yticks = [0.1,0.02],
        yticksSig = 1,
        grid=True,
    )
    mpvPlot1.saveToPDF(f"MPV")
    mpvPlot2.set_config(
        mpvPlot2.axs,
        legend=True,
        #title="MPV per Relative Row in Clusters",
        xlabel="Relative Row [Px]",
        ylabel="MPV [V]",
        xlim=(-1, data.maxWidth),
        ylim=(0, mpvPlot2.axs.get_ylim()[1]*1.2),
        xticks = [10,2],
        yticks = [0.1,0.02],
        yticksSig = 1,
        grid=True,
    )
    mpvPlot2.saveToPDF(f"MPV_Constrained")
    path = "Combined/"
    plotDepletionWidth(dataFiles,path,plotGen,fittings)
    plotEDL(dataFiles,path,plotGen,fittings)
    plotMaxVoltage(dataFiles,path,plotGen,fittings)
    plotDepletionWidth(dataFiles,path+"_",plotGen,constrainedFittings)
    plotEDL(dataFiles,path+"_",plotGen,constrainedFittings)
    plotMaxVoltage(dataFiles,path+"_",plotGen,constrainedFittings)


if __name__ == "__main__":
    config = configLoader.loadConfig()
    #config["filterDict"] = {"fileName":["angle6_6Gev_kit_4","angle6_6Gev_kitHV30_kit_5","angle6_6Gev_kitHV20_kit_6","angle6_6Gev_kitHV15_kit_7"]}
    #config["filterDict"] = {"fileName":["angle6_6Gev_kitHV20_kit_6","angle6_6Gev_kitHV15_kit_7"]}
    config = configLoader.loadConfig("config.json")
    dataFiles = initDataFiles(config)
    #dataFiles = [dataFiles[0]] + dataFiles[2:]
    plotGen = plotGenerator(config["pathToOutput"])
    runMPV(dataFiles,plotGen,config)