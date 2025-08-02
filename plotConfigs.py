from plotAnalysis import depthAnalysis, plotClass, correlationPlotter, fitAndPlotCCE, fit_dataFile
from AtlasDataAnalysis.Code.dataAnalysis.dataAnalysis import dataAnalysis, crossTalkFinder, initDataFiles
from AtlasDataAnalysis.Code.lowLevelFunctions import (
    calcDepth,
    adjustPeakVoltage,
    histogramErrors,
    landauFunc,
    lambert_W_ToT_to_u,
    chargeCollectionEfficiencyFunc,
    print_mem_usage,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from typing import Optional
from landau import landau
import scipy as scipy

def AngleDistribution(
    dataFile: dataAnalysis, depth: depthAnalysis, pathToOutput: str, saveToPDF: bool = True
):
    dList = np.linspace(1.2, 2.0, 5)
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    for i in range(len(dList)):
        bins, values = depth.findClusterAngleDistribution(dataFile, dList[i])
        axs.stairs(
            values,
            np.rad2deg(bins),
            label=f"{dList[i]*50:.0f} μm",
            baseline=None,
            color=plot.colorPalette[i],
        )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(40, 90),
        title="Angle Distribution",
        legend=True,
        xlabel="Equivalent Angle [Degrees]",
        ylabel="Frequency",
    )
    axs.vlines(
        dataFile.get_angle(), 0, axs.get_ylim()[1], colors=plot.textColor, linestyles="dashed"
    )
    axs.text(
        dataFile.get_angle(),
        axs.get_ylim()[1],
        dataFile.get_angle(),
        color=plot.textColor,
        fontweight="bold",
        horizontalalignment="right",
        verticalalignment="top",
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5000))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1000))
    if saveToPDF:
        plot.saveToPDF("VoltageDepth/AngleDistribution")
    else:
        return plot.fig


def AngleDistribution_2(
    dataFile: dataAnalysis, depth: depthAnalysis, pathToOutput: str, saveToPDF: bool = True
):
    d = depth.find_d_value(dataFile)
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    bins, values = depth.findClusterAngleDistribution(dataFile, d)
    axs.stairs(
        values, np.rad2deg(bins), label=f"{d*50:.2f} μm", baseline=None, color=plot.colorPalette[3]
    )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(40, 90),
        title="Angle Distribution",
        legend=True,
        xlabel="Equivalent Angle [Degrees]",
        ylabel="Frequency",
    )
    axs.vlines(
        dataFile.get_angle(), 0, axs.get_ylim()[1], colors=plot.textColor, linestyles="dashed"
    )
    axs.text(
        dataFile.get_angle(),
        axs.get_ylim()[1],
        dataFile.get_angle(),
        color=plot.textColor,
        fontweight="bold",
        horizontalalignment="right",
        verticalalignment="top",
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5000))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1000))
    if saveToPDF:
        plot.saveToPDF("VoltageDepth/AngleDistribution_2")
    else:
        return plot.fig


def WidthDistribution(
    dataFile: dataAnalysis, depth: depthAnalysis, pathToOutput: str, saveToPDF: bool = True
):
    x, y = depth.findClusterWidthDistribution(dataFile)
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    axs.bar(x, y, width=1, color=plot.colorPalette[3])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(None, None),
        title="Width Distribution",
        legend=False,
        xlabel="Width [px]",
        ylabel="Frequency",
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5000))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1000))
    if saveToPDF:
        plot.saveToPDF("VoltageDepth/WidthDistribution")
    else:
        return plot.fig


def ColumnWidthDistribution(dataFile, pathToOutput, layer=4, saveToPDF=True):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    bins = 30
    range = (0, 30)
    height, x = np.histogram(
        dataFile.get_cluster_attr("ColumnWidths", layer=layer, excludeCrossTalk=True)[0],
        bins=bins,
        range=range,
    )
    axs.bar(x[:-1], height, width=1, color=plot.colorPalette[3])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=range,
        title="Column Width Distribution",
        legend=False,
        xlabel="Column Width [px]",
        ylabel="Frequency",
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(10000))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(5000))
    if saveToPDF:
        plot.saveToPDF(f"ColumnWidthDistribution_{layer}")
    else:
        return plot.fig


def RowWidthDistribution(
    dataFile: dataAnalysis, pathToOutput: str, layer: int = 4, saveToPDF: bool = True
):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    bins = 60
    range = (0, 60)
    rowsWidths, _ = dataFile.get_cluster_attr("RowWidths", layer=layer, excludeCrossTalk=True)
    height, x = np.histogram(rowsWidths, bins=bins, range=range)
    axs.bar(x[:-1], height, width=1, color=plot.colorPalette[3])
    plot.set_config(
        axs,
        ylim=(1, None),
        xlim=range,
        title="Row Width Distribution",
        legend=False,
        xlabel="Row Width [px]",
        ylabel="Frequency",
    )
    axs.xaxis.set_major_locator(MultipleLocator(50))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(10))
    if saveToPDF:
        plot.saveToPDF(f"RowWidthDistribution_{layer}")
    else:
        return plot.fig


def VoltageDepthScatter(
    dataFile: dataAnalysis,
    depth: depthAnalysis,
    pathToOutput: str,
    annotate: bool = False,
    depthCorrection: bool = True,
    hideLowWidths: bool = True,
    fitting: str = "histogram",
    measuredAttribute: str = "Hit_Voltage",
    saveToPDF: bool = True,
):
    d = depth.find_d_value(dataFile)
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    cmap = plt.get_cmap("hsv")
    allXValues: list[float] = []
    allYValues: list[float] = []
    allYValuesErrors: list[float] = []
    for i in range(2, depth.maxClusterWidth + 1):
        x = calcDepth(
            d,
            i,
            dataFile.get_angle(),
            depthCorrection=depthCorrection,
            upTwo=True if dataFile.get_fileName() == "angle6_4Gev_kit_2" else False,
        )
        x = calcDepth(
            d,
            i,
            dataFile.get_angle(),
            depthCorrection=depthCorrection,
            upTwo=False,
        )
        y, y_err = depth.findPeak(dataFile, i, fitting=fitting, measuredAttribute=measuredAttribute)
        #if measuredAttribute == "Hit_Voltage":
        #    y, y_err = depth.findPeak(dataFile, i, fitting=fitting, measuredAttribute="ToT")
        #    y_err = y_err * (lambert_W_ToT_to_u(y, 0.161, 8, 110, 70) / y)
        #    y = lambert_W_ToT_to_u(y, 0.161, 8, 110, 70)
        # y, y_err = adjustPeakVoltage(y, y_err, d, i)
        if not hideLowWidths or (np.rad2deg(np.arctan(i / d)) > 85 and np.rad2deg(np.arctan(i / d)) < 86.8):
            axs.scatter(
                x, y, color=cmap((i - 2) / depth.maxClusterWidth), marker="x", s=15, label=str(i)
            )
            axs.errorbar(
                x,
                y,
                yerr=y_err,
                fmt="none",
                color=cmap((i - 2) / depth.maxClusterWidth),
                elinewidth=0.5,
                capsize=1,
            )
            allXValues = allXValues + list(x[1:-1])
            allYValues = allYValues + list(y[1:-1])
            allYValuesErrors = allYValues + list(y_err[1:-1])

    rightSide = axs.get_xlim()[1]
    allXValues_np = np.array(allXValues)
    allYValues_np = np.array(allYValues)
    allYValuesErrors_np = np.array(allYValuesErrors)
    y = allYValues_np[np.argsort(allXValues_np)]
    yerr = allYValuesErrors_np[np.argsort(allXValues_np)]
    x = allXValues_np[np.argsort(allXValues_np)]
    cut = (y>0.10) & (x < d*50*0.9)
    fitAndPlotCCE(axs, plot, x[cut], y[cut], yerr[cut],GeV=4 if dataFile.get_fileName() == "angle6_4Gev_kit_2" else 6)
    
    plot.set_config(
        axs,
        legend=True,
        ncols=4,
        labelspacing=0.3,
        loc="lower left",
        handletextpad=0.1,
        columnspacing=0.3,
        legendTitle="Cluster Width",
    )
    axs.xaxis.set_major_locator(MultipleLocator(10))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(2))
    if measuredAttribute == "Hit_Voltage":
        axs.hlines(0.162, 0, rightSide, colors=plot.colorPalette[1], linestyles="dashed")
        plot.set_config(
            axs,
            ylim=(0, 0.64),
            xlim=(0, rightSide),
            title="Voltage change withing a Cluster",
            xlabel="Depth [μm]",
            ylabel="Voltage [V]",
        )
        axs.yaxis.set_major_locator(MultipleLocator(0.05))
        axs.yaxis.set_major_formatter("{x:.2f}")
        axs.yaxis.set_minor_locator(MultipleLocator(0.01))
    elif measuredAttribute == "ToT":
        plot.set_config(
            axs,
            ylim=(0, 120),
            xlim=(0, rightSide),
            title="ToT withing a Cluster",
            xlabel="Depth [μm]",
            ylabel="ToT [TS]",
        )
        axs.yaxis.set_major_locator(MultipleLocator(10))
        axs.yaxis.set_major_formatter("{x:.0f}")
        axs.yaxis.set_minor_locator(MultipleLocator(2))
    if annotate:
        lineList = [40, 65]
        axs.vlines(lineList, 0.05, 0.3, colors=plot.textColor, linestyles="dashed")
        for i in lineList:
            axs.text(
                i,
                0.05,
                i,
                color=plot.textColor,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="top",
            )
    if saveToPDF:
        plot.saveToPDF(
            f"VoltageDepth/{"Voltage" if measuredAttribute == "Hit_Voltage" else "ToT"}DepthScatter"
            + f"{f"_nnlf" if fitting == "nnlf" else ""}"
            + f"{f"_NoDepthCorrection" if not depthCorrection else ""}"
            + f"{F"_NoHideLowWidths" if not hideLowWidths else ""}"
        )
    else:
        return plot.fig


def Hit_VoltageDistributionByPixel(
    dataFile: dataAnalysis,
    depth: depthAnalysis,
    clusterWidth: int,
    pathToOutput: str,
    _range: tuple[float, float] = (0.162, 2),
    measuredAttribute: str = "Hit_Voltage",
    saveToPDF: bool = True,
):
    plot = plotClass(
        pathToOutput + f"{dataFile.get_fileName()}/",
        shape=(1, clusterWidth),
        sharex=True,
        sizePerPlot=(10, 2),
        hspace=0,
    )
    axs = np.flip(plot.axs)
    hitPositionArray, _ = depth.loadOneLength(
        dataFile, clusterWidth, measuredAttribute=measuredAttribute
    )
    hitPositionErrorArray, _ = depth.loadOneLength(
        dataFile, clusterWidth, error=True, measuredAttribute=measuredAttribute
    )
    params_histogram = depth.findPeaks_widthRestricted(
        hitPositionArray,
        hitPositionErrorArray,
        fitting="histogram",
        _range=_range,
        params=[0, 1, 2, 3, 4, 5],
    )
    for j in range(clusterWidth):
        values = hitPositionArray[j][hitPositionArray[j] != 0]
        errors = hitPositionErrorArray[j][hitPositionArray[j] != 0]
        errors = errors[np.invert(np.isnan(values))]
        values = values[np.invert(np.isnan(values))]
        hist, binEdges, binCentres = depth.histogramHit_Voltage(values, _range=_range)
        histErrors = histogramErrors(values, binEdges, errors)
        axs[j].errorbar(binCentres, hist, histErrors, fmt="none", color=plot.colorPalette[6])
        axs[j].step(
            binEdges,
            np.append(hist[0], hist),
            c=plot.colorPalette[3],
            linewidth=1,
            label=f"Pixel {clusterWidth-j}",
        )
        x = np.linspace(np.min(binEdges) - 0.1, np.max(binEdges) + 0.1, 1000)
        x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = params_histogram[j]
        y = landauFunc(x, x_mpv, xi, scale)
        axs[j].plot(
            x,
            y,
            c=plot.colorPalette[0],
            label=f"Mpv:{x_mpv:.5f} ± {x_mpv_e:.5f}\nWidth:{xi:.5f} ± {xi_e:.5f}\nScale:{scale:.5f} ± {scale_e:.5f}",
        )
        axs[j].errorbar(
            x[np.argmax(y)],
            y[np.argmax(y)],
            xerr=[x_mpv_e],
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
        axs[j].get_xaxis().set_visible(False)
        plot.set_config(
            axs[j],
            ylim=(0, None),
            xlim=(float(np.min(binEdges)), float(np.max(binEdges))),
            legend=True,
        )
        #axs[j].yaxis.set_major_locator(MultipleLocator(100))
        #axs[j].yaxis.set_major_formatter("{x:.0f}")
        #axs[j].yaxis.set_minor_locator(MultipleLocator(20))
    if measuredAttribute == "Hit_Voltage":
        axs[0].set_xlabel("Hit Voltage [V]")
        axs[0].get_xaxis().set_visible(True)
        axs[0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
        axs[0].xaxis.set_major_locator(MultipleLocator(0.1))
        axs[0].xaxis.set_major_formatter("{x:.1f}")
        axs[0].xaxis.set_minor_locator(MultipleLocator(0.02))
        axs[-1].set_xlabel("Hit Voltage [V]")
        axs[-1].get_xaxis().set_visible(True)
        axs[-1].xaxis.set_label_position("top")
        axs[-1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        axs[-1].xaxis.set_major_locator(MultipleLocator(0.1))
        axs[-1].xaxis.set_major_formatter("{x:.1f}")
        axs[-1].xaxis.set_minor_locator(MultipleLocator(0.02))
    elif measuredAttribute == "ToT":
        axs[0].set_xlabel("ToT [TS]")
        axs[0].get_xaxis().set_visible(True)
        axs[0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
        axs[0].xaxis.set_major_locator(MultipleLocator(30))
        axs[0].xaxis.set_major_formatter("{x:.0f}")
        axs[0].xaxis.set_minor_locator(MultipleLocator(10))
        axs[-1].set_xlabel("ToT [TS]")
        axs[-1].get_xaxis().set_visible(True)
        axs[-1].xaxis.set_label_position("top")
        axs[-1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        axs[-1].xaxis.set_major_locator(MultipleLocator(30))
        axs[-1].xaxis.set_major_formatter("{x:.0f}")
        axs[-1].xaxis.set_minor_locator(MultipleLocator(10))

    plot.fig.suptitle(f"{clusterWidth} Width Cluster Charge Distribution By Pixel")
    if saveToPDF:
        plot.saveToPDF(
            f"VoltageDepth/ByWidth/{"Hit_Voltage" if measuredAttribute == "Hit_Voltage" else "ToT"}DistributionByPixel_{clusterWidth}"
        )
    else:
        return plot.fig


def HitDistributionInCluster(
    dataFile: dataAnalysis,
    depth: depthAnalysis,
    clusterWidth: int,
    pathToOutput: str,
    saveToPDF: bool = True,
):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    hitPositionArray = np.flip(depth.loadOneLength(dataFile, clusterWidth)[0])
    d = depth.find_d_value(dataFile)
    heights = np.zeros((clusterWidth))
    x = np.linspace(0, 1 - (0.5 / clusterWidth), clusterWidth + 1) * d * 50

    if clusterWidth > (d * np.tan(np.deg2rad(dataFile.get_angle()))):
        x = x * (clusterWidth / (d * np.tan(np.deg2rad(dataFile.get_angle()))))
    for j in range(clusterWidth):
        heights[j] = np.sum(hitPositionArray[j] > 0)
    axs.stairs(heights, x, color=plot.colorPalette[3], label="Original")
    for j in range(clusterWidth):
        heights[j] = np.sum(hitPositionArray[j] > 0.162)
    axs.stairs(heights, x, color=plot.colorPalette[0], label="Low Voltage Removed")
    for j in range(clusterWidth):
        heights[j] = np.sum(
            hitPositionArray[j][(hitPositionArray[0] <= 0.162) | (hitPositionArray[-1] <= 0.162)]
            > 0
        )
    axs.stairs(heights, x, color=plot.colorPalette[1], label="At Least one ead with Low Voltage")
    for j in range(clusterWidth):
        heights[j] = np.sum(
            hitPositionArray[j][(hitPositionArray[0] > 0.162) & (hitPositionArray[-1] > 0.162)] > 0
        )
    axs.stairs(
        heights, x, color=plot.colorPalette[2], label="Only Clusters Without Low Voltage Ends"
    )
    for j in range(clusterWidth):
        heights[j] = np.sum(
            hitPositionArray[j][(hitPositionArray[0] > 0.162) & (hitPositionArray[-1] > 0.162)]
            > 0.162
        )
    axs.stairs(
        heights,
        x,
        color=plot.colorPalette[4],
        label="Only Clusters Without Low Voltage Ends\n and Low Voltage Removed",
    )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(np.min(x), np.max(x)),
        title=f"Width {clusterWidth} Pixel Hit Distribution",
        legend=True,
        xlabel="Depth [μm]",
        ylabel="Frequency",
    )
    if saveToPDF:
        plot.saveToPDF(f"VoltageDepth/ByWidth/HitDistributionInCluster_{clusterWidth}")
    else:
        return plot.fig


def HitDistributionInClusterAllOnOne(
    dataFile: dataAnalysis,
    depth: depthAnalysis,
    pathToOutput: str,
    vmin: int = 2,
    vmax: int = 40,
    cutting: bool = False,
    saveToPDF: bool = True,
):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    cmap = plt.get_cmap("hsv")
    d = depth.find_d_value(dataFile)
    for clusterWidth in range(vmin, vmax + 1):
        hitPositionArray = np.flip(depth.loadOneLength(dataFile, clusterWidth)[0])
        heights = np.zeros((clusterWidth))
        x = calcDepth(
            d,
            clusterWidth,
            dataFile.get_angle(),
            depthCorrection=True if dataFile.get_fileName() == "angle6_4Gev_kit_2" else False,
            upTwo=False,
        )
        x = np.sort(x)
        diff = x[1]-x[0]
        x = np.append(x[0]-diff/2,x[:]+diff/2)
        
        for j in range(clusterWidth):
            if cutting:
                heights[j] = np.sum(
                    hitPositionArray[j][
                        (hitPositionArray[0] > 0.162) & (hitPositionArray[-1] > 0.162)
                    ]
                    > 0.162
                )
            else:
                heights[j] = np.sum(hitPositionArray[j] > 0)

        heights = heights / heights[0]
        axs.stairs(
            heights,
            x,
            color=cmap((clusterWidth) / (depth.maxClusterWidth)),
            baseline=None,
        )
    popt, pcov, _, __, ___ = fit_dataFile(
            dataFile,
            depth,
            depthCorrection=True if dataFile.get_fileName() == "angle6_4Gev_kit_2" else False,
            hideLowWidths=True,
            fitting="histogram",
            measuredAttribute="Hit_Voltage",
            GeV=4 if dataFile.get_fileName() == "angle6_4Gev_kit_2" else 6,
            maxClusterWidth=29,
        )
    x = np.linspace(0, np.max(x), 1000)
    MPVs = chargeCollectionEfficiencyFunc(x, *popt,GeV=4 if dataFile.get_fileName() == "angle6_4Gev_kit_2" else 6)
    y = np.zeros(x.size)
    def func(x,m,c):
        return m*x+c
    for i,MPV in enumerate(MPVs):
        width = func(MPV,0.3024912,-0.01518574)
        #width = (0.035/(1+np.exp(30*(MPV-0.18)))) + 0.4*(MPV-0.17)
        if width < 0.001:
            width = 0.001
        y[i] = (1 - landau.cdf(0.161, MPV, width))
    y=y/y[0]
    axs.plot(x[MPVs > 0.04], y[MPVs > 0.04], linestyle="dashed", label="Expected", color=plot.colorPalette[3])
    for i,MPV in enumerate(MPVs):
        width = func(MPV,0.28514695,0.0066855)
        width = (MPV-0.09)*0.4
        if width < 0.001:
            width = 0.001
        y[i] = (1 - landau.cdf(0.161, MPV, width))
    y=y/y[0]
    axs.plot(x[MPVs > 0.09], y[MPVs > 0.09], linestyle="dashed", label="Expected", color=plot.colorPalette[2])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(0, None),
        title="Hit Distribution In Cluster",
        legend=True,
        xlabel="Depth [μm]",
        ylabel="Percent [%]",
        ncols=4,
        labelspacing=0.3,
        loc="lower left",
        handletextpad=0.1,
        columnspacing=0.3,
        legendTitle="Cluster Width",
    )
    axs.xaxis.set_major_locator(MultipleLocator(10))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(5))
    axs.yaxis.set_major_locator(MultipleLocator(0.1))
    axs.yaxis.set_major_formatter("{x:.1f}")
    axs.yaxis.set_minor_locator(MultipleLocator(0.05))
    if saveToPDF:
        plot.saveToPDF(
            f"VoltageDepth/HitDistributionInClusterAllOnOne_{vmin}_{vmax}{f"_cut" if cutting else ""}"
        )
    else:
        return plot.fig


def CuttingComparison(
    dataFile: dataAnalysis, pathToOutput: str, layers: Optional[int] = None, saveToPDF: bool = True
):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/", shape=(2, 2), sizePerPlot=(5, 4))
    axs = plot.axs
    bins = 128
    range = (0, 256)
    height, x = np.histogram(dataFile.get_base_attr("ToT", layers=layers)[0], bins=bins, range=range)
    axs[0, 0].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[3],
        linewidth=1,
        zorder=2,
        label="Raw",
    )
    height, x = np.histogram(
        dataFile.get_base_attr("ToT", layers=layers, excludeCrossTalk=True)[0], bins=bins, range=range
    )
    axs[0, 0].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[0],
        linewidth=1,
        zorder=1,
        label="CrossTalk Cut",
    )
    plot.set_config(
        axs[0, 0],
        xlim=range,
        xlabel="ToT [TS]",
        ylabel="Frequency",
        title="Time over threshold distribution",
        legend=True,
    )
    if dataFile.get_fileName() in ["angle6_6Gev_kit_4", "angle6_4Gev_kit_2"]:
        axs[0, 0].set_ylim(0, 200000)
        axs[0, 0].yaxis.set_major_locator(MultipleLocator(10000))
        axs[0, 0].yaxis.set_major_formatter("{x:.0f}")
        axs[0, 0].yaxis.set_minor_locator(MultipleLocator(2000))
    elif dataFile.get_fileName() == "6Gev_kit_0":
        axs[0, 0].set_ylim(0, 200000)
        axs[0, 0].yaxis.set_major_locator(MultipleLocator(10000))
        axs[0, 0].yaxis.set_major_formatter("{x:.0f}")
        axs[0, 0].yaxis.set_minor_locator(MultipleLocator(2000))
    else:
        axs[0, 0].set_ylim(0, None)
        axs[0, 0].yaxis.set_major_locator(MultipleLocator(10000))
        axs[0, 0].yaxis.set_major_formatter("{x:.0f}")
        axs[0, 0].yaxis.set_minor_locator(MultipleLocator(2000))
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(32))
    axs[0, 0].xaxis.set_major_formatter("{x:.0f}")
    axs[0, 0].xaxis.set_minor_locator(MultipleLocator(8))
    axs[0, 0].spines["right"].set_visible(False)
    axs[0, 0].spines["top"].set_visible(False)

    bins = 132
    range = (0, bins)
    height, x = np.histogram(
        dataFile.get_base_attr("Column", layers=layers)[0], bins=bins, range=range
    )
    axs[1, 0].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[3],
        linewidth=1,
        zorder=2,
        label="Raw",
    )
    height, x = np.histogram(
        dataFile.get_base_attr("Column", layers=layers, excludeCrossTalk=True)[0],
        bins=bins,
        range=range,
    )
    axs[1, 0].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[0],
        linewidth=1,
        zorder=1,
        label="CrossTalk Cut",
    )
    plot.set_config(
        axs[1, 0],
        xlim=range,
        ylim=(0, None),
        xlabel="Column [px]",
        ylabel="Frequency",
        title="Column distribution",
        legend=True,
    )

    axs[1, 0].spines["right"].set_visible(False)
    axs[1, 0].spines["top"].set_visible(False)
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(12))
    axs[1, 0].xaxis.set_major_formatter("{x:.0f}")
    axs[1, 0].xaxis.set_minor_locator(MultipleLocator(3))
    axs[1, 0].yaxis.set_major_locator(MultipleLocator(10000))
    axs[1, 0].yaxis.set_major_formatter("{x:.0f}")
    axs[1, 0].yaxis.set_minor_locator(MultipleLocator(2000))

    bins = 372
    range = (0, bins)
    height, x = np.histogram(dataFile.get_base_attr("Row", layers=layers)[0], bins=bins, range=range)
    axs[1, 1].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[3],
        linewidth=1,
        zorder=2,
        label="Raw",
    )
    height, x = np.histogram(
        dataFile.get_base_attr("Row", layers=layers, excludeCrossTalk=True)[0], bins=bins, range=range
    )
    axs[1, 1].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[0],
        linewidth=1,
        zorder=1,
        label="CrossTalk Cut",
    )
    plot.set_config(
        axs[1, 1],
        xlim=range,
        ylim=(0, None),
        xlabel="Row [px]",
        ylabel="Frequency",
        title="Row distribution",
        legend=True,
    )

    axs[1, 1].spines["right"].set_visible(False)
    axs[1, 1].spines["top"].set_visible(False)
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(30))
    axs[1, 1].xaxis.set_major_formatter("{x:.0f}")
    axs[1, 1].xaxis.set_minor_locator(MultipleLocator(10))
    axs[1, 1].yaxis.set_major_locator(MultipleLocator(2000))
    axs[1, 1].yaxis.set_major_formatter("{x:.0f}")
    axs[1, 1].yaxis.set_minor_locator(MultipleLocator(500))
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(50))
    axs[1, 1].xaxis.set_major_formatter("{x:.0f}")
    axs[1, 1].xaxis.set_minor_locator(MultipleLocator(10))

    bins = 372
    range = (0, bins)
    height, x = np.histogram(
        dataFile.get_cluster_attr("RowWidths", layers=layers)[0], bins=bins, range=range
    )
    axs[0, 1].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[3],
        linewidth=1,
        zorder=2,
        label="Raw",
    )
    height, x = np.histogram(
        dataFile.get_cluster_attr("RowWidths", layers=layers, excludeCrossTalk=True)[0],
        bins=bins,
        range=range,
    )
    axs[0, 1].step(
        x,
        np.append(height[0], height),
        color=plot.colorPalette[0],
        linewidth=1,
        zorder=1,
        label="CrossTalk Cut",
    )
    plot.set_config(
        axs[0, 1],
        xlim=range,
        ylim=(0, None),
        xlabel="Row Width [px]",
        ylabel="Frequency",
        title="Row Width distribution",
        legend=True,
    )

    if dataFile.get_fileName() == "angle6_6Gev_kit_4":
        axs[0, 1].yaxis.set_major_locator(MultipleLocator(5000))
        axs[0, 1].yaxis.set_major_formatter("{x:.0f}")
        axs[0, 1].yaxis.set_minor_locator(MultipleLocator(1000))
    elif dataFile.get_fileName() == "6Gev_kit_0":
        axs[0, 1].yaxis.set_major_locator(MultipleLocator(50000))
        axs[0, 1].yaxis.set_major_formatter("{x:.0f}")
        axs[0, 1].yaxis.set_minor_locator(MultipleLocator(10000))
    else:
        axs[0, 1].yaxis.set_major_locator(MultipleLocator(5000))
        axs[0, 1].yaxis.set_major_formatter("{x:.0f}")
        axs[0, 1].yaxis.set_minor_locator(MultipleLocator(1000))
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(10))
    axs[0, 1].xaxis.set_major_formatter("{x:.0f}")
    axs[0, 1].xaxis.set_minor_locator(MultipleLocator(2))
    axs[0, 1].spines["right"].set_visible(False)
    axs[0, 1].spines["top"].set_visible(False)
    plot.fig.suptitle(
        f"{dataFile.get_fileName()} removed cross talk comparison", fontsize="x-large"
    )
    if saveToPDF:
        plot.saveToPDF(f"CutComparison{f"_{layers}" if layers is not None else ""}")
    else:
        return plot.fig


def ClustersCountOverTime(
    dataFile: dataAnalysis,
    pathToOutput: str,
    layers: list[int] = [1, 2, 3, 4],
    saveToPDF: bool = True,
):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    for layer in layers:
        times = dataFile.get_cluster_attr("Times", layer=layer, excludeCrossTalk=True)[0] / 1000
        bins = 60
        maxTime = np.max(times)
        maxTime = 300
        range = (0, maxTime)
        bins = int(maxTime / 1)
        height, x = np.histogram(times, bins=bins, range=range)
        axs.stairs(height, x, baseline=None, color=plot.colorPalette[layer], label=f"Layer:{layer}")
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=range,
        title="Clusters Count Over Time",
        legend=True,
        xlabel="Time [s]",
        ylabel="Frequency",
    )
    if saveToPDF:
        plot.saveToPDF(f"ClustersCountOverTime_{layer}")
    else:
        return plot.fig


def RowRowCorrelation(
    dataFile: dataAnalysis,
    pathToOutput: str,
    pathToCalcData: str,
    layers: list[int] = [1, 2, 3, 4],
    excludeCrossTalk: bool = True,
    recalc: bool = False,
    log: bool = True,
    maxLine: Optional[int] = None,
    saveToPDF: bool = True,
):
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/", sizePerPlot=(8, 8))
    axs = plot.axs

    rowRowPlotter = correlationPlotter(
        pathToCalcData, layers=layers, excludeCrossTalk=excludeCrossTalk, maxLine=maxLine
    )
    RowRow = rowRowPlotter.RowRowCorrelation(dataFile, recalc=recalc)
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
    for _x, _y in tempCrossTalkFinder.crossTalkFunction().items():
        for i, j in _y:
            x.append(i)
            y.append(j)

    axs.scatter(x, y, c="r", s=1)
    if saveToPDF:
        plot.saveToPDF(
            f"RowRowCorrelation{"_cut" if excludeCrossTalk else ""}{"_"+"".join(str(x) for x in layers) if layers is not None else ""}"
        )
    else:
        return plot.fig

def LandauMPVWidthScatter(
    dataFile: dataAnalysis,
    depth: depthAnalysis,
    pathToOutput: str,
    fitting: str = "histogram",
    measuredAttribute: str = "Hit_Voltage",
    saveToPDF: bool = True,
    _range: tuple[float, float] = (0.162, 4),
    hideLowWidths: bool = True,
):
    d = depth.find_d_value(dataFile)
    plot = plotClass(pathToOutput + f"{dataFile.get_fileName()}/")
    axs = plot.axs
    cmap = plt.get_cmap("hsv")
    allXValues: list[float] = []
    allYValues: list[float] = []
    allYValueErrors = []
    for i in range(2, depth.maxClusterWidth + 1):
        if not hideLowWidths or (np.rad2deg(np.arctan(i / d)) > 85 and np.rad2deg(np.arctan(i / d)) < 87):
            hitPositionArray, _ = depth.loadOneLength(
                    dataFile, i, measuredAttribute=measuredAttribute
                )
            hitPositionErrorArray, _ = depth.loadOneLength(
                    dataFile, i, error=True, measuredAttribute=measuredAttribute
                )
            params_histogram = depth.findPeaks_widthRestricted(
                hitPositionArray,
                hitPositionErrorArray,
                fitting=fitting,
                _range=_range,
                params=[0, 1, 2, 3, 4, 5],
            )
            axs.scatter(
                params_histogram[1:-1,0], params_histogram[1:-1,1], color=cmap((i - 2) / depth.maxClusterWidth), marker="x", s=15
            )
            axs.errorbar(
                params_histogram[1:-1,0], 
                params_histogram[1:-1,1],
                xerr=params_histogram[1:-1,3],
                yerr=params_histogram[1:-1,4],
                fmt="none",
                color=cmap((i - 2) / depth.maxClusterWidth),
                elinewidth=0.5,
                capsize=1,
            )
            allXValues = allXValues + list(params_histogram[1:-1,0])
            allYValues = allYValues + list(params_histogram[1:-1,1])
            allYValueErrors = allYValueErrors + list(params_histogram[1:-1,4])
    allXValues = np.array(allXValues)
    allYValues = np.array(allYValues)
    allYValueErrors = np.array(allYValueErrors)
    x = np.linspace(0.00, 0.6, 100)
    def func(x,m,c):
        return m*x+c
    popt,pcov = scipy.optimize.curve_fit(func,allXValues[allXValues>0.14],allYValues[allXValues>0.14],sigma=allYValueErrors[allXValues>0.14]/allYValues[allXValues>0.14],absolute_sigma=False)
    y = func(x,*popt)
    print(popt)
    axs.plot(
        x,
        y,
        linestyle="dashed",
        color=plot.colorPalette[3],
        label="Expected Width",
    )
    if measuredAttribute == "Hit_Voltage":
        plot.set_config(
            axs,
            ylim=(0, 0.2),
            xlim=(0, 0.6),
            title="Voltage MPV vs Width",
            xlabel="MPV",
            ylabel="Width",
            legend=True
        )
    elif measuredAttribute == "ToT":
        plot.set_config(
            axs,
            ylim=(0, None),
            xlim=(0, None),
            title="ToT MPV vs Width",
            xlabel="MPV",
            ylabel="Width",        )
    if saveToPDF:
        plot.saveToPDF(
            f"VoltageDepth/{"Voltage" if measuredAttribute == "Hit_Voltage" else "ToT"}LandauMPVWidthScatter"
            + f"{f"_nnlf" if fitting == "nnlf" else ""}"
        )
    else:
        return plot.fig


if __name__ == "__main__":
    print_mem_usage()
    import AtlasDataAnalysis.Code.dataAnalysis.configLoader as configLoader

    config = configLoader.loadConfig()
    #config["filterDict"] = {"telescope":"kit","fileName":["angle6_6Gev_kit_4","angle6_6Gev_kitHV30_kit_5","angle6_6Gev_kitHV20_kit_6"]}
    #config["filterDict"] = {"telescope":"kit","fileName":["angle6_6Gev_kit_4","angle6_4Gev_kit_2"]}
    #config["maxClusterWidth"] = 40
    dataFiles = initDataFiles(config)
    for dataFile in dataFiles:
        depth = depthAnalysis(
            config["pathToCalcData"],
            maxLine=config["maxLine"],
            maxClusterWidth=config["maxClusterWidth"],
            layers=config["layers"],
            excludeCrossTalk=config["excludeCrossTalk"],
        )
        """
        AngleDistribution(dataFile, depth, config["pathToOutput"])
        WidthDistribution(dataFile, depth, config["pathToOutput"])
        AngleDistribution_2(dataFile, depth, config["pathToOutput"])
        ColumnWidthDistribution(dataFile, config["pathToOutput"], layer=4)
        RowWidthDistribution(dataFile, config["pathToOutput"], layer=4)
        ColumnWidthDistribution(dataFile, config["pathToOutput"], layer=1)
        RowWidthDistribution(dataFile, config["pathToOutput"], layer=1)
        ClustersCountOverTime(dataFile, config["pathToOutput"])
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=False,
            hideLowWidths=False,
            measuredAttribute="ToT",
        )
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=True,
            hideLowWidths=True,
            measuredAttribute="ToT",
        )
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=False,
            hideLowWidths=True,
            measuredAttribute="ToT",
        )
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=True,
            hideLowWidths=False,
            measuredAttribute="ToT",
        )
        
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=False,
            hideLowWidths=False,
        )
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=True,
            hideLowWidths=True,
        )
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=False,
            hideLowWidths=True,
        )
        VoltageDepthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            annotate=False,
            depthCorrection=True,
            hideLowWidths=False,
        )
        """
        """
        LandauMPVWidthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            measuredAttribute = "Hit_Voltage",
            )
        
        LandauMPVWidthScatter(
            dataFile,
            depth,
            config["pathToOutput"],
            measuredAttribute = "ToT",
            _range=(0,256)
            )
        HitDistributionInClusterAllOnOne(dataFile,depth,config["pathToOutput"],vmin=2,vmax=config["maxClusterWidth"])
        if config["maxClusterWidth"] > 15:
            HitDistributionInClusterAllOnOne(dataFile,depth,config["pathToOutput"],vmin=2,vmax=15)
            if config["maxClusterWidth"] > 30:
                HitDistributionInClusterAllOnOne(dataFile,depth,config["pathToOutput"],vmin=16,vmax=30)
                HitDistributionInClusterAllOnOne(dataFile,depth,config["pathToOutput"],vmin=31,vmax=config["maxClusterWidth"])
            else:
                HitDistributionInClusterAllOnOne(dataFile,depth,config["pathToOutput"],vmin=16,vmax=config["maxClusterWidth"])
        else:
            HitDistributionInClusterAllOnOne(dataFile,depth,config["pathToOutput"],vmin=2,vmax=config["maxClusterWidth"])
        """
        #CuttingComparison(dataFile,config["pathToOutput"],layers=config["layers"])
        RowRowCorrelation(dataFile,config["pathToOutput"],config["pathToOutput"],layers=config["layers"] ,excludeCrossTalk=False,maxLine=config["maxLine"])
        RowRowCorrelation(dataFile,config["pathToOutput"],config["pathToOutput"],layers=config["layers"] ,excludeCrossTalk=True,maxLine=config["maxLine"])
        #iList = [3, 5, 8, 11, 13, 15, 18, 20, 22, 24, 25, 27]
        #iList = [30,35,38,45]
        #for i in iList:
            #HitDistributionInCluster(dataFile,depth,i,config["pathToOutput"])
            #Hit_VoltageDistributionByPixel(dataFile,depth,i,config["pathToOutput"],measuredAttribute = "ToT",_range=(40, 256))
            #Hit_VoltageDistributionByPixel(dataFile,depth,i,config["pathToOutput"],_range=(0.162, 2))
