from plotAnalysis import depthAnalysis, plotClass
from dataAnalysis import dataAnalysis, initDataFiles
from lowLevelFunctions import (
    calcDepth,
    adjustPeakVoltage,
    fitVoltageDepth,
    chargeCollectionEfficiencyFunc,
    depletionWidthFunc,
)
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import scipy



def Comparison_RowWidthDistribution(
    dataFiles: list[dataAnalysis],
    pathToOutput,
    layer=4,
    name="",
    minTimes=None,
    maxTimes=None,
    minTime=400000,
    maxTime=600000,
    excludeCrossTalk=True,
    saveToPDF=True,
):
    plot = plotClass(pathToOutput + f"Shared/")
    axs = plot.axs
    bins = 60
    range = (0, 60)
    minTime = 400000
    maxTime = 600000
    for i, dataFile in enumerate(dataFiles):
        rowsWidths = dataFile.get_cluster_attr(
            "RowWidths", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
        times = dataFile.get_cluster_attr("Times", layer=layer, excludeCrossTalk=excludeCrossTalk)
        # print(f"{np.min(times):.0f}")
        # print(f"{np.max(times):.0f}")
        if minTimes is not None and maxTimes is not None:
            minTime = minTimes[i]
            maxTime = maxTimes[i]
        if maxTime > np.max(times):
            print(dataFile.get_fileName())
        rowsWidths = rowsWidths[(times > minTime) & (times < maxTime)]
        height, x = np.histogram(rowsWidths, bins=bins, range=range)
        axs.stairs(
            height,
            x - 0.5,
            baseline=None,
            color=plot.colorPalette[i],
            label=f"{dataFile.get_fileName()}",
        )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=range,
        title="Row Width Distribution",
        legend=True,
        xlabel="Row Width [px]",
        ylabel="Frequency",
    )
    # axs.set_yscale("log")
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(1000))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(200))
    if saveToPDF:
        plot.saveToPDF(f"Comparison_RowWidthDistribution_{layer}{name}")
    else:
        return plot.fig


def Comparison_ClustersCountOverTime(
    dataFiles: list[dataAnalysis],
    pathToOutput,
    layer: int = None,
    name="",
    saveToPDF=True,
    returnFirstPeaks=False,
):
    plot = plotClass(pathToOutput + f"Shared/")
    axs = plot.axs
    firstPeaks = []
    for i, dataFile in enumerate(dataFiles):
        times = dataFile.get_cluster_attr("Times", layer=layer, excludeCrossTalk=True) / 1000
        maxTime = np.max(times)
        maxTime = 600
        minTime = 0
        range = (minTime, maxTime)
        bins = int(np.ptp(range) / 1)
        height, x = np.histogram(times, bins=bins, range=range)
        axs.stairs(
            height, x, baseline=None, color=plot.colorPalette[i], label=f"{dataFile.get_fileName()}"
        )
        firstPeaks.append(x[np.where(height > 300)][0] * 1000)
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
        plot.saveToPDF(f"Comparison_ClustersCountOverTime_{layer}{name}")
    if returnFirstPeaks:
        return np.array(firstPeaks, dtype=int)
    return plot.fig


def Comparison_AngleDistribution(
    dataFiles: list[dataAnalysis],
    pathToOutput,
    pathToCalcData,
    layer=4,
    name="",
    minTimes=None,
    maxTimes=None,
    minTime=300000,
    maxTime=600000,
    excludeCrossTalk=True,
    xlim=(70, 90),
    maxClusterWidth=40,
):
    plot = plotClass(pathToOutput + f"Shared/")
    axs = plot.axs
    depth = depthAnalysis(
        pathToCalcData,
        maxLine=None,
        maxClusterWidth=maxClusterWidth,
        layers=[layer],
        excludeCrossTalk=excludeCrossTalk,
    )
    for i, dataFile in enumerate(dataFiles):
        d = depth.find_d_value(dataFile)
        rowWidths = dataFile.get_cluster_attr(
            "RowWidths", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
        columnWidths = dataFile.get_cluster_attr(
            "ColumnWidths", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
        times = dataFile.get_cluster_attr("Times", layer=layer, excludeCrossTalk=excludeCrossTalk)
        if minTimes is not None and maxTimes is not None:
            minTime = minTimes[i]
            maxTime = maxTimes[i]
        rowWidths = rowWidths[(times > minTime) & (times < maxTime) & (columnWidths <= 2)]
        x, heights = np.unique(rowWidths[rowWidths < maxClusterWidth], return_counts=True)
        bins = np.append(np.atan((x[0] - 0.5) / d), np.atan((x + 0.5) / d))
        heights = heights / np.rad2deg(np.diff(bins))
        axs.stairs(
            heights,
            np.rad2deg(bins),
            label=f"{d*50:.2f} μm - {dataFile.get_fileName()}",
            baseline=None,
            color=plot.colorPalette[i],
        )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=xlim,
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
    plot.saveToPDF(f"Comparison_AngleDistribution_{layer}{name}")


def fit_dataFile(
    dataFile: dataAnalysis,
    depth: depthAnalysis,
    depthCorrection=True,
    hideLowWidths=True,
    fitting="histogram",
    measuredAttribute="Hit_Voltage",
):
    d = depth.find_d_value(dataFile)
    allXValues = []
    allYValues = []
    allYValuesErrors = []
    for i in range(2, depth.maxClusterWidth + 1):
        x = calcDepth(
            d,
            i,
            dataFile.get_angle(),
            depthCorrection=depthCorrection,
            upTwo=True if dataFile.get_fileName() == "angle6_4Gev_kit_2" else False,
        )
        y, y_err = depth.findPeak(dataFile, i, fitting=fitting, measuredAttribute=measuredAttribute)
        if i < 10:
            y, y_err = adjustPeakVoltage(y, y_err, d, i)
        if not hideLowWidths or (
            np.rad2deg(np.atan(i / d)) > 85 and np.rad2deg(np.atan(i / d)) < 87
        ):
            allXValues = allXValues + list(x[1:-1])
            allYValues = allYValues + list(y[1:-1])
            allYValuesErrors = allYValues + list(y_err[1:-1])
    allXValues = np.array(allXValues)
    allYValues = np.array(allYValues)
    allYValuesErrors = np.array(allYValuesErrors)
    y = allYValues[np.argsort(allXValues)]
    yerr = allYValuesErrors[np.argsort(allXValues)]
    x = allXValues[np.argsort(allXValues)]
    popt, pcov = fitVoltageDepth(x[x < d * 50], y[x < d * 50], yerr[x < d * 50])
    return popt, pcov, x[x < d * 50], y[x < d * 50], yerr[x < d * 50]


def Comparison_CCE_Vs_Depth(
    dataFiles: list[dataAnalysis],
    pathToOutput,
    pathToCalcData,
    depthCorrection=True,
    hideLowWidths=True,
    fitting="histogram",
    maxClusterWidth=30,
    layer=4,
    excludeCrossTalk=True,
    name="",
    measuredAttribute="Hit_Voltage",
    saveToPDF=True,
):
    plot = plotClass(pathToOutput + f"Shared/")
    axs = plot.axs
    cmap = plt.colormaps["plasma"]
    depth = depthAnalysis(
        pathToCalcData,
        maxLine=None,
        maxClusterWidth=maxClusterWidth,
        layers=[layer],
        excludeCrossTalk=excludeCrossTalk,
    )
    cellText = []
    if measuredAttribute == "Hit_Voltage":
        cellColumns = ["Base Charge [V]", "Depletion Depth [μm]", "Diffusion length [μm]"]
    elif measuredAttribute == "ToT":
        cellColumns = ["Base ToT [V]", "Depletion Depth [μm]", "Diffusion length [μm]"]
    cellRows = []
    for dataFile in dataFiles:
        popt, pcov, x, y, yerr = fit_dataFile(
            dataFile,
            depth,
            depthCorrection=depthCorrection,
            hideLowWidths=hideLowWidths,
            fitting=fitting,
            measuredAttribute=measuredAttribute,
        )
        x = np.linspace(0, np.max(x), 1000)
        y = chargeCollectionEfficiencyFunc(x, *popt)
        (V_0, t_epi, edl) = popt
        (V_0_e, t_epi_e, edl_e) = np.sqrt(np.diag(pcov))
        cellText.append(
            [f"{V_0:.3f}±{V_0_e:.3f}", f"{t_epi:.1f}±{t_epi_e:.2f}", f"{edl:.1f}±{edl_e:.2f}"]
        )
        label = f"{dataFile.get_voltage()}V"
        if dataFile.get_voltage() == 48:
            if dataFile.get_fileName() == "angle6_4Gev_kit_2":
                label = f"{label} 4Gev"
            else:
                label = f"{label} 6Gev"
        cellRows.append(label)
        color = cmap((1 / 48 * dataFile.get_voltage()))
        if dataFile.get_fileName() == "angle6_4Gev_kit_2":
            color = "r"
        axs.plot(x, y, linestyle="dashed", label=label, color=color)
    plot.set_config(axs, legend=True, loc="upper right", ncols=2)
    axs.xaxis.set_major_locator(MultipleLocator(10))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(5))
    if measuredAttribute == "Hit_Voltage":
        plot.set_config(
            axs,
            ylim=(-0.3, 1),
            xlim=(0, None),
            title="Voltage change withing a Cluster",
            xlabel="Depth [μm]",
            ylabel="Voltage [V]",
        )
        axs.yaxis.set_major_locator(MultipleLocator(0.2))
        axs.yaxis.set_major_formatter("{x:.2f}")
        axs.yaxis.set_minor_locator(MultipleLocator(0.05))
    elif measuredAttribute == "ToT":
        plot.set_config(
            axs,
            ylim=(-20, 120),
            xlim=(0, None),
            title="ToT withing a Cluster",
            xlabel="Depth [μm]",
            ylabel="ToT [TS]",
        )
        axs.set_ylim(0, 120)
        axs.yaxis.set_major_locator(MultipleLocator(10))
        axs.yaxis.set_major_formatter("{x:.2f}")
        axs.yaxis.set_minor_locator(MultipleLocator(2))
    axs.table(
        cellText=cellText,
        rowLabels=cellRows,
        colLabels=cellColumns,
        bbox=[0.2, 0, 0.8, 0.5],
        edges="horizontal",
    )
    if saveToPDF:
        plot.saveToPDF(
            f"Comparison_{"Hit_Voltage" if measuredAttribute == "Hit_Voltage" else "ToT"}_Vs_Depth_{layer}{name}"
        )
    else:
        return plot.fig


def Scatter_Epi_Thickness_Vs_Bias_Voltage(
    dataFiles: list[dataAnalysis],
    pathToOutput,
    pathToCalcData,
    depthCorrection=True,
    hideLowWidths=True,
    fitting="histogram",
    maxClusterWidth=30,
    layer=4,
    excludeCrossTalk=True,
    name="",
    measuredAttribute="Hit_Voltage",
    saveToPDF=True,
):
    plot = plotClass(pathToOutput + f"Shared/")
    axs = plot.axs
    cmap = plt.colormaps["plasma"]
    depth = depthAnalysis(
        pathToCalcData,
        maxLine=None,
        maxClusterWidth=maxClusterWidth,
        layers=[layer],
        excludeCrossTalk=excludeCrossTalk,
    )
    t_epi_list = []
    t_epi_e_list = []
    V_list = []
    for dataFile in dataFiles:
        popt, pcov, x, y, yerr = fit_dataFile(
            dataFile,
            depth,
            depthCorrection=depthCorrection,
            hideLowWidths=hideLowWidths,
            fitting=fitting,
            measuredAttribute=measuredAttribute,
        )
        x = np.linspace(0, np.max(x), 1000)
        y = chargeCollectionEfficiencyFunc(x, *popt)
        (V_0, t_epi, edl) = popt
        (V_0_e, t_epi_e, edl_e) = np.sqrt(np.diag(pcov))
        color = cmap((1 / 48 * dataFile.get_voltage()))
        if dataFile.get_fileName() == "angle6_4Gev_kit_2":
            color = "r"
        axs.scatter(dataFile.get_voltage(), t_epi, color=color, marker="x", s=15)
        axs.errorbar(
            dataFile.get_voltage(),
            t_epi,
            yerr=t_epi_e,
            fmt="none",
            color=cmap((1 / 48 * dataFile.get_voltage())),
            elinewidth=0.5,
            capsize=1,
        )
        t_epi_list.append(t_epi)
        t_epi_e_list.append(t_epi_e)
        V_list.append(dataFile.get_voltage())
    t_epi_list = np.array(t_epi_list)
    t_epi_e_list = np.array(t_epi_e_list)
    V_list = np.array(V_list)
    initial_guess = [30, 1]
    bounds = [(0, np.inf), (0, np.inf)]
    bounds = tuple(zip(*bounds))
    popt, pcov = scipy.optimize.curve_fit(
        depletionWidthFunc,
        V_list,
        t_epi_list,
        p0=initial_guess,
        bounds=bounds,
        sigma=t_epi_e_list / t_epi_list,
        absolute_sigma=False,
        maxfev=10000000,
    )
    (a, b) = popt
    (a_e, b_e) = np.sqrt(np.diag(pcov))
    x = np.linspace(0, 50, 1000)
    y = depletionWidthFunc(x, a, b)
    axs.plot(
        x,
        y,
        color=plot.colorPalette[0],
        linestyle="dashed",
        label=f"a : {a:.5f} ± {a_e:.5f}\n∆V_bi : {b:.5f} ± {b_e:.5f}",
    )
    plot.set_config(
        axs,
        ylim=(0, 45),
        xlim=(-2, 50),
        title="Depletion Depth Vs Bias Voltage",
        xlabel="Bias Voltage [V]",
        ylabel="Depletion Depth [μm]",
        legend=True,
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1))
    if saveToPDF:
        plot.saveToPDF(
            f"Comparison_{"Hit_Voltage" if measuredAttribute == "Hit_Voltage" else "ToT"}_Scatter_Epi_Thickness_Vs_Bias_Voltage_{layer}{name}"
        )
    else:
        return plot.fig


if __name__ == "__main__":
    import configLoader
    config = configLoader.loadConfig()

    dataFiles = initDataFiles(config)
    firstPeaks = Comparison_ClustersCountOverTime(
        dataFiles[:8], config["pathToOutput"], layer=4, name="_kit", returnFirstPeaks=True
    )
    Comparison_RowWidthDistribution(
        dataFiles[:8],
        config["pathToOutput"],
        layer=4,
        name="_kit",
        minTimes=firstPeaks,
        maxTimes=firstPeaks + 200000,
        excludeCrossTalk=True,
    )
    Comparison_AngleDistribution(
        dataFiles[:8],
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=30,
        layer=4,
        name="_kit",
        minTimes=firstPeaks,
        maxTimes=firstPeaks + 200000,
        excludeCrossTalk=True,
        xlim=(0, 90),
    )
    Comparison_AngleDistribution(
        dataFiles[:8],
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=30,
        layer=4,
        name="_kit_tight",
        minTimes=firstPeaks,
        maxTimes=firstPeaks + 200000,
        excludeCrossTalk=True,
        xlim=(82, 90),
    )
    Comparison_CCE_Vs_Depth(
        dataFiles, config["pathToOutput"], config["pathToCalcData"], maxClusterWidth=30
    )
    Comparison_CCE_Vs_Depth(
        dataFiles,
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=30,
        measuredAttribute="ToT",
    )
    Scatter_Epi_Thickness_Vs_Bias_Voltage(
        dataFiles,
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=30,
        measuredAttribute="Hit_Voltage",
    )
    Scatter_Epi_Thickness_Vs_Bias_Voltage(
        dataFiles,
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=30,
        measuredAttribute="ToT",
    )
