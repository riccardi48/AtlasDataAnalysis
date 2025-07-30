from plotAnalysis import depthAnalysis, plotClass, fit_dataFile
from dataAnalysis import dataAnalysis, initDataFiles
from lowLevelFunctions import (
    chargeCollectionEfficiencyFunc,
    depletionWidthFunc,
)
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Optional, Union
import numpy.typing as npt
import pandas as pd

def Comparison_RowWidthDistribution(
    dataFiles: list[dataAnalysis],
    pathToOutput: str,
    layer: int = 4,
    name: str = "",
    minTimes: Optional[npt.NDArray[np.float64]] = None,
    maxTimes: Optional[npt.NDArray[np.float64]] = None,
    minTime: float = 400000,
    maxTime: float = 600000,
    excludeCrossTalk: bool = True,
    saveToPDF: bool = True,
):
    plot = plotClass(pathToOutput + f"Shared/")
    axs = plot.axs
    bins = 60
    range = (0, 60)
    minTime = 400000
    maxTime = 600000
    for i, dataFile in enumerate(dataFiles):
        rowsWidths, _ = dataFile.get_cluster_attr(
            "RowWidths", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
        times, _ = dataFile.get_cluster_attr(
            "Times", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
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
    pathToOutput: str,
    layer: Optional[int] = None,
    name: str = "",
    saveToPDF: bool = True,
    returnFirstPeaks: bool = False,
) -> tuple[npt.NDArray[np.int_], object]:
    plot = plotClass(pathToOutput + f"Shared/")
    axs = plot.axs
    firstPeaks = []
    for i, dataFile in enumerate(dataFiles):
        times = dataFile.get_cluster_attr("Times", layer=layer, excludeCrossTalk=True)[0] / 1000
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
    return np.array(firstPeaks, dtype=int), plot.fig


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
        rowWidths, _ = dataFile.get_cluster_attr(
            "RowWidths", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
        columnWidths, _ = dataFile.get_cluster_attr(
            "ColumnWidths", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
        times, _ = dataFile.get_cluster_attr(
            "Times", layer=layer, excludeCrossTalk=excludeCrossTalk
        )
        if minTimes is not None and maxTimes is not None:
            minTime = minTimes[i]
            maxTime = maxTimes[i]
        rowWidths = rowWidths[(times > minTime) & (times < maxTime) & (columnWidths <= 2)]
        x, heights = np.unique(rowWidths[rowWidths < maxClusterWidth], return_counts=True)
        bins = np.append(np.arctan((x[0] - 0.5) / d), np.arctan((x + 0.5) / d))
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
    cmap = plt.get_cmap("plasma")
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
            GeV=4 if dataFile.get_fileName() == "angle6_4Gev_kit_2" else 6,
        )
        x = np.linspace(0, np.max(x), 1000)
        y = chargeCollectionEfficiencyFunc(x, *popt,GeV=4 if dataFile.get_fileName() == "angle6_4Gev_kit_2" else 6)
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
        color: Union[tuple[float, ...], str] = tuple(
            map(float, cmap((1 / 48 * dataFile.get_voltage())))
        )
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
            ylim=(-0.1, 0.4),
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
    cmap = plt.get_cmap("plasma")
    depth = depthAnalysis(
        pathToCalcData,
        maxLine=None,
        maxClusterWidth=maxClusterWidth,
        layers=[layer],
        excludeCrossTalk=excludeCrossTalk,
    )
    t_epi_list: list[float] = []
    t_epi_e_list: list[float] = []
    V_list: list[float] = []
    for dataFile in dataFiles:
        popt, pcov, x, y, yerr = fit_dataFile(
            dataFile,
            depth,
            depthCorrection=depthCorrection,
            hideLowWidths=hideLowWidths,
            fitting=fitting,
            measuredAttribute=measuredAttribute,
            GeV=4 if dataFile.get_fileName() == "angle6_4Gev_kit_2" else 6,
        )
        x = np.linspace(0, np.max(x), 1000)
        y = chargeCollectionEfficiencyFunc(x, *popt,GeV=4 if dataFile.get_fileName() == "angle6_4Gev_kit_2" else 6)
        (V_0, t_epi, edl) = popt
        (V_0_e, t_epi_e, edl_e) = np.sqrt(np.diag(pcov))

        color: Union[tuple[float, ...], str] = tuple(
            map(float, cmap((1 / 48 * dataFile.get_voltage())))
        )
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
        t_epi_list.append(t_epi)  # Convert to m
        t_epi_e_list.append(t_epi_e) # Convert to m
        V_list.append(float(dataFile.get_voltage()))
    t_epi_np = np.array(t_epi_list, dtype=float)
    t_epi_e_np = np.array(t_epi_e_list, dtype=float)
    V_np = np.array(V_list, dtype=float)
    initial_guess = [30, 1]
    unzippedBounds = [(0, np.inf), (0, np.inf)]
    lower_bounds, upper_bounds = zip(*unzippedBounds)
    bounds = (list(lower_bounds), list(upper_bounds))
    print(V_np)
    print(t_epi_np)
    print(t_epi_e_np)
    value_list = []
    for c in np.linspace(0, 10, 101):
        func = lambda V, a, b: depletionWidthFunc(V, a, b, c = c)
        popt, pcov = scipy.optimize.curve_fit(
            func,
            V_np,
            t_epi_np,
            p0=initial_guess,
            bounds=bounds,
            sigma=t_epi_e_np / t_epi_np,
            absolute_sigma=False,
            maxfev=10000000,
        )
        (a, b) = popt
        (a_e, b_e) = np.sqrt(np.diag(pcov))
        #print(f"a: {a:.5f} ± {a_e:.5f} µm/√V, ∆V_bi: {b:.5f} ± {b_e:.5f} V, c: {c:.5f} ± 0.0001 µm")
        a = a / 10000  # Convert to cm
        a_e = a_e / 10000  # Convert to cm
        epsilon = 1.034 * 10**(-12)
        q = 1.602 * 10**(-19)
        Na = (2*epsilon)/(q*(a**2))
        Na_e = (4*epsilon)/(q*(a**3)) * a_e
        n = 1.5 * 10**10
        kT_q = 0.0259
        Nd = ((n**2) * np.exp(b/kT_q) )/ Na

        dND_da = Nd / Na * (4 * epsilon / (q * a**3))
        dND_db = (kT_q) * Nd
        Nd_e = np.sqrt((dND_da * a_e)**2 + (dND_db * b_e)**2)
        #print(f"Na: {Na:.2e} cm^-3, Nd: {Nd:.2e} cm^-3")
        value_list.append((a*10000,a_e*10000, b,b_e, c,Na , Na_e/Na,Nd , Nd_e/Nd))
    value_list = np.array(value_list, dtype=float)
    df = pd.DataFrame(
        value_list,
        columns=[
            "a [µm/√V]",
            "a_e [µm/√V]",
            "∆V_bi [V]",
            "∆V_bi_e [V]",
            "c [µm]",
            "Na [cm^-3]",
            "Na_e [cm^-3]",
            "Nd [cm^-3]",
            "Nd_e [cm^-3]"
        ],
        index=[f"c = {c:.2f}" for c in np.linspace(0, 10, 101)],
        )
    print(df[(df["Nd [cm^-3]"] > df["Na [cm^-3]"]) & (df["Nd [cm^-3]"] < 1e21)])
    c = 3.5
    c_e = 0
    func = lambda V, a, b: depletionWidthFunc(V, a, b, c = c)
    popt, pcov = scipy.optimize.curve_fit(
        func,
        V_np,
        t_epi_np,
        p0=initial_guess,
        bounds=bounds,
        sigma=t_epi_e_np / t_epi_np,
        absolute_sigma=False,
        maxfev=10000000,
    )
    (a, b) = popt
    (a_e, b_e) = np.sqrt(np.diag(pcov))

    x = np.linspace(0, 50, 1000)
    y = depletionWidthFunc(x, a, b ,c)
    axs.plot(
        x,
        y,
        color=plot.colorPalette[0],
        linestyle="dashed",
        label=f"a : {a:.5f} ± {a_e:.5f} µm/√V\n∆V_bi : {b:.5f} ± {b_e:.5f} V\nc : {c:.5f} ± {c_e:.5f} µm",
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

def VoltageCalibrationHistograms(pathToOutput: str, layer: int = 4, saveToPDF: bool = True):
    plot = plotClass(pathToOutput + "Shared/", shape=(2, 2), sizePerPlot=(5, 4))
    axs = plot.axs
    calibrationArray = np.load(f"/home/atlas/rballard/Code/tot_calibration/data_{layer}.npy")
    u_0 = calibrationArray[: , :, 0].flatten()
    a = calibrationArray[: , :, 1].flatten()
    b = calibrationArray[: , :, 2].flatten()
    c = calibrationArray[: , :, 3].flatten()
    axs[0, 0].hist(u_0, bins=50, color=plot.colorPalette[0])
    plot.set_config(
        axs[0, 0],
        title="u_0",
        xlabel="u_0 [V]",
        ylabel="Frequency",
    )
    axs[1, 0].hist(a, bins=50, color=plot.colorPalette[0])
    plot.set_config(
        axs[1, 0],
        title="a",
        xlabel="a [V/TS]",
        ylabel="Frequency",
    )
    axs[0, 1].hist(b, bins=50, color=plot.colorPalette[0])
    plot.set_config(
        axs[0, 1],
        title="b",
        xlabel="b [V]",
        ylabel="Frequency",
    )
    axs[1, 1].hist(c, bins=50, color=plot.colorPalette[0])
    plot.set_config(
        axs[1, 1],
        title="c",
        xlabel="c [V]",
        ylabel="Frequency",
    )
    if saveToPDF:
        plot.saveToPDF(
            f"VoltageCalibrationHistograms_{layer}"
        )
    else:
        return plot.fig
    




    
if __name__ == "__main__":
    import configLoader

    config = configLoader.loadConfig()
    config["maxClusterWidth"] = 29
    dataFiles = initDataFiles(config)
    firstPeaks, _ = Comparison_ClustersCountOverTime(
        dataFiles[:8], config["pathToOutput"], layer=4, name="_kit", returnFirstPeaks=True
    )
    Comparison_RowWidthDistribution(
        dataFiles[:8],
        config["pathToOutput"],
        layer=4,
        name="_kit",
        minTimes=firstPeaks.astype(float),
        maxTimes=firstPeaks.astype(float) + 200000.0,
        excludeCrossTalk=True,
    )
    Comparison_AngleDistribution(
        dataFiles[:8],
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=config["maxClusterWidth"],
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
        maxClusterWidth=config["maxClusterWidth"],
        layer=4,
        name="_kit_tight",
        minTimes=firstPeaks,
        maxTimes=firstPeaks + 200000,
        excludeCrossTalk=True,
        xlim=(82, 90),
    )
    Comparison_CCE_Vs_Depth(
        dataFiles, config["pathToOutput"], config["pathToCalcData"], maxClusterWidth=config["maxClusterWidth"]
    )
    Comparison_CCE_Vs_Depth(
        dataFiles,
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=config["maxClusterWidth"],
        measuredAttribute="ToT",
    )
    Scatter_Epi_Thickness_Vs_Bias_Voltage(
        dataFiles,
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=config["maxClusterWidth"],
        measuredAttribute="Hit_Voltage",
    )
    Scatter_Epi_Thickness_Vs_Bias_Voltage(
        dataFiles,
        config["pathToOutput"],
        config["pathToCalcData"],
        maxClusterWidth=config["maxClusterWidth"],
        measuredAttribute="ToT",
    )
    VoltageCalibrationHistograms(config["pathToOutput"], layer=4)