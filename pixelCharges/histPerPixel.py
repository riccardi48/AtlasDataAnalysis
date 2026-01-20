from plotCluster import plotCluster, clusterPlotter
from orthClusterCharge import getOrthClusterCharge

# from funcs import angle_with_error_mc,isTrack,characterizeCluster,isTypes
import sys

sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import numpy as np
from scipy.stats import linregress
from landau import landau
from matplotlib.ticker import MultipleLocator
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
from scipy import stats
from perfectCluster.findPerfectCluster import isPerfectCluster
from dataAnalysis._fileReader import calcDataFileManager
from perfectCluster.funcs import convertRowsForFit, convertToRelative
from tqdm import tqdm
import matplotlib.pyplot as plt

def getColor(dataFile):
    cmap = plt.get_cmap("plasma")
    color=cmap(dataFile.voltage/48.6)
    if dataFile.fileName == "angle6_4Gev_kit_2":
        color = "r"
    return color

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),
                frameon=False,)


def landauFunc(
    x,
    x_mpv,
    xi,
    scaler,
    # threshold: float = 0.16,
):
    # x0 = convert_x_mpv_to_x0(x_mpv, xi)
    # y = stats.landau.pdf((x - x0) / xi) * scaler
    y = landau.pdf(x, x_mpv, xi) * scaler
    # y = np.reshape(y, np.size(y))
    return y


def convert_x_mpv_to_x0(x_mpv, xi):
    return x_mpv + 0.22278298 * xi


def landauCDFFunc(
    x,
    x_mpv,
    xi,
    # threshold: float = 0.16,
):
    x0 = convert_x_mpv_to_x0(x_mpv, xi)
    # y = stats.landau.cdf((x - x0) / xi)
    y = landau.cdf(x, x_mpv, xi)
    # y = np.reshape(y, np.size(y))
    return y


def landauBinned(x, x_mpv, xi, scaler, edges):
    return (landauCDFFunc(edges[1:], x_mpv, xi) - landauCDFFunc(edges[:-1], x_mpv, xi)) * scaler


from scipy.stats import norm


def gaussianCDFFunc(x, mu, sig):
    return norm.cdf((x - mu) / sig)


def gaussianBinned(mu, sigma, scaler, edges):
    return (gaussianCDFFunc(edges[1:], mu, sigma) - gaussianCDFFunc(edges[:-1], mu, sigma)) * scaler

def gaussianPDFFunc(x, mu, sig, scaler):
    return norm.pdf((x - mu) / sig) * scaler


def histogramHit_Voltage(
    values,
    _range=(0.162, 2),
    points_per_bin=100,
    min_bin_width=None,
    max_bin_width=None,
):
    values = values[(values > _range[0]) & (values < _range[1])]
    values = np.sort(values)
    if min_bin_width is None:
        min_bin_width = (values.max() - values.min()) / 80
    if max_bin_width is None:
        max_bin_width = (values.max() - values.min()) / 10

    n = len(values)
    n_bins = n // points_per_bin

    if n_bins < 1:
        points_per_bin = 10
        n_bins = n // points_per_bin
        if n_bins < 1:
            points_per_bin = 1
            n_bins = n // points_per_bin
            if n_bins < 1:
                raise ValueError("Too few data points for the desired points per bin.")

    edges = np.interp(np.linspace(0, n, n_bins + 1), np.arange(n), values)

    new_edges = [edges[0]]
    for i in range(1, len(edges)):
        proposed_edge = edges[i]
        last_edge = new_edges[-1]
        width = proposed_edge - last_edge

        if width < min_bin_width:
            continue
        elif width > max_bin_width:
            num_subbins = int(np.ceil(width / max_bin_width))
            sub_edges = np.linspace(last_edge, proposed_edge, num_subbins + 1)[1:]
            new_edges.extend(sub_edges)
        else:
            new_edges.append(proposed_edge)

    final_edges = np.array(new_edges)
    hist, binEdges = np.histogram(values, bins=final_edges)
    binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    return hist, binEdges, binCentres


def histogramHit_Voltage_Errors(
    values,
    valuesErrors,
    _range=(0.162, 2),
    points_per_bin=100,
    min_bin_width=None,
    max_bin_width=None,
):
    valuesErrors = valuesErrors[(values > _range[0]) & (values < _range[1])]
    values = values[(values > _range[0]) & (values < _range[1])]
    sortIndex = np.argsort(values)
    valuesErrors = valuesErrors[sortIndex]
    values = values[sortIndex]
    if min_bin_width is None:
        min_bin_width = (values.max() - values.min()) / 80
    if max_bin_width is None:
        max_bin_width = (values.max() - values.min()) / 10

    n = len(values)
    n_bins = n // points_per_bin

    if n_bins < 1:
        points_per_bin = 10
        n_bins = n // points_per_bin
        if n_bins < 1:
            points_per_bin = 1
            n_bins = n // points_per_bin
            if n_bins < 1:
                raise ValueError("Too few data points for the desired points per bin.")

    edges = np.interp(np.linspace(0, n, n_bins + 1), np.arange(n), values)

    new_edges = [edges[0]]
    for i in range(1, len(edges)):
        proposed_edge = edges[i]
        last_edge = new_edges[-1]
        width = proposed_edge - last_edge

        if width < min_bin_width:
            continue
        elif width > max_bin_width:
            num_subbins = int(np.ceil(width / max_bin_width))
            sub_edges = np.linspace(last_edge, proposed_edge, num_subbins + 1)[1:]
            new_edges.extend(sub_edges)
        else:
            new_edges.append(proposed_edge)
    binEdges = np.array(new_edges)
    #hist, binEdges = np.histogram(values, bins=binEdges)
    hist = np.sum(
            gaussianBinned(
                np.tile([values], (binEdges.size - 1, 1)),
                np.tile([valuesErrors], (binEdges.size - 1, 1)),
                1,
                np.tile(binEdges[:, np.newaxis], (1, values.size)),
            ),
            axis=1,
        )
    binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    return hist, binEdges, binCentres


def isFlat(cluster):
    return np.unique(cluster.getColumns(True)).size == 1


from scipy.optimize import brentq


def getBestFitting(values, valuesErrors, x0, p):
    x_mpvList = []
    xiList = []
    scaleList = []
    x_mpv_eList = []
    xi_eList = []
    scale_eList = []
    for points_per_bin in [100, 200]:
        for _range in ((0.160, 2.32), (0.160, 2), (0.160, 1.732)):
            for min_bin_width in [
                (_range[1] - _range[0]) / 80,
                (_range[1] - _range[0]) / 120,
            ]:
                hist, binEdges, binCentres = histogramHit_Voltage_Errors(
                    values,
                    valuesErrors,
                    _range=_range,
                    points_per_bin=points_per_bin,
                    min_bin_width=min_bin_width,
                )
                if p == 1:
                    func = lambda x, x_mpv, x_xi, scaler: landauBinned(
                        x, x_mpv, x_xi, scaler, binEdges
                    )
                    p0 = [0.4, 0.1, np.sum(hist) * 2]
                    popt, pcov = curve_fit(
                        func,
                        binCentres,
                        hist,
                        maxfev=1200,
                        p0=p0,
                    )
                    x_mpv, xi, scale = popt
                    x_mpv_e, xi_e, scale_e = np.sqrt(np.diag(pcov))
                else:
                    if 1 - p <= landauCDFFunc(x0, x0, 1):
                        bounds = ([x0*1.001, 0], [0.6, np.inf])
                        p0 = [x0 * 2, np.sum(hist) * 2]
                    else:
                        bounds = ([0, 0], [x0*0.999, np.inf])
                        p0 = [x0 / 2, np.sum(hist) * 2]
                    lower = 1e-12
                    upper = 100000
                    """
                    print("****")
                    print(f"p={p}")
                    print(f"x0={x0}")
                    print(f"bounds={bounds}")
                    print(f"p0={p0}")
                    print(f"p0-lower = {landauCDFFunc(x0, p0[0], lower) - (1 - p)}")
                    print(f"p0-upper = {landauCDFFunc(x0, p0[0], upper) - (1 - p)}")
                    print(f"bounds-upper = {landauCDFFunc(x0, bounds[0][0], lower) - (1 - p)}")
                    print(f"bounds-lower = {landauCDFFunc(x0, bounds[1][0], upper) - (1 - p)}")
                    for u in np.linspace(bounds[0][0],bounds[1][0],100):
                        print(f"{u:.4f} : {landauCDFFunc(x0, u, lower) - (1 - p):.4f} - {landauCDFFunc(x0, u, upper) - (1 - p):.4f}")
                    """
                    find_sigma = lambda mpv: brentq(
                        lambda s: landauCDFFunc(x0, mpv, s) - (1 - p),
                        lower,  # lower bound for sigma
                        upper,  # upper bound
                    )
                    func = lambda x, x_mpv, scaler: landauBinned(
                        x, x_mpv, find_sigma(x_mpv), scaler, binEdges
                    )
                    popt, pcov = curve_fit(
                        func,
                        binCentres,
                        hist,
                        maxfev=1200,
                        bounds=bounds,
                        p0=p0,
                    )
                    x_mpv, scale = popt
                    x_mpv_e, scale_e = np.sqrt(np.diag(pcov))
                    xi = find_sigma(x_mpv)
                    xi_e = (
                        abs(find_sigma(x_mpv + x_mpv_e) - xi)
                        + abs(find_sigma(x_mpv - x_mpv_e) - xi)
                    ) / 2
                x_mpvList.append(x_mpv)
                xiList.append(xi)
                scaleList.append(scale)
                x_mpv_eList.append(x_mpv_e)
                xi_eList.append(xi_e)
                scale_eList.append(scale_e)

    e_stat = np.sqrt(np.mean(np.array(x_mpv_eList) ** 2))
    e_bin = np.std(np.array(x_mpvList))
    x_mpv = np.mean(x_mpvList)
    x_mpv_e = np.sqrt(e_stat**2 + e_bin**2)
    e_stat = np.sqrt(np.mean(np.array(xi_eList) ** 2))
    e_bin = np.std(np.array(xiList))
    xi = np.mean(xiList)
    xi_e = np.sqrt(e_stat**2 + e_bin**2)
    e_stat = np.sqrt(np.mean(np.array(scale_eList) ** 2))
    e_bin = np.std(np.array(scaleList))
    scale = np.mean(scaleList)
    scale_e = np.sqrt(e_stat**2 + e_bin**2)
    return x_mpv, xi, scale, x_mpv_e, xi_e, scale_e


config = configLoader.loadConfig()
# config["filterDict"] = {"telescope":"kit","fileName":"angle6_6Gev_kitHV30_kit_5"}
# config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)
_range = (0.160, 2)
mpvPlot = plotClass(f"{config["pathToOutput"]}ClusterTracks/Collected/")
width = 30
minPval = 0.5
for k,dataFile in enumerate(dataFiles):
    dataFile.init_cluster_voltages()
    clusters = dataFile.get_clusters(excludeCrossTalk=True, layer=4)
    relativeRowsList = []
    presentRelativeRowsList = []
    for cluster in tqdm(
        dataFile.get_perfectClusters(minPval=minPval, layer=4, maxRow=25), desc="Finding Efficiency"
    ):
        rows = cluster.getRows(True)[cluster.section]
        expectedRows = np.linspace(
            np.sort(rows)[-1 * cluster.flipped],
            np.sort(rows)[-1 * cluster.flipped] + width + width * 2 * -1 * cluster.flipped,
            width + 1,
        ).astype(int)[:-1]
        expectedRows = expectedRows[(expectedRows >= 0) & (expectedRows < 372)]
        expectedRowsRelative, _ = convertRowsForFit(expectedRows, expectedRows, flipped=False)
        x, _ = convertRowsForFit(rows, rows, flipped=cluster.flipped)
        missing = np.array([r for r in expectedRowsRelative if r not in x])
        relativeRowsList.extend(missing[missing < width])
        presentRelativeRowsList.extend(expectedRowsRelative)
    rowFrequency = np.zeros(width)
    for row in presentRelativeRowsList:
        rowFrequency[row] += 1
    missingRowFrequency = np.zeros(width)
    for row in relativeRowsList:
        missingRowFrequency[row] += 1
    rowPercent = missingRowFrequency / (rowFrequency + 1e-10)
    error = np.sqrt((rowPercent * (1 - rowPercent)) / (rowFrequency + 1e-10))
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Collected0/"
    landauAreaList = []
    CountsList = []
    paramsList = []
    chargeList = []
    chargeErrorList = []
    plot = plotClass(
        base_path,
        shape=(1, width),
        sharex=True,
        sizePerPlot=(10, 2),
        hspace=0,
    )
    axs = plot.axs
    histLists = [[] for _ in range(width)]
    histErrorsLists = [[] for _ in range(width)]
    lengthInDW = 820
    rowPitch = 50
    possibleRows = int(np.ceil(lengthInDW / 50)) + 1
    # calcFileManager = calcDataFileManager(config["pathToCalcData"], "TSParams", config["maxLine"])
    # calcFileName = calcFileManager.generateFileName(
    #    attribute=f"{dataFile.fileName}",
    # )
    # estimate, spread = calcFileManager.loadFile(calcFileName=calcFileName)
    l = 0
    for cluster in dataFile.get_perfectClusters(minPval=minPval, layer=4, maxRow=25):
        rows = cluster.getRows(True)[cluster.section]
        voltage = cluster.getHit_Voltages(True)[cluster.section]
        voltageErrors = cluster.getHit_VoltageErrors(True)[cluster.section]
        # relativeRows = abs(cluster.getRows(True)[cluster.section] - np.max(cluster.getRows(True)[cluster.section]))
        relativeRows, voltage = convertToRelative(rows, voltage, flipped=cluster.flipped)
        _, voltageErrors = convertToRelative(rows, voltageErrors, flipped=cluster.flipped)
        # if np.ptp(relativeRows) < possibleRows-2:
        #    continue

        for i in range(relativeRows.size):
            if relativeRows[i] >= width:
                continue
            if (
                not np.isnan(voltage[i])
                and not np.isinf(voltage[i])
                and not voltage[i] == 0
                and not voltageErrors[i] == 0
                and not np.isnan(voltageErrors[i])
                and not np.isinf(voltageErrors[i])
            ):
                histLists[relativeRows[i]].append(voltage[i])
                histErrorsLists[relativeRows[i]].append(voltageErrors[i])
            else:
                # print(i, voltage[i], voltageErrors[i])
                l += 1
    print(f"Skipped {l} hits due to invalid voltage")
    plotMPV = plotClass(
        base_path,sizePerPlot=(6,4)
    )
    for i in range(width):
        values = np.array(histLists[i])
        valuesErrors = np.array(histErrorsLists[i])
        if values[(values >= _range[0]) & (values <= _range[1])].size == 0:
            plot.set_config(
                axs[i],
                ylim=(0, None),
                xlim=(_range[0], _range[1]),
                legend=True,
            )
            continue
        # hist, binEdges, binCentres = histogramHit_Voltage(values, _range=_range)

        p = 1 - rowPercent[i]
        if p <= 0.05 or len(values) < 100:
            continue
        # print(p)
        # print(landauCDFFunc(0.161, 0.161, 1e-6))
        # print(landauCDFFunc(0.161, 0.161, 1000))
        # print(landauCDFFunc(0.161, 0.01, 1e-6))
        # print(landauCDFFunc(0.161, 0.01, 1000))
        # print(landauCDFFunc(0.161, 2, 1e-6))
        # print(landauCDFFunc(0.161, 2, 1000))
        x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = getBestFitting(
            values, valuesErrors, _range[0], 1
        )
        plotMPV.axs.scatter(
            i, x_mpv, color=plotMPV.colorPalette[0], marker="x", label="Unconstrained"
        )
        plotMPV.axs.errorbar(
            i,
            x_mpv,
            yerr=x_mpv_e,
            fmt="none",
            color=plotMPV.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
        if p <= 0.98:
            x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = getBestFitting(
                values, valuesErrors, _range[0], p
            )
            plotMPV.axs.scatter(
                i, x_mpv, color=plotMPV.colorPalette[2], marker="x", label="Constrained"
            )
            plotMPV.axs.errorbar(
                i,
                x_mpv,
                yerr=x_mpv_e,
                fmt="none",
                color=plotMPV.colorPalette[2],
                elinewidth=1,
                capsize=3,
            )

        hist, binEdges, binCentres = histogramHit_Voltage_Errors(
            values, valuesErrors, _range=_range
        )
        x = np.linspace(0, np.max(binEdges) + 0.1, 1000)
        """
        fitCutOff = 51
        if i < fitCutOff:
            func = lambda x, x_mpv, x_xi, scaler: landauBinned(x, x_mpv, x_xi, scaler, binEdges)
            p0 = [0.4, 0.1, np.sum(hist * (binEdges[1:] - binEdges[:-1]))]
            popt, pcov = curve_fit(
                func,
                binCentres,
                hist * (binEdges[1:] - binEdges[:-1]),
                maxfev=12000,
                p0=p0,
            )
        else:
            ratio = (
                np.array(paramsList)[1:fitCutOff, 1] / np.array(paramsList)[1:fitCutOff, 0]
            ).mean()
            func = lambda x, x_mpv, scaler: landauBinned(x, x_mpv, x_mpv * ratio, scaler, binEdges)
            p0 = [0.4, np.sum(hist * (binEdges[1:] - binEdges[:-1]))]
            popt, pcov = curve_fit(
                func,
                binCentres,
                hist * (binEdges[1:] - binEdges[:-1]),
                maxfev=8000,
                p0=p0,
            )
            popt = [popt[0], popt[0] * ratio, popt[1]]
            pcov = np.diag(
                [pcov[0, 0], (popt[1] * np.sqrt((pcov[0, 0] / popt[0]) ** 2)), pcov[1, 1]]
            )
        
        x_mpv, xi, scale = popt
        x_mpv_e, xi_e, scale_e = np.sqrt(np.diag(pcov))
        """
        print(
            f"Pixel {i}: Mpv={x_mpv:.5f} ± {x_mpv_e:.5f}, Width={xi:.5f} ± {xi_e:.5f}, Scale={scale:.5f} ± {scale_e:.5f}"
        )

        axs[i].stairs(
            hist / (binEdges[1:] - binEdges[:-1]),
            binEdges,
            color=plot.colorPalette[3],
            baseline=None,
            label=f"Pixel {i} Count: {np.sum(hist):.0f}",  # Landau Area: {landau.cdf(_range[1], x_mpv, xi) * scale:.0f}",
        )
        CountsList.append(np.sum(hist))
        landauAreaList.append(landau.cdf(_range[1], x_mpv, xi) * scale)
        paramsList.append(list([x_mpv, xi, scale]) + list([x_mpv_e, xi_e, scale_e]))
        chargeList.append(np.sum(hist * (binCentres)) / CountsList[0])
        chargeErrorList.append(
            np.sqrt(np.sum(valuesErrors[(values >= _range[0]) & (values <= _range[1])] ** 2))
            / CountsList[0]
        )
        y = landauFunc(x, x_mpv, xi, scale)
        axs[i].plot(
            x,
            y,
            c=plot.colorPalette[0],
            label=f"Mpv:{x_mpv:.3f} ± {x_mpv_e:.3f}\nWidth:{xi:.3f} ± {xi_e:.3f}\nScale:{scale:.3f} ± {scale_e:.3f}",
        )
        axs[i].scatter(
            binCentres,
            landauBinned(binCentres, x_mpv, xi, scale, binEdges) / (binEdges[1:] - binEdges[:-1]),
            c=plot.colorPalette[2],
            marker="x",
            label=f"Landau Binned",
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
        axs[i].get_xaxis().set_visible(False)
        plot.set_config(
            axs[i],
            ylim=(0, np.max(y) * 1.1),
            xlim=(0, _range[1]),
            legend=True,
        )
    axs[-1].set_xlabel("Hit Voltage [V]")
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    axs[-1].xaxis.set_major_locator(MultipleLocator(0.2))
    axs[-1].xaxis.set_major_formatter("{x:.1f}")
    axs[-1].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].set_xlabel("Hit Voltage [V]")
    axs[0].get_xaxis().set_visible(True)
    axs[0].xaxis.set_label_position("top")
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].xaxis.set_major_locator(MultipleLocator(0.2))
    axs[0].xaxis.set_major_formatter("{x:.1f}")
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    plot.fig.suptitle(f"{width} Width Cluster Charge Distribution By Pixel")
    plot.saveToPDF(f"Voltages")

    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(
        np.arange(len(landauAreaList)), landauAreaList, color=plot.colorPalette[0], marker="x"
    )
    plot.set_config(
        axs,
        title="Landau Area per Pixel in Cluster",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="Landau Area",
        ylim=(0, None),
    )
    plot.saveToPDF(f"LandauArea")
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(np.arange(len(CountsList)), CountsList, color=plot.colorPalette[0], marker="x")
    plot.set_config(
        axs,
        title="Counts per Pixel in Cluster",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="Counts",
        ylim=(0, None),
    )
    plot.saveToPDF(f"Counts")
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(
        np.arange(len(CountsList)),
        CountsList / np.max(CountsList),
        color=plot.colorPalette[0],
        marker="x",
    )
    plot.set_config(
        axs,
        title="Counts per Pixel in Cluster",
        xlabel="Pixel in Cluster %",
        ylabel="Counts",
        ylim=(0, None),
    )
    plot.saveToPDF(f"Counts_Percent")
    paramsList = np.array(paramsList)

    x = np.arange(len(paramsList[:, 0]))
    # x = (np.arange(len(paramsList[:,0]))-0.5)*np.cos(np.deg2rad(dataFile.angle))*50
    plotMPV.axs.hlines(
        0.161,
        -1,
        paramsList[:, 0].size + 1,
        linestyle="--",
        color=plotMPV.colorPalette[5],
        label="Threshold",
    )
    plotMPV.set_config(
        plotMPV.axs,
        title="MPV per Pixel in Cluster [V]",
        xlabel="Relative Row [Px]",
        ylabel="MPV [V]",
        xlim=(-1, paramsList[:, 0].size + 1),
        ylim=(0, paramsList[:, 0].max() * 1.1),
    )
    plotMPV.axs.xaxis.set_major_locator(MultipleLocator(5))
    plotMPV.axs.xaxis.set_major_formatter("{x:.0f}")
    plotMPV.axs.xaxis.set_minor_locator(MultipleLocator(1))
    plotMPV.axs.yaxis.set_major_locator(MultipleLocator(0.05))
    plotMPV.axs.yaxis.set_major_formatter("{x:.2f}")
    plotMPV.axs.yaxis.set_minor_locator(MultipleLocator(0.01))
    plotMPV.axs.grid(True)
    legend_without_duplicate_labels(plotMPV.axs)
    plotMPV.saveToPDF(f"MPV")
    mpvPlot.axs.plot(
        x,
        paramsList[:, 0],
        color=getColor(dataFile),
        label=f"{dataFile.fileName}",
    )
    mpvPlot.axs.fill_between(
        x,
        paramsList[:, 0]+paramsList[:, 3],
        paramsList[:, 0]+paramsList[:, 3],
        alpha=0.2,
        color=getColor(dataFile),
    )
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(np.arange(len(chargeList)), chargeList, color=plot.colorPalette[0], marker="x")
    axs.errorbar(
        np.arange(len(chargeList)),
        chargeList,
        yerr=chargeErrorList,
        fmt="none",
        color=plot.colorPalette[0],
        elinewidth=1,
        capsize=3,
    )
    plot.set_config(
        axs,
        title="Charge per Pixel in Cluster",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="Charge",
        ylim=(0, None),
    )
    plot.saveToPDF(f"Charge")
    calcFileManager = calcDataFileManager(config["pathToCalcData"], "MPV_Params", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"{dataFile.fileName}",
    )
    calcFileManager.saveFile(calcFileName=calcFileName, array=paramsList)

    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(paramsList[:, 0], paramsList[:, 1], color=plot.colorPalette[0], marker="x")
    axs.errorbar(
        paramsList[:, 0],
        paramsList[:, 1],
        xerr=paramsList[:, 3],
        yerr=paramsList[:, 4],
        fmt="none",
        color=plot.colorPalette[0],
        elinewidth=1,
        capsize=3,
    )
    plot.set_config(
        axs,
        title="Charge per Pixel in Cluster",
        xlabel="MPV",
        ylabel="Width",
        ylim=(0, paramsList[:, 1].max() * 1.1),
        xlim=(0, paramsList[:, 0].max() * 1.1),
    )
    plot.saveToPDF(f"MPV_vs_Width")

    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    n = 100000
    widthCutOff = 26
    clusterCharges = np.array(
        [
            landau.sample(x_mpv, xi, n)
            for x_mpv, xi in zip(paramsList[:widthCutOff, 0], paramsList[:widthCutOff, 1])
        ],
        dtype=float,
    )
    clusterCharges[clusterCharges < 0.161] = 0
    clusterChargesSummed = np.sum(clusterCharges, axis=0)
    height, x = np.histogram(clusterChargesSummed, bins=500, range=(0, 50))
    binCentres = (x[:-1] + x[1:]) / 2
    func = lambda _x,mu,sig,scaler : gaussianBinned(mu,sig,scaler,x)
    popt, pcov = curve_fit(
        func,
        binCentres,
        height*(x[1:] - x[:-1]),
        maxfev=8000,
    )
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    x = np.linspace(0, np.max(x), 1000)
    y = gaussianPDFFunc(x,*popt)
    mu, sig, scale = popt
    mu_e, sig_e, scale_e = np.sqrt(np.diag(pcov))
    axs.plot(
        x,
        y,
        c=plot.colorPalette[0],
        label=f"Peak:{mu:.5f} ± {mu_e:.5f}\nStd:{sig:.5f} ± {sig_e:.5f}\nScale:{scale:.5f} ± {scale_e:.5f}",
    )
    plot.set_config(
        axs,
        title=f"Simulated Cluster Charge Distribution First {widthCutOff} Pixels",
        xlabel="Charge [V]",
        ylabel="Count",
        ylim=(0, None),
        xlim=(0, None),
        legend=True,
    )
    plot.saveToPDF(f"Sim_Cluster_Charge")
    continue
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    clusterCharges = np.array(
        [
            landau.sample(x_mpv, xi, n)
            for x_mpv, xi in zip(paramsList[:width, 0], paramsList[:width, 1])
        ],
        dtype=float,
    )
    clusterCharges[clusterCharges < 0.161] = 0
    axs.scatter(
        np.arange(width),
        np.sum(clusterCharges > 0.161, axis=1),
        marker="x",
        color=plot.colorPalette[1],
    )
    plot.set_config(
        axs,
        title=f"Simulated Cluster Charge Counts First {widthCutOff} Pixels",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="Count",
        ylim=(0, None),
        xlim=(0, None),
    )
    plot.saveToPDF(f"Sim_Cluster_Counts")

mpvPlot.set_config(
    mpvPlot.axs,
    title="MPV per Pixel in Cluster",
    xlabel="Pixel in Cluster (0 is seed pixel)",
    ylabel="MPV",
    ylim=(0, 0.5),
    legend=True,
)
mpvPlot.saveToPDF(f"MPV")
