from plotCluster import plotCluster, clusterPlotter
from orthClusterCharge import getOrthClusterCharge
#from funcs import angle_with_error_mc,isTrack,characterizeCluster,isTypes
import sys
sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import numpy as np
from scipy.stats import linregress
from landau import landau
from matplotlib.ticker import MultipleLocator
from plotAnalysis import plotClass
from landau import landau
from scipy.optimize import curve_fit

def landauFunc(
    x,
    x_mpv,
    xi,
    scaler,
    #threshold: float = 0.16,
):
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    return y

def landauCDFFunc(x,
    x_mpv,
    xi,
    #threshold: float = 0.16,
):
    y = landau.cdf(x, x_mpv, xi)
    y = np.reshape(y, np.size(y))
    return y


def landauBinned(x, x_mpv, xi, scaler, edges):
    return (landauCDFFunc(edges[1:],x_mpv,xi)-landauCDFFunc(edges[:-1],x_mpv,xi)) * scaler


def histogramHit_Voltage(
    values,
    _range = (0.162, 2),
    points_per_bin = 100,
    min_bin_width = None,
    max_bin_width = None,
    ):
    values = values[(values > _range[0]) & (values < _range[1])]
    values = np.sort(values)
    if min_bin_width is None:
        min_bin_width = (values.max() - values.min()) / 150
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
    hist = hist / (binEdges[1:] - binEdges[:-1])
    return hist, binEdges, binCentres

def isFlat(cluster):
    return np.unique(cluster.getColumns(True)).size == 1

config = configLoader.loadConfig()
# config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)
_range = (0.162, 4)
mpvPlot = plotClass(f"{config["pathToOutput"]}ClusterTracks/Collected/")
k=0
for dataFile in dataFiles:
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Collected0/"
    clusters = dataFile.get_clusters(excludeCrossTalk=True, layer=4)
    dataFile.init_cluster_voltages()
    width = 30
    landauAreaList = []
    CountsList = []
    paramsList = []
    chargeList = []
    plot = plotClass(
        base_path,
        shape=(1, width),
        sharex=True,
        sizePerPlot=(10, 2),
        hspace=0,
    )
    axs = plot.axs
    histLists = [[] for _ in range(width)]
    lengthInDW = 820
    rowPitch = 50
    possibleRows = int(np.ceil(lengthInDW/50))+1
    for cluster in clusters:
        if not isFlat(cluster):
            continue
        relativeRows = abs(cluster.getRows(True) - np.max(cluster.getRows(True)))
        Timestamps = cluster.getTSs(True)
        relativeTS = Timestamps - np.min(Timestamps)
        voltage = cluster.getHit_Voltages(excludeCrossTalk=True)
        if np.ptp(relativeRows) < possibleRows-4:
            continue
        sortIndexes = np.argsort(relativeRows)
        relativeRows = relativeRows[sortIndexes]
        relativeTS = relativeTS[sortIndexes]
        voltage = voltage[sortIndexes]
        gaps = np.diff(relativeRows)
        if np.any(gaps>5):
            gap = np.where(gaps>5)[0][0] + 1
            if gap > relativeRows.size/2:
                relativeTS = relativeTS[:gap]
                relativeRows = relativeRows[:gap]
                voltage = voltage[:gap]
            else:
                relativeTS = relativeTS[gap:]
                relativeRows = relativeRows[gap:]
                voltage = voltage[gap:]
        if np.all(relativeTS[-5:-1]<=2) and not np.all(relativeTS[1:5]<=2):
            relativeTS = np.flip(relativeTS)
            relativeRows = np.flip(relativeRows)
            voltage = np.flip(voltage)
        if np.ptp(relativeRows) < possibleRows-2:
            continue
        for i in relativeRows:
            if i >= width:
                break
            if not np.isnan(voltage[relativeRows == i][0]):
                histLists[i].append(voltage[relativeRows == i][0])
        print(f"{cluster.getIndex()}", end="\r")
    for i in range(width):
        values = np.array(histLists[i])
        if values[(values >= _range[0]) & (values <= _range[1])].size == 0:
            plot.set_config(
                axs[i],
                ylim=(0, None),
                xlim=(float(np.min(binEdges)), float(np.max(binEdges))),
                legend=True,
            )
            continue
        values = values[np.invert(np.isnan(values))]
        hist, binEdges, binCentres = histogramHit_Voltage(values, _range=_range)
        x = np.linspace(0, np.max(binEdges) + 0.1, 1000)
        fitCutOff = 51
        if i < fitCutOff:
            func = lambda x, x_mpv, x_xi, scaler: landauBinned(x, x_mpv, x_xi, scaler, binEdges)
            p0 = [0.4,0.1,np.sum(hist*(binEdges[1:]-binEdges[:-1]))]
            popt, pcov = curve_fit(
                    func,
                    binCentres,
                    hist * (binEdges[1:] - binEdges[:-1]),
                    maxfev = 12000,
                    p0=p0,
            )
        else:
            ratio = (np.array(paramsList)[1:fitCutOff,1]/np.array(paramsList)[1:fitCutOff,0]).mean()
            func = lambda x, x_mpv, scaler: landauBinned(x, x_mpv, x_mpv*ratio, scaler, binEdges)
            p0 = [0.4,np.sum(hist*(binEdges[1:]-binEdges[:-1]))]
            popt, pcov = curve_fit(
                    func,
                    binCentres,
                    hist * (binEdges[1:] - binEdges[:-1]),
                    maxfev = 8000,
                    p0=p0,
            )
            popt = [popt[0], popt[0]*ratio, popt[1]]
            pcov = np.diag([pcov[0,0], (popt[1]*np.sqrt((pcov[0,0]/popt[0])**2)), pcov[1,1]])
        x_mpv, xi, scale = popt
        x_mpv_e, xi_e, scale_e = np.sqrt(np.diag(pcov))
        print(f"Pixel {i}: Mpv={x_mpv:.5f} ± {x_mpv_e:.5f}, Width={xi:.5f} ± {xi_e:.5f}, Scale={scale:.5f} ± {scale_e:.5f}")
        y = landauFunc(x, x_mpv, xi, scale)
        axs[i].stairs(
            hist,
            binEdges,
            color=plot.colorPalette[3],
            baseline=None,
            label=f"Pixel {i} Count: {np.sum(hist*(binEdges[1:]-binEdges[:-1])):.0f}",# Landau Area: {landau.cdf(_range[1], x_mpv, xi) * scale:.0f}",
        )
        CountsList.append(np.sum(hist*(binEdges[1:]-binEdges[:-1])))
        landauAreaList.append(landau.cdf(_range[1], x_mpv, xi) * scale)
        paramsList.append(list(popt)+list(np.sqrt(np.diag(pcov))))
        chargeList.append(np.sum(hist*(binCentres))/CountsList[0])
        axs[i].plot(
            x,
            y,
            c=plot.colorPalette[0],
            label=f"Mpv:{x_mpv:.3f} ± {x_mpv_e:.3f}\nWidth:{xi:.3f} ± {xi_e:.3f}\nScale:{scale:.3f} ± {scale_e:.3f}",
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
            xlim=(0, float(np.max(binEdges))),
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
    axs.scatter(np.arange(len(landauAreaList)), landauAreaList, color=plot.colorPalette[0], marker="x")
    plot.set_config(
        axs,
        title="Landau Area per Pixel in Cluster",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="Landau Area",
        ylim=(0,None),
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
        ylim=(0,None),
    )
    plot.saveToPDF(f"Counts")
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(np.arange(len(CountsList)), CountsList/np.max(CountsList), color=plot.colorPalette[0], marker="x")
    plot.set_config(
        axs,
        title="Counts per Pixel in Cluster",
        xlabel="Pixel in Cluster %",
        ylabel="Counts",
        ylim=(0,None),
    )
    plot.saveToPDF(f"Counts_Percent")
    paramsList = np.array(paramsList)
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    x = np.arange(len(paramsList[:,0]))
    #x = (np.arange(len(paramsList[:,0]))-0.5)*np.cos(np.deg2rad(dataFile.angle))*50
    axs.scatter(x, paramsList[:,0], color=plot.colorPalette[0], marker="x")
    axs.errorbar(
            x,
            paramsList[:,0],
            yerr=paramsList[:,3],
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
    plot.set_config(
        axs,
        title="MPV per Pixel in Cluster",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="MPV",
        ylim=(0,paramsList[:,0].max()*1.1),
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(0.05))
    axs.yaxis.set_major_formatter("{x:.2f}")
    axs.yaxis.set_minor_locator(MultipleLocator(0.01))
    axs.grid(True)
    plot.saveToPDF(f"MPV")
    if k < 7:
        mpvPlot.axs.scatter(x, paramsList[:,0], color=plot.colorPalette[k], marker="x",label=f"{dataFile.fileName}")
        mpvPlot.axs.errorbar(
                x,
                paramsList[:,0],
                yerr=paramsList[:,3],
                fmt="none",
                color=plot.colorPalette[k],
                elinewidth=1,
                capsize=3,
            )
    else:
        break
    k+=1
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(np.arange(len(chargeList)), chargeList, color=plot.colorPalette[0], marker="x")
    plot.set_config(
        axs,
        title="Charge per Pixel in Cluster",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="Charge",
        ylim=(0,None),
    )
    plot.saveToPDF(f"Charge")
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    axs.scatter(paramsList[:,0], paramsList[:,1], color=plot.colorPalette[0], marker="x")
    axs.errorbar(
            paramsList[:,0],
            paramsList[:,1],
            xerr=paramsList[:,3],
            yerr=paramsList[:,4],
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
        ylim=(0,paramsList[:,1].max()*1.1),
        xlim=(0,paramsList[:,0].max()*1.1),
    )
    plot.saveToPDF(f"MPV_vs_Width")

    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    n = 100000
    widthCutOff = 26
    clusterCharges = np.array([landau.sample(x_mpv,xi,n) for x_mpv,xi in zip(paramsList[:widthCutOff,0],paramsList[:widthCutOff,1])],dtype=float)
    clusterCharges[clusterCharges < 0.161] = 0
    clusterChargesSummed = np.sum(clusterCharges,axis=0)
    height, x = np.histogram(clusterChargesSummed, bins=500, range=(0, 50))
    binCentres = (x[:-1] + x[1:]) / 2
    popt, pcov = curve_fit(
                landauFunc,
                binCentres,
                height,
                maxfev = 8000,
        )
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
    x = np.linspace(0, np.max(x), 1000)
    y = landauFunc(x, *popt)
    x_mpv, xi, scale = popt
    x_mpv_e, xi_e, scale_e = np.sqrt(np.diag(pcov))
    axs.plot(
            x,
            y,
            c=plot.colorPalette[0],
            label=f"Mpv:{x_mpv:.5f} ± {x_mpv_e:.5f}\nWidth:{xi:.5f} ± {xi_e:.5f}\nScale:{scale:.5f} ± {scale_e:.5f}",
        )
    plot.set_config(
        axs,
        title=f"Simulated Cluster Charge Distribution First {widthCutOff} Pixels",
        xlabel="Charge [V]",
        ylabel="Count",
        ylim=(0,None),
        xlim=(0,None),
        legend=True,
    )
    plot.saveToPDF(f"Sim_Cluster_Charge")
    plot = plotClass(
        base_path,
    )
    axs = plot.axs
    clusterCharges = np.array([landau.sample(x_mpv,xi,n) for x_mpv,xi in zip(paramsList[:width,0],paramsList[:width,1])],dtype=float)
    clusterCharges[clusterCharges < 0.161] = 0
    axs.scatter(np.arange(width), np.sum(clusterCharges>0.161,axis=1), marker="x", color=plot.colorPalette[1])
    plot.set_config(
        axs,
        title=f"Simulated Cluster Charge Counts First {widthCutOff} Pixels",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="Count",
        ylim=(0,None),
        xlim=(0,None),
    )
    plot.saveToPDF(f"Sim_Cluster_Counts")

mpvPlot.set_config(
        mpvPlot.axs,
        title="MPV per Pixel in Cluster",
        xlabel="Pixel in Cluster (0 is seed pixel)",
        ylabel="MPV",
        ylim=(0,0.5),
        legend=True,
    )
mpvPlot.saveToPDF(f"MPV")