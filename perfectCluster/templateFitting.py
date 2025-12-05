import sys
from funcs import (
    fitTemplate2,
    fitTemplate,
    getTemplate,
    findSections,
    gaussian_loglike_pval,
    convertRowsForFit,
    scaleOnGaussian,
    scaleTemplate,
    isFlat,
    isOnePixel,
    isOnEdge,
    filterForTemplate,
)
from findPerfectCluster import graphTSonRows
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from dataAnalysis.handlers._perfectCluster import findBestSections
from dataAnalysis.handlers._genericClusterFuncs import findConnectedSections
from dataAnalysis.handlers._goodCluster import isGoodCluster
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm
from pixelCharges.track import getOrthClusterCharge,angleFromCharge

def graphTSonRows(
    axs,
    cluster,
    section,
    flatScaler,
    angleScaler,
    flipped,
    estimateTemplate,
    spreadTemplate,
    excludeCrossTalk=True,
):
    estimate, spread = scaleTemplate(estimateTemplate, spreadTemplate, angleScaler, flatScaler)
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)
    relativeRows = Rows - np.min(Rows)
    relativeTS = Timestamps - np.min(Timestamps)
    mask = np.ones(len(relativeRows), dtype=bool)
    mask[section,] = False
    x = relativeRows
    if flipped:
        x = abs(x - np.max(x))
    axs.scatter(
        x[~mask],
        relativeTS[~mask],
        color=plot.colorPalette[2],
        marker="x",
        label="In Fitted Sections",
    )
    axs.scatter(
        x[mask],
        relativeTS[mask],
        color=plot.colorPalette[8],
        marker="x",
        label="Not in Fitted Sections",
    )
    if len(section) != 0:
        if flipped:
            anchor = np.argmax(relativeRows[section])
            rowsForFunc = relativeRows[section][anchor] - relativeRows
        else:
            anchor = np.argmin(relativeRows[section])
            rowsForFunc = relativeRows - relativeRows[section][anchor]
        sortIndexes = np.argsort(relativeRows)
        rowsForFunc = rowsForFunc[sortIndexes]
        relativeRows = relativeRows[sortIndexes]
        rowsToGraph = np.linspace(
            relativeRows[rowsForFunc == 0][0],
            relativeRows[rowsForFunc == 0][0] + len(estimate) + (len(estimate) * -int(flipped) * 2),
            len(estimate),
        )
        if flipped:
            rowsToGraph = abs(rowsToGraph - np.max(rowsToGraph))
        axs.plot(
            rowsToGraph,
            estimate,
            color=plot.colorPalette[0],
            linestyle="dashed",
            label=f"Fitted: flatScaler {flatScaler}, {np.rad2deg(np.atan(angleScaler*np.tan(np.deg2rad(86.5)))):.2f} Degrees",
        )
        axs.fill_between(
            rowsToGraph, estimate - spread, estimate + spread, alpha=0.2, color=plot.colorPalette[0]
        )
        rowsToGraph = np.linspace(
            relativeRows[rowsForFunc == 0][0],
            relativeRows[rowsForFunc == 0][0]
            + len(estimateTemplate)
            + (len(estimateTemplate) * -int(flipped) * 2),
            len(estimateTemplate),
        )
        if flipped:
            rowsToGraph = abs(rowsToGraph - np.max(rowsToGraph))
        axs.plot(
            rowsToGraph,
            estimateTemplate,
            color=plot.colorPalette[3],
            linestyle="dashed",
            label=f"Original",
        )
        axs.fill_between(
            rowsToGraph,
            estimateTemplate - spreadTemplate,
            estimateTemplate + spreadTemplate,
            alpha=0.2,
            color=plot.colorPalette[3],
        )
    plot.set_config(
        axs,
        title="Relative TS in cluster",
        xlabel="Relative Row",
        ylabel="Relative TS",
        legend=True,
        ylim=(-0.5, np.max(relativeTS) + 5),
        xlim=(np.min(relativeRows) - 1, np.max(relativeRows) + 1),
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1))

orthCharge, orthCharge_e = getOrthClusterCharge(layer=4)

np.set_printoptions(precision=3, linewidth=125)

config = configLoader.loadConfig()
#config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    dataFile.init_cluster_voltages()
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
    estimate, spread = getTemplate(config)
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    params1List = []
    params2List = []
    firstPixel = []
    i = 0
    k = 0
    for cluster in tqdm(clusters[10500:], desc="Running over clusters"):
        if not isGoodCluster(cluster):
            continue
        sections = findSections(cluster, excludeCrossTalk=True)
        section = np.arange(len(cluster.getRows(excludeCrossTalk=True)))
        Timestamps = cluster.getTSs(excludeCrossTalk=True)[section]
        Rows = cluster.getRows(excludeCrossTalk=True)[section]
        relativeTS = Timestamps - np.min(Timestamps)
        if np.all(relativeTS <= 4):
            continue
        i += 1
        # print(Rows,Timestamps)
        # start = time.time()
        angleList = np.linspace(0.1, 1.8, 35)
        angleList_ = np.linspace(70,89.75,80)
        angleList = np.tan(np.deg2rad(angleList_))/np.tan(np.deg2rad(dataFile.angle))
        flatList = np.linspace(0.2, 2, 37)
        flatScalerList, angleScalerList, pValList, flippedList, permList = fitTemplate2(
            cluster,
            estimate,
            spread,
            excludeCrossTalk=True,
            minPval=0.05,
            angleList=angleList,
            flatList=flatList,
        )
        
        # params,flipped = fitTemplate(cluster,section,estimate,spread,excludeCrossTalk=True)
        # pVal = gaussian_loglike_pval(scaleOnGaussian(*filterForTemplate(*convertRowsForFit(Rows,Timestamps,flipped=flipped),*scaleTemplate(estimate,spread,params[0],params[1]))))
        # if pVal < 0.2:
        #    k += 1
        #    continue
        # end = time.time()
        # print(f"{end - start} seconds to fit")
        # if params[0] == 0.5 and params[1] == 0.5:
        #    k += 1
        #    continue
        # x,y = convertRowsForFit(Rows,Timestamps,flipped=flipped)
        # params1List.append(params[0])
        # params2List.append(params[1])
        # firstPixel.append(y[x==0])
        # print(params)
        # print(flipped)
        path = base_path + f"Cluster_{cluster.getIndex()}/"
        # print(scaleTemplate(estimate,spread,*params)[0])
        # print(estimate)
        #print(flatScalerList)
        #print(angleScalerList)
        #print(pValList)
        #print(flippedList)
        #print(permList)
        # plot = plotClass(path)
        # axs = plot.axs
        # graphTSonRows(axs,cluster,section,*scaleTemplate(estimate,spread,*params),flipped,estimate,spread)
        # plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Row_vs_RelativeTS")
        plot = plotClass(path)
        axs = plot.axs
        graphTSonRows(axs, cluster, section, 1, 1, False, estimate, spread)
        plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Row_vs_RelativeTS")
        if len(flatScalerList) > 0:
            plot = plotClass(path)
            axs = plot.axs
            graphTSonRows(
                axs,
                cluster,
                section,
                flatScalerList[np.argmax(pValList)],
                angleScalerList[np.argmax(pValList)],
                flippedList[np.argmax(pValList)],
                estimate,
                spread,
            )
            sections = np.array(sections,dtype=object)
            section = []
            for i in permList[np.argmax(pValList)]:
                section.extend(sections[int(i)])
            plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Row_vs_RelativeTS_many")
            angleScalerList = np.rad2deg(np.atan(angleScalerList*np.tan(np.deg2rad(dataFile.angle))))
            angleList = angleList_
            plot = plotClass(path)
            axs = plot.axs
            array, yedges, xedges = np.histogram2d(
                angleScalerList,
                flatScalerList,
                bins=(angleList.size,flatList.size),
                range=(
                    (
                        angleList[0] - (angleList[1] - angleList[0]) / 2,
                        angleList[-1] + (angleList[1] - angleList[0]) / 2,
                    ),
                    (
                        flatList[0] - (flatList[1] - flatList[0]) / 2,
                        flatList[-1] + (flatList[1] - flatList[0]) / 2,
                    )
                ),
            )
            array[:,:] = 0
            array[
                np.round((angleScalerList-np.min(angleList))/(angleList[1] - angleList[0])).astype(int),
                np.round((flatScalerList-np.min(flatList))/(flatList[1] - flatList[0])).astype(int),
                  ] = pValList
            axs.imshow(
                array,
                aspect="auto",
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                vmax = 1,
            )
            axs.set_xticks(flatList[::2])
            axs.set_yticks(angleList[::4])
            axs.set_xticks(
                np.linspace(
                    flatList[0] - (flatList[1] - flatList[0]) / 2,
                    flatList[-1] + (flatList[1] - flatList[0]) / 2,
                    flatList.size + 1,
                ),
                minor=True,
            )
            axs.set_yticks(
                np.linspace(
                    angleList[0] - (angleList[1] - angleList[0]) / 2,
                    angleList[-1] + (angleList[1] - angleList[0]) / 2,
                    angleList.size + 1,
                ),
                minor=True,
            )
            axs.grid(which="minor", color="black", linestyle="-", linewidth=1)
            angle, angle_e = angleFromCharge(
                orthCharge,
                orthCharge_e,
                cluster.getClusterCharge(True,section),
                cluster.getClusterChargeError(True,section),
            )
            angle_high,_ = angleFromCharge(
                orthCharge,
                orthCharge_e,
                cluster.getClusterCharge(True,section)*1.2,
                cluster.getClusterChargeError(True,section)*1.2,
            )
            angle_low,_ = angleFromCharge(
                orthCharge,
                orthCharge_e,
                cluster.getClusterCharge(True,section)*0.8,
                cluster.getClusterChargeError(True,section)*0.8,
            )
            print(cluster.getClusterCharge(True,section))
            print(angle)
            print([angle_low,angle_high])
            axs.hlines(angle,flatList[0] - (flatList[1] - flatList[0]) / 2,flatList[-1] + (flatList[1] - flatList[0]) / 2, color=plot.colorPalette[0], linestyle="--")
            axs.hlines([angle_low,angle_high],flatList[0] - (flatList[1] - flatList[0]) / 2,flatList[-1] + (flatList[1] - flatList[0]) / 2, color=plot.colorPalette[2], linestyle="--")
            plot.set_config(
                axs,
                title=f"Fitted Params {angle}",
                xlabel="Flat Scaler",
                ylabel="Angle Scaler",
                ylim=(yedges[0],yedges[-1]),
                xlim=(xedges[0],xedges[-1]),
            )

            plot.saveToPDF(f"Fitting_Params")

        input("")
    print("")
    print(f"{len(params1List)/i*100:.2f}% of {i} clusters had fits")
    print(f"{k/i*100:.2f}% of {i} clusters had failed fits")
    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    array, yedges, xedges = np.histogram2d(
        params1List, params2List, bins=(100, 100), range=((0, 2), (0, 2))
    )
    axs.imshow(
        array, aspect="auto", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plot.set_config(
        axs,
        title="Fitted Params",
        xlabel="Flat Scaler",
        ylabel="Angle Scaler",
    )
    plot.saveToPDF(f"Fitting_Params")
    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    params1List = np.array(params1List)
    params2List = np.array(params2List)
    ratio = (params1List + params2List) / 2
    angle = np.rad2deg(np.tan(ratio * np.arctan(np.deg2rad(dataFile.angle))))
    height, x = np.histogram(angle, bins=90, range=(0, 90))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    plot.set_config(
        axs,
        title="Angle of clusters",
        xlabel="Angle",
        ylabel="Count",
        # xlim=(75,90),
    )
    plot.saveToPDF(f"Fitting_Params_Angle")
