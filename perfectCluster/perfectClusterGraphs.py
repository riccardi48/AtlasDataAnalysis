import sys
from funcs import (
    findConnectedSections,
    getTemplate,
    isPerfectCluster,
    loadOrCalcMPV,
    filterForTemplate,
    convertRowsForFit,
)

sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm
from pixelCharges.plotCluster import clusterPlotter
from landau import landau

config = configLoader.loadConfig()
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    dataFile.init_cluster_voltages()
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
    estimate, spread = getTemplate(config)
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    MPV_ParamsList = []
    missingRowsList = []
    presentRowsList = []
    presentRowsList2 = []
    relativeRowsList = []
    missingRelativeRowsList = []
    presentRelativeRowsList = []
    MPV_Params = loadOrCalcMPV(dataFile, config)
    cutOff = 12
    for cluster in tqdm(clusters, desc="Running over clusters"):
        if not isPerfectCluster(cluster, estimate, spread, minPval=0.2, excludeCrossTalk=True):
            continue
        rows = cluster.getRows(True)[cluster.section]
        columns = cluster.getColumns(True)[cluster.section]
        timestamps = cluster.getTSs(True)[cluster.section]
        voltages = cluster.getHit_Voltages(True)[cluster.section]
        sections = findConnectedSections(rows, columns)
        if len(sections) == 1:
            continue
        sectionsMinMax = []
        for section in sections:
            sectionsMinMax.append((np.min(rows[section]), np.max(rows[section])))
        # print(sectionsMinMax)
        # print(len(cluster.perm))
        expectedRows = np.linspace(
            np.sort(rows)[-1 * cluster.flipped],
            np.sort(rows)[-1 * cluster.flipped]
            + len(MPV_Params[:, 0])
            + len(MPV_Params[:, 0]) * 2 * -1 * cluster.flipped,
            len(MPV_Params[:, 0]) + 1,
        ).astype(int)[:-1]
        expectedRows = expectedRows[(expectedRows >= 0) & (expectedRows < 372)]
        expectedRowsRelative, _ = convertRowsForFit(expectedRows, expectedRows, flipped=False)
        x, _ = convertRowsForFit(rows, rows, flipped=cluster.flipped)
        missing = np.array([r for r in expectedRowsRelative if r not in x])
        index = (x < len(estimate)) & (x >= 0)
        x = x[index]
        # print(MPV_Params[:,0][missing])
        MPV_ParamsList.extend(MPV_Params[missing[missing < len(MPV_Params)]])

        # print([1-landau.cdf(0.16,mpv,width) for mpv,width in zip(MPV_Params[:,0][missing],MPV_Params[:,1][missing])])
        missingRow = np.array([r for r in expectedRows if r not in rows])
        missingRowsList.extend(missingRow[missing < len(MPV_Params[:, 0])])
        relativeRowsList.extend(missing[missing < len(MPV_Params[:, 0])])
        presentRelativeRowsList.extend(expectedRowsRelative)
        presentRowsList.extend(expectedRows[expectedRowsRelative <= cutOff])
        presentRowsList2.extend(expectedRows[expectedRowsRelative > cutOff])
        # print(missing)
        #path = base_path + f"Cluster_{cluster.getIndex()}/"
        #clusterPlotter(cluster, path, "Relative TS").finishPlot(
        #    "Relative TS",
        #    cluster.getTSs(True) - np.min(cluster.getTSs(True)),
        #    True,
        #    cmap="plasma_r",
        #)
        #input()
    MPV_ParamsList = np.array(MPV_ParamsList)
    missingRowsList = np.array(missingRowsList)
    presentRowsList = np.array(presentRowsList)
    presentRowsList2 = np.array(presentRowsList2)
    relativeRowsList = np.array(relativeRowsList)
    missingRelativeRowsList = np.array(missingRelativeRowsList)
    presentRelativeRowsList = np.array(presentRelativeRowsList)
    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    array, yedges, xedges = np.histogram2d(
        MPV_ParamsList[:, 0], missingRowsList, bins=(100, 372), range=((0, 0.5), (0.5, 371.5))
    )
    axs.imshow(
        array, aspect="auto", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plot.set_config(
        axs,
        title="Expected MPV vs Row Position",
        xlabel="Row",
        ylabel="MPV Gap",
    )
    plot.saveToPDF(f"MPVofGAPS")

    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    array, yedges, xedges = np.histogram2d(
        MPV_ParamsList[:, 0][relativeRowsList <= cutOff],
        missingRowsList[relativeRowsList <= cutOff],
        bins=(100, 372),
        range=((0, 0.5), (0.5, 371.5)),
    )
    axs.imshow(
        array, aspect="auto", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plot.set_config(
        axs,
        title="Expected MPV vs Row Position",
        xlabel="Row",
        ylabel="MPV Gap",
    )
    plot.saveToPDF(f"MPVofGAPS_CloseRows")

    landauCDFList = np.array(
        [
            landau.cdf(0.16, mpv, width)
            for mpv, width in zip(MPV_ParamsList[:, 0], MPV_ParamsList[:, 1])
        ]
    )
    # print(landauCDFList)
    # print(landauCDFList[relativeRowsList<=cutOff])
    rowFrequency = np.zeros(372)
    for row in presentRowsList:
        rowFrequency[row] += 1
    missingRowFrequency = np.zeros(372)
    for row in missingRowsList[relativeRowsList <= cutOff]:
        missingRowFrequency[row] += 1
    rowPercent = missingRowFrequency / (rowFrequency + 1e-10)
    # print(rowPercent)
    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    xRange = (0, 371)
    yRange = (0, 0.2)
    array, yedges, xedges = np.histogram2d(
        landauCDFList[relativeRowsList <= cutOff],
        missingRowsList[relativeRowsList <= cutOff],
        bins=(100, 372),
        range=((yRange), (xRange[0] - 0.5, xRange[1] + 0.5)),
    )
    axs.imshow(
        array, aspect="auto", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    # axs.plot(np.arange(rowPercent.size),rowPercent,color=plot.colorPalette[0],linestyle='dashed',label="Missing Row %")
    axs.scatter(
        np.arange(rowPercent.size),
        rowPercent,
        color=plot.colorPalette[0],
        marker="x",
        label="Missing Row %",
    )
    plot.set_config(
        axs,
        title="Chance of no Hit vs Missing Row Percentage",
        xlabel="Row",
        ylabel="Chance of no Hit",
        ylim=(yRange),
        xlim=(xRange),
        legend=True,
    )
    plot.saveToPDF(f"ChanceOfNoHitvsRowPercentage_CloseRows")

    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs

    rowFrequency = np.zeros(372)
    for row in presentRowsList2:
        rowFrequency[row] += 1
    missingRowFrequency = np.zeros(372)
    for row in missingRowsList[relativeRowsList > cutOff]:
        missingRowFrequency[row] += 1
    rowPercent = missingRowFrequency / (rowFrequency + 1e-10)

    array, yedges, xedges = np.histogram2d(
        landauCDFList[relativeRowsList > cutOff],
        missingRowsList[relativeRowsList > cutOff],
        bins=(100, 372),
        range=((yRange), (xRange[0] - 0.5, xRange[1] + 0.5)),
    )
    axs.imshow(
        array, aspect="auto", origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    axs.scatter(
        np.arange(rowPercent.size),
        rowPercent,
        color=plot.colorPalette[0],
        marker="x",
        label="Missing Row %",
    )
    plot.set_config(
        axs,
        title="Chance of no Hit vs Missing Row Percentage",
        xlabel="Row",
        ylabel="Chance of no Hit",
        ylim=(yRange),
        xlim=(xRange),
        legend=True,
    )
    plot.saveToPDF(f"ChanceOfNoHitvsRowPercentage_FarRows")

    rowFrequency = np.zeros(len(MPV_Params[:, 0]))
    for row in presentRelativeRowsList:
        rowFrequency[row] += 1
    missingRowFrequency = np.zeros(len(MPV_Params[:, 0]))
    for row in relativeRowsList:
        missingRowFrequency[row] += 1
    rowPercent = missingRowFrequency / (rowFrequency + 1e-10)
    error = np.sqrt((rowPercent * (1 - rowPercent)) / (rowFrequency + 1e-10))
    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    landauCDFList = np.array(
        [
            landau.cdf(0.16, mpv, width)
            for mpv, width in zip(MPV_Params[:, 0], MPV_Params[:, 1])
        ]
        )
    axs.plot(np.arange(rowPercent.size), 1-landauCDFList, color=plot.colorPalette[1], label="Estimated Efficiency")
    axs.scatter(
        np.arange(rowPercent.size),
        1-rowPercent,
        color=plot.colorPalette[0],
        marker="x",
        label="Missing Row %",
    )
    axs.errorbar(
        np.arange(rowPercent.size),
        1-rowPercent,
        yerr=error,
        fmt="none",
        color=plot.colorPalette[0],
        elinewidth=1,
        capsize=3,
    )
    plot.set_config(
        axs,
        title="Efficiency by relative Row",
        xlabel="Row",
        ylabel="Efficiency",
        legend=True,
        #ylim=(0.8,1),
    )
    plot.saveToPDF(f"Efficiency_Relative_Row")

