import sys
from functions import timeSectionPlots
sys.path.append("..")
from dataAnalysis.handlers._genericClusterFuncs import isFlat
from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np
from astropy.timeseries import LombScargle
from matplotlib.ticker import MultipleLocator


config = configLoader.loadConfig()
config["filterDict"] = {"angle":86.5,"voltage":48.6,"telescope":"lancs"}
dataFiles = initDataFiles(config)

for i, dataFile in enumerate(dataFiles[:2]):
    path = f"TimeTests/{dataFile.fileName}/"
    dataFile.init_cluster_voltages()
    # times,indexes = dataFile.get_cluster_attr("Times", layer=4, excludeCrossTalk=True,returnIndexes=True)
    clusters = dataFile.get_clusters(excludeCrossTalk=True, layers=config["layers"])
    #clusters =  [cluster for cluster in clusters if isFlat(cluster)]
    firstTime = clusters[0].getTimes(True)[0]
    # clusters = dataFile.get_perfectClusters(excludeCrossTalk=True,layers=config["layers"])
    times = np.array([cluster.getTimes(True)[0] for cluster in clusters]) - firstTime
    clusterCharges = np.array([cluster.getClusterCharge(True) for cluster in clusters])
    clusterWidths = np.array([cluster.getRowWidth(True) for cluster in clusters])
    values = np.ones_like(times)
    max_freq = 1 / 10
    min_freq = 1 / 100000
    freqs = np.linspace(min_freq, max_freq, 100000)
    power = LombScargle(times, values).power(freqs)
    sortIndex = np.argsort(power)[::-1]
    lowFreq = freqs[sortIndex][freqs[sortIndex] < 0.01][0]
    highFreq = freqs[sortIndex][freqs[sortIndex] > 0.01][0]
    highFreq = 12.5 / 1000
    lowFreq = 0.018 / 1000
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    timesMod = times % (1 / lowFreq)
    height, x = np.histogram(timesMod, bins=100)
    offset = x[np.argmin(height)]
    timesMod = (timesMod - offset) % (1 / lowFreq)
    height, x = np.histogram(timesMod, bins=100)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1], label=f"{lowFreq*1000:.2f} Hz")
    plot.set_config(
        axs,
        legend=True,
    )
    plot.saveToPDF(f"TimeGroups_low")
    period = 1 / lowFreq
    highPeriod = 1 / highFreq
    allChargesMod = np.array([])
    allTimesMod = np.array([])
    allWidthsMod = np.array([])
    cutEdges = 0.05*period
    plotRatio = plotClass(config["pathToOutput"] + path)
    for n in range(int(np.max(times) / period)):
        index = np.array(
            np.where((times >= period * n + offset+cutEdges) & (times <= period * (n + 1) + offset-cutEdges))[0]
        )
        timesToUse = times[index]
        timesMod = timesToUse % (1 / highFreq)
        height, x = np.histogram(timesMod, bins=100)
        smallOffset = (x[np.argmax(height)] + 0.8 / highFreq) % (1 / highFreq)
        print((x[np.argmax(height)] + 0.5 / highFreq) % (1 / highFreq))
        print(x[np.argmin(height)])
        timesMod = (timesMod - smallOffset) % (1 / highFreq)
        chargeToBeUsed = clusterCharges[index]
        widthToBeUsed = clusterWidths[index]
        cutIndex = (widthToBeUsed > 18)&(widthToBeUsed < 35)&(chargeToBeUsed > 8)&(chargeToBeUsed < 20)
        timeSectionPlots(config,path,n,dataFile,timesMod,widthToBeUsed,chargeToBeUsed,highFreq,cutIndex)
        plotRatio.axs.scatter(n,np.sum(cutIndex),color=plotRatio.colorPalette[0],marker="x")
        plotRatio.axs.scatter(n,np.sum(np.invert(cutIndex)),color=plotRatio.colorPalette[2],marker="x")
        allChargesMod = np.concatenate((allChargesMod, chargeToBeUsed))
        allTimesMod = np.concatenate((allTimesMod, timesMod))
        allWidthsMod = np.concatenate((allWidthsMod, widthToBeUsed))
    plotRatio.set_config(
        plotRatio.axs,
        ylim=(0,None),
    )
    plotRatio.saveToPDF(f"TimeGroups_Ratios")
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    height, x = np.histogram(allTimesMod, bins=100)
    axs.stairs(
        height, x, baseline=None, color=plot.colorPalette[1], label=f"{highFreq*1000:.2f} Hz"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, None),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_combined")
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    height, x = np.histogram(allTimesMod[allWidthsMod > 18], bins=100)
    axs.stairs(
        height, x, baseline=None, color=plot.colorPalette[1], label=f"{highFreq*1000:.2f} Hz"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, None),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_combined_highWidth")
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    height, x = np.histogram(allTimesMod[allWidthsMod < 6], bins=100)
    axs.stairs(
        height, x, baseline=None, color=plot.colorPalette[1], label=f"{highFreq*1000:.2f} Hz"
    )
    plot.set_config(
        axs,
        legend=True,
        xlim=(0, None),
        ylim=(0,None),
    )
    plot.saveToPDF(f"TimeGroups_high_combined_lowWidth")

    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    array, xedges, yedges = np.histogram2d(
        allTimesMod, allChargesMod, (100, 40), ((np.min(allTimesMod), np.max(allTimesMod)), (0, 20))
    )
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    array = array.transpose()
    # array = array / np.sum(array,axis=1)[:, np.newaxis]
    im = axs.imshow(
        array,
        aspect=(xedges[-1] - xedges[0]) / (yedges[-1] - yedges[0]),
        origin="lower",
        extent=extent,
    )
    plot.set_config(axs, xlabel="Time", ylabel="Charge")
    plot.saveToPDF(f"TimeGroups_high_hist2d_charge")
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    array = array / np.sum(array, axis=1)[:, np.newaxis]
    im = axs.imshow(
        array,
        aspect=(xedges[-1] - xedges[0]) / (yedges[-1] - yedges[0]),
        origin="lower",
        extent=extent,
    )
    plot.set_config(axs, xlabel="Time", ylabel="Charge")
    plot.saveToPDF(f"TimeGroups_high_hist2d_charge_norm")

    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    array, xedges, yedges = np.histogram2d(
        allTimesMod,
        allWidthsMod,
        (100, 30),
        ((np.min(allTimesMod), np.max(allTimesMod)), (0.5, 30.5)),
    )
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    array = array.transpose()
    # array = array / np.sum(array,axis=1)[:, np.newaxis]
    im = axs.imshow(
        array,
        aspect=(xedges[-1] - xedges[0]) / (yedges[-1] - yedges[0]),
        origin="lower",
        extent=extent,
    )
    plot.set_config(axs, xlabel="Times", ylabel="Widths")
    plot.saveToPDF(f"TimeGroups_high_hist2d_width")

    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    array = array / np.sum(array, axis=1)[:, np.newaxis]
    im = axs.imshow(
        array,
        aspect=(xedges[-1] - xedges[0]) / (yedges[-1] - yedges[0]),
        origin="lower",
        extent=extent,
    )
    plot.set_config(axs, xlabel="Time", ylabel="Charge")
    plot.saveToPDF(f"TimeGroups_high_hist2d_width_norm")

    minTime = 135000
    maxTime = 135500
    timeRange = (minTime, maxTime)
    bins = int(np.ptp(timeRange) / 1)
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    height, x = np.histogram(times, bins=bins, range=timeRange)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1], label=f"Layer 4 Bin width 1ms")
    x = np.linspace(timeRange[0], timeRange[1], 1000)
    highFreq = freqs[sortIndex][freqs[sortIndex] > 0.01][0]
    y = np.sin(highFreq * 2 * np.pi * x)
    plot.set_config(
        axs,
        ylim=(0, axs.get_ylim()[1] * 1.2),
        xlim=timeRange,
        title="Clusters Count Over Time",
        legend=True,
        xlabel="Time [ms]",
        ylabel="Count",
    )
    axs.vlines(
        highPeriod
        * np.arange(
            int(np.ptp(times[times < timeRange[1]]) / highPeriod)
        )
        + smallOffset,
        0,
        axs.get_ylim()[1],
        color=plot.colorPalette[2],
        label="12.5 Hz"
    )
    plot.saveToPDF(f"ClusterTimes_small")

    minTime = 0
    maxTime = 600000
    timeRange = (minTime, maxTime)
    bins = int(np.ptp(timeRange) / 1000)
    plot = plotClass(config["pathToOutput"] + path)
    axs = plot.axs
    height, x = np.histogram(times, bins=bins, range=timeRange)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1], label=f"Layer 4 Bin width 1s")
    x = np.linspace(timeRange[0], timeRange[1], 1000)
    lowFreq = freqs[sortIndex][freqs[sortIndex] < 0.01][0]
    y = np.sin(lowFreq * 2 * np.pi * x)

    plot.set_config(
        axs,
        ylim=(0, axs.get_ylim()[1] * 1.2),
        xlim=timeRange,
        title="Clusters Count Over Time",
        legend=True,
        xlabel="Time [ms]",
        ylabel="Count",
    )
    axs.vlines(
        period * np.arange(int(np.max(times) / period)) + offset,
        0,
        axs.get_ylim()[1],
        color=plot.colorPalette[2],
        label="55.6s Period"
    )
    plot.saveToPDF(f"ClusterTimes_big")

