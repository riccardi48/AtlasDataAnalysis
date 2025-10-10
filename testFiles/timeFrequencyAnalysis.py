import sys

sys.path.append("..")

from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np
from astropy.timeseries import LombScargle

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for i, dataFile in enumerate(dataFiles):
    times,indexes = dataFile.get_cluster_attr("Times", layer=4, excludeCrossTalk=True,returnIndexes=True)
    values = np.ones_like(times)
    max_freq = 1 / 10
    min_freq = 1 / 100000
    freqs = np.linspace(min_freq, max_freq, 100000)

    power = LombScargle(times, values).power(freqs)
    sortIndex = np.argsort(power)[::-1]
    minTime = 135000
    maxTime = 135500
    range = (minTime, maxTime)
    bins = int(np.ptp(range) / 1)
    plot = plotClass(config["pathToOutput"] + "TimeTests/FFT/")
    axs = plot.axs
    height, x = np.histogram(times, bins=bins, range=range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1], label=f"Layer 4 Bin width 1ms")
    x = np.linspace(range[0], range[1], 1000)
    highFreq = freqs[sortIndex][freqs[sortIndex]>0.01][0]
    y = np.sin(highFreq*2 * np.pi * x)
    axs.plot(x, y * np.max(height)*0.5, color=plot.textColor, label=f"{highFreq*1000:.3} Hz Sine Wave\nPeriod {1/highFreq/1000:.3} s")
    plot.set_config(
        axs,
        ylim=(0, axs.get_ylim()[1]*1.2),
        xlim=range,
        title="Clusters Count Over Time",
        legend=True,
        xlabel="Time [ms]",
        ylabel="Count",
    )
    plot.saveToPDF(f"ClusterTimes_{dataFile.fileName}_small")
    
    minTime = 0
    maxTime = 600000
    range = (minTime, maxTime)
    bins = int(np.ptp(range) / 1000)
    plot = plotClass(config["pathToOutput"] + "TimeTests/FFT/")
    axs = plot.axs
    height, x = np.histogram(times, bins=bins, range=range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[1], label=f"Layer 4 Bin width 1s")
    x = np.linspace(range[0], range[1], 1000)
    lowFreq = freqs[sortIndex][freqs[sortIndex]<0.01][0]
    y = np.sin(lowFreq*2 * np.pi * x)
    axs.plot(x, y * np.max(height)*0.5, color=plot.textColor, label=f"{lowFreq*1000:.3} Hz Sine Wave\nPeriod {1/lowFreq/1000:.3} s")
    plot.set_config(
        axs,
        ylim=(0, axs.get_ylim()[1]*1.2),
        xlim=range,
        title="Clusters Count Over Time",
        legend=True,
        xlabel="Time [ms]",
        ylabel="Count",
    )
    plot.saveToPDF(f"ClusterTimes_{dataFile.fileName}_big")
