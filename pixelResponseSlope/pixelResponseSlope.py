import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
import numpy as np
from dataAnalysis._fileReader import calcDataFileManager

config = configLoader.loadConfig()
#config["filterDict"] = {"telescope":"kit","fileName":"angle6_4Gev_kit_2"}
config["filterDict"] = {"telescope":"kit","angle":45}
dataFiles = initDataFiles(config)

def linearLine(x,m,c):
    return m*x + c

def getPixelResponseSlope(rows,chargeCollected,chargeCollected_e):
    rows = rows[chargeCollected>0]
    chargeCollected_e = chargeCollected_e[chargeCollected>0]
    chargeCollected = chargeCollected[chargeCollected>0]
    if len(rows) <= 1 or np.unique(rows).size < 1:
        return np.nan
    relativeRows = rows-np.min(rows)
    popt,pcov = curve_fit(linearLine,relativeRows,chargeCollected,sigma=chargeCollected_e,absolute_sigma=True)
    return popt[0]

def cauchyFunc(x,gamma,scale):
    return 1/np.pi * (gamma / (x**2 + gamma**2)) * scale

def sumOfCauchyFunc(x,gamma1,gamma2,gamma3,gamma4,scale1,scale2,scale3,scale4):
    return cauchyFunc(x,gamma1,scale1) + cauchyFunc(x,gamma2,scale2) + cauchyFunc(x,gamma3,scale3) + cauchyFunc(x,gamma4,scale4)

def loadOrCalcPRS(dataFile):
    calcFileManager = calcDataFileManager(config["pathToCalcData"], "PRS", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"{dataFile.fileName}",
    )
    fileCheck = calcFileManager.fileExists(calcFileName=calcFileName)
    if not fileCheck:
        dataFile.init_cluster_voltages()
        PRS = np.array([getPixelResponseSlope(cluster.getRows(excludeCrossTalk=True),cluster.getHit_Voltages(excludeCrossTalk=True),cluster.getHit_VoltageErrors(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True)])
        calcFileManager.saveFile(calcFileName=calcFileName,array=PRS)
    else:
        PRS = calcFileManager.loadFile(calcFileName=calcFileName)
    return PRS



for dataFile in dataFiles:
    dataFile.init_cluster_voltages()
    _range = (-0.2005,0.2005)
    _range = (-1.0025,1.0025)
    bins = 401
    xlim = (-0.1,0.1)
    xlim = (-0.95,0.95)
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/")
    axs = plot.axs
    PRS = loadOrCalcPRS(dataFile)
    clusters,indexes = dataFile.get_clusters(excludeCrossTalk=True,returnIndexes=True,layer=4)
    PRS = PRS[indexes]
    notRemoved = (~np.isnan(PRS)) & (PRS != 1) & (PRS != -1)
    PRS = PRS[notRemoved]
    height, x = np.histogram(PRS, bins=bins, range=_range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    binCentres = (x[:-1] + x[1:]) / 2
    popt, pcov = curve_fit(
            sumOfCauchyFunc,
            binCentres[binCentres<0],
            height[binCentres<0],
            maxfev=10000,
       )
    _x = np.linspace(_range[0],_range[1],1000)
    y = sumOfCauchyFunc(_x, *popt)
    axs.plot(_x[_x<0], y[_x<0], color=plot.colorPalette[2])
    axs.plot(_x[_x>0], y[_x>0], color=plot.colorPalette[2], linestyle = "dashed")
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=xlim,
        title=f"Pixel Response Slope Histogram {dataFile.fileName} {_range}",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_{dataFile.fileName}_{_range[0]}_{_range[1]}")

    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/removedPeak/")
    axs = plot.axs
    height = height - sumOfCauchyFunc(binCentres, *popt)
    height[binCentres==0] = 0
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    plot.set_config(
        axs,
        ylim=(0, np.max(height[(binCentres<0.5)&(binCentres>-0.5)])),
        xlim=xlim,
        title=f"Pixel Response Slope Histogram {dataFile.fileName} {_range}",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_{dataFile.fileName}_{_range[0]}_{_range[1]}")

    clusterWidths = np.array([cluster.getRowWidth(excludeCrossTalk=True) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)])
    print(np.sum(clusterWidths>15))
    print(np.sum(clusterWidths<=15))
    print(clusterWidths.size)

    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/shortVsLong/")
    axs = plot.axs
    LongPRS = np.array([getPixelResponseSlope(cluster.getRows(excludeCrossTalk=True),cluster.getHit_Voltages(excludeCrossTalk=True),cluster.getHit_VoltageErrors(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)[(clusterWidths>15)&clusterWidths<50]])
    notRemoved = (~np.isnan(LongPRS)) & (LongPRS != 1) & (LongPRS != -1)
    LongPRS = LongPRS[notRemoved]
    height, x = np.histogram(LongPRS, bins=bins, range=_range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    binCentres = (x[:-1] + x[1:]) / 2
    popt, pcov = curve_fit(
            sumOfCauchyFunc,
            binCentres[binCentres<0],
            height[binCentres<0],
            maxfev=10000,
       )
    _x = np.linspace(_range[0],_range[1],1000)
    y = sumOfCauchyFunc(_x, *popt)
    axs.plot(_x[_x<0], y[_x<0], color=plot.colorPalette[2])
    axs.plot(_x[_x>0], y[_x>0], color=plot.colorPalette[2], linestyle = "dashed")
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=xlim,
        title=f"Pixel Response Slope Histogram {dataFile.fileName} Only >15",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_Long_{dataFile.fileName}_{_range[0]}_{_range[1]}")

    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/shortVsLong/removedPeaks/")
    axs = plot.axs
    height = height - sumOfCauchyFunc(binCentres, *popt)
    height[binCentres==0] = 0
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=xlim,
        title=f"Pixel Response Slope Histogram {dataFile.fileName} {_range}",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_{dataFile.fileName}_Long")

    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/shortVsLong/")
    axs = plot.axs
    ShortPRS = np.array([getPixelResponseSlope(cluster.getRows(excludeCrossTalk=True),cluster.getHit_Voltages(excludeCrossTalk=True),cluster.getHit_VoltageErrors(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)[clusterWidths<=15]])
    notRemoved = (~np.isnan(ShortPRS)) & (ShortPRS != 1) & (ShortPRS != -1)
    ShortPRS = ShortPRS[notRemoved]
    height, x = np.histogram(ShortPRS, bins=bins, range=_range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    binCentres = (x[:-1] + x[1:]) / 2
    popt, pcov = curve_fit(
            sumOfCauchyFunc,
            binCentres[binCentres<0],
            height[binCentres<0],
            maxfev=10000,
       )
    _x = np.linspace(_range[0],_range[1],1000)
    y = sumOfCauchyFunc(_x, *popt)
    axs.plot(_x[_x<0], y[_x<0], color=plot.colorPalette[2])
    axs.plot(_x[_x>0], y[_x>0], color=plot.colorPalette[2], linestyle = "dashed")
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=xlim,
        title=f"Pixel Response Slope Histogram {dataFile.fileName} Only <=15",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_Short_{dataFile.fileName}_{_range[0]}_{_range[1]}")

    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/shortVsLong/removedPeaks/")
    axs = plot.axs
    height = height - sumOfCauchyFunc(binCentres, *popt)
    height[binCentres==0] = 0
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=xlim,
        title=f"Pixel Response Slope Histogram {dataFile.fileName} {_range}",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_{dataFile.fileName}_Short")

"""
    plot = plotClass(config["pathToOutput"] + "PixelResponseSlope/ScatterTest/")
    axs = plot.axs
    PRS = np.array([getPixelResponseSlope(cluster.getRows(excludeCrossTalk=True),cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)])
    Columns = np.array([np.mean(cluster.getColumns(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=4)])
    Columns = Columns[~np.isnan(PRS)]
    PRS = PRS[~np.isnan(PRS)]
    _range = (-0.1,0.1)
    axs.hist2d(Columns,PRS,bins=(135,210),range=((0,135),_range),cmin=1)
    plot.set_config(
        axs,
        ylim=_range,
        xlim=(0, None),
        title=f"Pixel Response Slope Histogram {dataFile.fileName}",
        xlabel="Column",
        ylabel="Pixel Response Slope",
        )
    plot.saveToPDF(f"PixelResponseSlope_Scatter_{dataFile.fileName}")
 """