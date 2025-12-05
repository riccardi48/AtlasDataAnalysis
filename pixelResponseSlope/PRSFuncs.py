import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
from plotAnalysis import plotClass
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
from dataAnalysis._fileReader import calcDataFileManager

def linearLine(x,m,c):
    return m*x + c

def getPixelResponseSlope(rows,chargeCollected,chargeCollected_e):
    if len(rows) <1:
        return np.nan
    valid = (chargeCollected>0)
    valid[-1] = False
    valid[0] = False
    #valid[rows<np.max(rows)-15] = False
    rows = rows[valid]
    chargeCollected_e = chargeCollected_e[valid]
    chargeCollected = chargeCollected[valid]
    if len(rows) <= 2 or np.unique(rows).size < 2 or len(rows) > 60:
        return np.nan
    relativeRows = rows-np.min(rows)
    #popt,pcov = curve_fit(linearLine,relativeRows,chargeCollected,sigma=chargeCollected_e,absolute_sigma=True,maxfev=10000)
    result = linregress(relativeRows,chargeCollected,nan_policy="omit")
    return result.slope
    return popt[0]

def cauchyFunc(x,gamma,scale):
    return 1/np.pi * (gamma / (x**2 + gamma**2)) * scale

def sumOfCauchyFunc(x,gamma1,gamma2,gamma3,scale1,scale2,scale3):
    return cauchyFunc(x,gamma1,scale1) + cauchyFunc(x,gamma2,scale2) + cauchyFunc(x,gamma3,scale3)

def loadOrCalcPRS(dataFile,config):
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

def plotHist(dataFile,path,height,x,_range,xlim,name=""):
    plot = plotClass(path)
    axs = plot.axs
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
        title=f"Pixel Response Slope Histogram {dataFile.fileName} {name}",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_{dataFile.fileName}_{name}")

def plotHistRemoved(dataFile,path,height,x,xlim,name=""):
    plot = plotClass(path)
    axs = plot.axs
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    binCentres = (x[:-1] + x[1:]) / 2
    maxY = np.max(height[(binCentres<0.5)&(binCentres>-0.5)])
    plot.set_config(
        axs,
        ylim=(0, maxY),
        xlim=xlim,
        title=f"Pixel Response Slope Histogram {dataFile.fileName} {name}",
        xlabel="Pixel Response Slope",
        ylabel="Frequency",
        )
    plot.saveToPDF(f"PixelResponseSlope_{dataFile.fileName}_{name}")
