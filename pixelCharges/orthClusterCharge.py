import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
import numpy as np
import numpy.typing as npt
from landau import landau
from dataAnalysis._fileReader import calcDataFileManager
from scipy.optimize import curve_fit

def landauFunc(
    x: npt.NDArray[np.float64],
    x_mpv: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    scaler: npt.NDArray[np.float64],
    #threshold: float = 0.16,
) -> npt.NDArray[np.float64]:
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    #y[x < threshold] = 0
    return y


def loadOrCalcOrthClusterCharge(dataFile,config,layer):
    calcFileManager = calcDataFileManager(config["pathToCalcData"], "ChargePeak", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"{dataFile.fileName}",
        name=f"{layer}"
    )
    fileCheck = calcFileManager.fileExists(calcFileName=calcFileName)
    if not fileCheck:
        peak,peak_e = calcChargePeak(dataFile,layer)
        calcFileManager.saveFile(calcFileName=calcFileName,array=[peak,peak_e])
    else:
        peak,peak_e = calcFileManager.loadFile(calcFileName=calcFileName)
    return peak,peak_e


def getOrthClusterCharge(Gev=6,layer=4):
    config = configLoader.loadConfig()
    if Gev == 4:
        config["filterDict"] = {"telescope":"kit","fileName":["4Gev_kit_1"]}
    else:
        #config["filterDict"] = {"telescope":"kit","fileName":["long_term_6Gev_kit_01","long_term_6Gev_kit_1","long_term_6Gev_kit_0","long_term_6Gev_kit_2","6Gev_kit_0"]}
        config["filterDict"] = {"telescope":"kit","fileName":["long_term_6Gev_kit_01"]}
    dataFiles = initDataFiles(config)
    peaks = []
    peaks_e = []
    for dataFile in dataFiles:
        peak,peak_e = loadOrCalcOrthClusterCharge(dataFile,config,layer)
        peaks.append(peak)
        peaks_e.append(peak_e)
    print(peaks,peaks_e)
    peaks = np.array(peaks, dtype=float)
    peaks_e = np.array(peaks_e, dtype=float)
    weights = 1 / peaks_e**2
    mean = np.sum(weights * peaks) / np.sum(weights)
    variance = np.sum(weights * (peaks - mean)**2) / ((len(peaks) - 1) * np.sum(weights) / len(peaks))
    if len(peaks) == 1:
        mean_error = peaks_e[0]
    else:
        mean_error = np.sqrt(variance / len(peaks))
    return mean, mean_error


def calcChargePeak(dataFile,layer):
    dataFile.init_cluster_voltages()
    clusterCharges = np.array([np.sum(cluster.getHit_Voltages(excludeCrossTalk=True)) for cluster in dataFile.get_clusters(excludeCrossTalk=True,layer=layer)])
    clusterCharges = clusterCharges[clusterCharges > 0]
    height, x = np.histogram(clusterCharges, bins=1000, range=(0, 20))
    binCentres = (x[:-1] + x[1:]) / 2
    cutOff=1
    popt, pcov = curve_fit(
        landauFunc,
        binCentres[(binCentres<cutOff)&(binCentres>0.161)],
        height[(binCentres<cutOff)&(binCentres>0.161)],
        )
    mpv = popt[0]
    mpv_e = np.sqrt(np.diag(pcov))[0]
    return mpv , mpv_e