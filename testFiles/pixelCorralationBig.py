import sys

sys.path.append("..")

from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataAnalysis.handlers._crossTalkFinder import crossTalkFinder
import matplotlib.pyplot as plt
from dataAnalysis._fileReader import calcDataFileManager

def checkHit(layer1, TS1, TriggerID1, layer2, TS2, TriggerID2, timeVariance=100, triggerVariance=1):
    if layer1 != layer2:
        return False
    elif abs(TriggerID1 - TriggerID2) > triggerVariance:
        return False
    elif abs(TS1 - TS2) > timeVariance and abs((TS1 + 512) - (TS2 + 512)) > timeVariance:
        return False
    return True


def calcToT(TS, TS2):
    return (TS2 * 2 - TS) % 256


def checkCorrelationType(ToT1, ToT2):
    if ToT1 < 30 and ToT2 < 30:
        return 1
    elif ToT1 < 30 and ToT2 >= 30 and ToT2 < 255:
        return 2
    elif ToT1 >= 30 and ToT2 < 30 and ToT1 < 255:
        return 3
    elif ToT1 >= 30 and ToT1 < 255 and ToT2 >= 30 and ToT2 < 255:
        return 4
    elif ToT1 >= 255:
        return 5
    elif ToT2 >= 255:
        return 6
    return 0


def findCorrelatedToT(pixelDF1, pixelDF2):
    pixel1ToT = []
    pixel2ToT = []
    for i, pixelDFRow1 in pixelDF1.iterrows():
        for j, pixelDFRow2 in pixelDF2.iterrows():
            if checkHit(
                pixelDFRow1["Layer"],
                pixelDFRow1["TS"],
                pixelDFRow1["TriggerID"],
                pixelDFRow2["Layer"],
                pixelDFRow2["TS"],
                pixelDFRow2["TriggerID"],
            ):
                pixel1ToT.append(calcToT(pixelDFRow1["TS"], pixelDFRow1["TS2"]))
                pixel2ToT.append(calcToT(pixelDFRow2["TS"], pixelDFRow2["TS2"]))
                pixelDF1.drop(i)
                pixelDF2.drop(j)
    pixel1ToT = np.array(pixel1ToT)
    pixel2ToT = np.array(pixel2ToT)
    return pixel1ToT, pixel2ToT

def calcCorrelation(df):
    correlationArray = np.zeros((372, 372, 7), dtype=int)
    for row1 in range(372):
        for row2 in range(372):
            pixel1ToT, pixel2ToT = findCorrelatedToT(
                df[(df["Row"] == row1)], df[(df["Row"] == row2)]
            )
            correlationTypes = np.array(
                [checkCorrelationType(pixel1ToT[i], pixel2ToT[i]) for i in range(len(pixel1ToT))]
            )
            correlationTypesSummed = np.unique(correlationTypes, return_counts=True)
            if correlationTypesSummed[0].size != 0:
                correlationArray[row1, row2, correlationTypesSummed[0]] += correlationTypesSummed[1]
    return correlationArray

def loadOrCalcCorrelation(dataFile,config,column = 60):
    calcFileManager = calcDataFileManager(config["pathToCalcData"], "correlationArray", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"{dataFile.fileName}",
    )
    fileCheck = calcFileManager.fileExists(calcFileName=calcFileName)
    if not fileCheck:
        df = dataFile.get_dataFrame()
        shortDF = df[(df["Column"] == column) & (df["Layer"] == 4)][:1000]
        correlationArray = calcCorrelation(shortDF)
        calcFileManager.saveFile(calcFileName=calcFileName,array=correlationArray)
    else:
        correlationArray = calcFileManager.loadFile(calcFileName=calcFileName)
    return correlationArray


config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    correlationArray = loadOrCalcCorrelation(dataFile,config)
    for i in range(7):
        plot = plotClass(config["pathToOutput"] + "Correlations/manyPixel/", sizePerPlot=(12, 12))
        axs = plot.axs
        extent = (
            0.5,
            371.5,
            0.5,
            371.5,
        )
        im = axs.imshow(correlationArray[1:, 1:, i], aspect="equal", origin="lower", extent=extent)
        plot.set_config(
            axs,
            title=f"Row Row correlation {dataFile.fileName} type {i}",
            xlabel="Row",
            ylabel="Row",
        )
        axs.xaxis.set_major_locator(MultipleLocator(30))
        axs.xaxis.set_major_formatter("{x:.0f}")
        axs.xaxis.set_minor_locator(MultipleLocator(10))
        axs.yaxis.set_major_locator(MultipleLocator(30))
        axs.yaxis.set_major_formatter("{x:.0f}")
        axs.yaxis.set_minor_locator(MultipleLocator(10))
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("Frequency", rotation=270, labelpad=15)
        plot.saveToPDF(f"{dataFile.fileName}_Row_Row_correlation_{i}")
