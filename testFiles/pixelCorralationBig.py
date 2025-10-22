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
    # Pre-extract columns as numpy arrays for faster access
    layer1 = pixelDF1["Layer"].values
    ts1 = pixelDF1["TS"].values
    ts2_1 = pixelDF1["TS2"].values
    trigger1 = pixelDF1["TriggerID"].values
    
    layer2 = pixelDF2["Layer"].values
    ts2 = pixelDF2["TS"].values
    ts2_2 = pixelDF2["TS2"].values
    trigger2 = pixelDF2["TriggerID"].values
    
    pixel1ToT = []
    pixel2ToT = []
    used_idx1 = set()
    used_idx2 = set()
    
    # Use numpy arrays instead of iterrows
    for i in range(len(pixelDF1)):
        if i in used_idx1:
            continue
        for j in range(len(pixelDF2)):
            if j in used_idx2:
                continue
            if checkHit(layer1[i], ts1[i], trigger1[i], 
                       layer2[j], ts2[j], trigger2[j]):
                pixel1ToT.append(calcToT(ts1[i], ts2_1[i]))
                pixel2ToT.append(calcToT(ts2[j], ts2_2[j]))
                used_idx1.add(i)
                used_idx2.add(j)
                break  # Move to next i since this one is matched
    
    return np.array(pixel1ToT), np.array(pixel2ToT)

def calcCorrelation(df):
    correlationArray = np.zeros((372, 372, 7), dtype=int)
    
    # Pre-group by Row for faster filtering
    grouped = df.groupby("Row")
    row_data = {row: group for row, group in grouped}
    
    for row1 in range(372):
        if row1 not in row_data:
            continue
        df_row1 = row_data[row1]
        
        for row2 in range(372):
            if row2 not in row_data:
                continue
            df_row2 = row_data[row2]
            
            pixel1ToT, pixel2ToT = findCorrelatedToT(df_row1, df_row2)
            
            if len(pixel1ToT) == 0:
                continue
            
            # Vectorize correlation type checking
            correlationTypes = np.array([
                checkCorrelationType(pixel1ToT[i], pixel2ToT[i]) 
                for i in range(len(pixel1ToT))
            ])
            
            # Count occurrences
            unique_types, counts = np.unique(correlationTypes, return_counts=True)
            correlationArray[row1, row2, unique_types] += counts
    
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
