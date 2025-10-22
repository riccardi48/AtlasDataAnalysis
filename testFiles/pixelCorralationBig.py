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

def checkHit_vectorized(layer1, ts1, trigger1, layer2, ts2, trigger2, 
                        timeVariance=100, triggerVariance=1):
    """Vectorized version that works on arrays with broadcasting.
    
    Args:
        layer1, ts1, trigger1: shape (N, 1)
        layer2, ts2, trigger2: shape (1, M)
    Returns:
        Boolean array of shape (N, M)
    """
    layer_match = (layer1 == layer2)
    trigger_match = np.abs(trigger1 - trigger2) <= triggerVariance
    
    time_diff = np.abs(ts1 - ts2)
    time_diff_wrapped = np.abs((ts1 + 512) - (ts2 + 512))
    time_match = (time_diff <= timeVariance) | (time_diff_wrapped <= timeVariance)
    
    return layer_match & trigger_match & time_match


def calcToT_vectorized(ts, ts2):
    """Vectorized ToT calculation."""
    return (ts2 * 2 - ts) % 256


def checkCorrelationType_vectorized(tot1, tot2):
    """Vectorized correlation type checking.
    
    Returns integer array with values 0-6 based on ToT thresholds.
    """
    result = np.zeros(tot1.shape, dtype=int)
    
    # Boolean masks for conditions
    tot1_low = tot1 < 30
    tot1_mid = (tot1 >= 30) & (tot1 < 255)
    tot1_high = tot1 >= 255
    
    tot2_low = tot2 < 30
    tot2_mid = (tot2 >= 30) & (tot2 < 255)
    tot2_high = tot2 >= 255
    
    # Apply rules in order (later assignments override earlier ones)
    result[tot1_low & tot2_low] = 1
    result[tot1_low & tot2_mid] = 2
    result[tot1_mid & tot2_low] = 3
    result[tot1_mid & tot2_mid] = 4
    result[tot1_high] = 5
    result[tot2_high] = 6  # This overrides rule 5 if tot2 >= 255
    
    return result


def findCorrelatedToT_vectorized(pixelDF1, pixelDF2, timeVariance=100, triggerVariance=1):
    """Fully vectorized version using numpy broadcasting."""
    if len(pixelDF1) == 0 or len(pixelDF2) == 0:
        return np.array([]), np.array([])
    
    # Extract columns as numpy arrays
    layer1 = pixelDF1["Layer"].values
    ts1 = pixelDF1["TS"].values
    ts2_1 = pixelDF1["TS2"].values
    trigger1 = pixelDF1["TriggerID"].values
    
    layer2 = pixelDF2["Layer"].values
    ts2 = pixelDF2["TS"].values
    ts2_2 = pixelDF2["TS2"].values
    trigger2 = pixelDF2["TriggerID"].values
    
    # Reshape for broadcasting: (N, 1) vs (1, M)
    matches = checkHit_vectorized(
        layer1[:, None], ts1[:, None], trigger1[:, None],
        layer2[None, :], ts2[None, :], trigger2[None, :],
        timeVariance, triggerVariance
    )
    
    # Find matching pairs (greedy: first match for each pixel1)
    pixel1ToT = []
    pixel2ToT = []
    used_j = set()
    
    for i in range(len(pixelDF1)):
        # Find all matches for this pixel1
        match_indices = np.where(matches[i, :])[0]
        
        # Take the first unused match
        for j in match_indices:
            if j not in used_j:
                pixel1ToT.append(calcToT_vectorized(ts1[i], ts2_1[i]))
                pixel2ToT.append(calcToT_vectorized(ts2[j], ts2_2[j]))
                used_j.add(j)
                break
    
    return np.array(pixel1ToT), np.array(pixel2ToT)


def calcCorrelation(df, timeVariance=100, triggerVariance=1):
    """Optimized correlation calculation with full vectorization."""
    correlationArray = np.zeros((372, 372, 7), dtype=int)
    
    # Pre-group by Row for O(1) lookup
    grouped = df.groupby("Row", sort=False)
    row_data = {row: group for row, group in grouped}
    
    # Get all rows that actually exist in the data
    existing_rows = sorted(row_data.keys())
    
    for row1 in existing_rows:
        df_row1 = row_data[row1]
        
        for row2 in existing_rows:
            df_row2 = row_data[row2]
            
            pixel1ToT, pixel2ToT = findCorrelatedToT_vectorized(
                df_row1, df_row2, timeVariance, triggerVariance
            )
            
            if len(pixel1ToT) == 0:
                continue
            
            # Fully vectorized correlation type checking
            correlationTypes = checkCorrelationType_vectorized(pixel1ToT, pixel2ToT)
            
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
        shortDF = df[(df["Column"] == column) & (df["Layer"] == 4)]
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
