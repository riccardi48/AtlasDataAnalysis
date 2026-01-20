import sys
from .genericFuncs import convertToRelative
sys.path.append("/home/atlas/rballard/AtlasDataAnalysis")
import numpy as np
from tqdm import tqdm

def getPercentFromDict(efficiencyDict,maxWidth=30):
    rowFrequency = np.zeros(maxWidth)
    for row in efficiencyDict["expectedRelativeRows"]:
        rowFrequency[row] += 1
    missingRowFrequency = np.zeros(maxWidth)
    for row in efficiencyDict["missingRelativeRows"]:
        missingRowFrequency[row] += 1
    EfficiencyFrequency = rowFrequency - missingRowFrequency
    rowPercent = EfficiencyFrequency / (rowFrequency + 1e-10)
    errors = getBinomialError(EfficiencyFrequency, rowFrequency, rowPercent)
    return rowPercent,errors

def wilsonError(N, M, z=0):
    p = N / M
    e = (1 / (1 + ((z**2) / M))) * (
        p + ((z**2) / (2 * M)) + z * np.sqrt(((p * (1 - p)) / M) + ((z**2) / (4 * (M**2))))
    )
    return e

def getBinomialError(NList,MList,pList):
    return np.array(
        [
            abs(pList-wilsonError(NList, MList, z=-1)),
            abs(pList-wilsonError(NList, MList, z=1)),
        ]
    )

def genExpectedRows(rows,flipped,maxWidth):
    return np.linspace(
            np.sort(rows)[-1 * flipped],
            np.sort(rows)[-1 * flipped] + maxWidth + maxWidth * 2 * -1 * flipped,
            maxWidth + 1,
        ).astype(int)[:-1]

def calcEfficiency(clusters, maxWidth=30):
    mainDict = {"expectedRows":[],
                "expectedRelativeRows":[],
                "missingRows":[],
                "missingRelativeRows":[],
                }
    for cluster in tqdm(clusters, desc="Finding Efficiency"):
        rows = cluster.getRows(True)[cluster.section]
        columns = cluster.getColumns(True)[cluster.section]
        timestamps = cluster.getTSs(True)[cluster.section]
        expectedRows = genExpectedRows(rows,cluster.flipped,maxWidth)
        if np.any(expectedRows[(expectedRows <= 0) | (expectedRows >= 372)]):
            continue
        expectedRowsRelative, _ = convertToRelative(expectedRows, expectedRows, flipped=False)
        x, _ = convertToRelative(rows, rows, flipped=cluster.flipped)
        missingRelativeRows = np.array([r for r in expectedRowsRelative if r not in x])
        index = (x < maxWidth) & (x >= 0)
        x = x[index]
        if len(missingRelativeRows[missingRelativeRows < maxWidth]) != 0:
            missingRow = np.array([r for r in expectedRows if r not in rows])
            mainDict["missingRows"].extend(missingRow[missingRelativeRows < maxWidth])
            mainDict["missingRelativeRows"].extend(missingRelativeRows[missingRelativeRows < maxWidth])
        mainDict["expectedRows"].extend(expectedRows[expectedRowsRelative < maxWidth])
        mainDict["expectedRelativeRows"].extend(expectedRowsRelative[expectedRowsRelative < maxWidth])
    return mainDict