import numpy as np
import sys

sys.path.append("..")
from landau import landau
from scipy.stats import linregress
from dataAnalysis import clusterClass


def angle_with_error_mc(orthCharge, error_orthCharge, charge, error_charge, n_samples=10000):
    """
    Monte Carlo error propagation for: angle = 90 - rad2deg(arcsin(orthCharge/charge))
    """
    # Generate random samples from normal distributions
    orthCharge_samples = np.random.normal(orthCharge, error_orthCharge, n_samples)
    charge_samples = np.random.normal(charge, error_charge, n_samples)

    # Calculate ratio
    ratio_samples = orthCharge_samples / charge_samples

    # Only keep valid samples (ratio in [-1, 1])
    valid_mask = np.abs(ratio_samples) <= 1
    valid_ratios = ratio_samples[valid_mask]

    if len(valid_ratios) == 0:
        return np.nan, np.nan

    # Calculate full angle formula for valid samples
    angle_samples = 90 - np.rad2deg(np.arcsin(valid_ratios))

    # Mean and standard deviation
    mean_angle = np.mean(angle_samples)
    error_angle = np.std(angle_samples)
    validity_fraction = len(valid_ratios) / n_samples

    return mean_angle, error_angle


def line(x, slope, intercept):
    return slope * x + intercept


def lineFromTwoPoints(x1, x2, y1, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c


def RowsToMicroMeter(rows):
    return rows * 50  # 50 micrometer per row


def MicroMeterToRows(micrometer):
    return micrometer / 50  # 50 micrometer per row


def ColumnsToMicroMeter(columns):
    return columns * 150  # 150 micrometer per column


def MicroMeterToColumns(micrometer):
    return micrometer / 150  # 150 micrometer per column


def R2MM(rows):
    return rows * 50  # 50 micrometer per row


def MM2R(micrometer):
    return micrometer / 50  # 50 micrometer per row


def C2MM(columns):
    return columns * 150  # 150 micrometer per column


def MM2C(micrometer):
    return micrometer / 150  # 150 micrometer per column


typeDict = {
    "0": "Single Pixel\n",
    "1": "Single Row\n",
    "2": "Single Column\n",
    "3": "Straight\n",
    "4": "Steep Angle\n",
    "5": "Flat Timestamps\n",
    # "6": "Sloped Timestamps\n",
    "7": "Gaps\n",
    "8": "Expected Charge\n",
    "9": "Expected Length\n",
    "a": "Short Length\n",
    "b": "Long Length\n",
}


def characterizeCluster(cluster: clusterClass):
    rows = cluster.getRows(excludeCrossTalk=True).astype(int)
    columns = cluster.getColumns(excludeCrossTalk=True).astype(int)
    TS = cluster.getTSs(True).astype(int)
    cluster.clusterType = ""
    if rows.size == 1:
        cluster.clusterType += "0"
    if np.unique(columns).size == 1:
        cluster.clusterType += "2"
    if np.unique(rows).size == 1:
        cluster.clusterType += "1"
    else:
        lines = findValidLines(rows, columns)
        if len(lines) != 0:
            cluster.clusterType += "3"
            m,c = findBestLine(cluster,lines)
            clusterLength = calcClusterLength(rows, m, c)
            if clusterLength > 10 * 50 and clusterLength < 30 * 50:
                cluster.clusterType += "9"
            elif clusterLength <= 10 * 50:
                cluster.clusterType += "a"
            elif clusterLength >= 30 * 50:
                cluster.clusterType += "b"
            if m < -0.5 or m > 0.5:
                cluster.clusterType += "4"
        if isFlatTimeStamp(TS):
            cluster.clusterType += "5"
        if isCorrectTimeStamps(rows, TS):
            cluster.clusterType += "6"
        if not isConnected(rows, columns):
            cluster.clusterType += "7"
        if cluster.getClusterCharge(True) > 10 and cluster.getClusterCharge(True) < 22:
            cluster.clusterType += "8"
    return cluster.clusterType
    # print(residual)


def calcClusterLength(rows, m, c):
    return np.sqrt(
        (
            line(RowsToMicroMeter(np.max(rows)), m, c)
            - line(RowsToMicroMeter(np.min(rows)), m, c)
        )
        ** 2
        + (RowsToMicroMeter(np.ptp(rows))) ** 2
    )


def isSteepAngle(slope, clusterLength):
    return (slope > 0.5 or slope < -0.5) and clusterLength > 5 * 50


def isFlatTimeStamp(TS):
    relativeTS = TS - np.min(TS)
    return np.max(relativeTS) <= 3


def isCorrectTimeStamps(rows, TS):
    if np.ptp(rows) == 0:
        return False
    relativeRows = rows - np.min(rows)
    relativeTS = TS - np.min(TS)
    slope = np.ptp(relativeTS)/np.ptp(relativeRows)
    #result = linregress(relativeRows, relativeTS, nan_policy="omit")
    return slope < 0 or (np.max(relativeTS) > 10 and slope < -0.2)


def isConnected(rows, columns):
    unused = list(np.arange(rows.size))
    used = []
    used.append(unused[0])
    while len(used) != len(unused):
        indexes = np.array([x for x in unused if x not in used])
        newNeighbors = np.full(indexes.shape, False)
        for i in used:
            newNeighbors = (
                newNeighbors
                | ((abs(rows[i] - rows[indexes]) == 1) & (abs(columns[i] - columns[indexes]) == 0))
                | ((abs(rows[i] - rows[indexes]) == 0) & (abs(columns[i] - columns[indexes]) == 1))
                | ((abs(rows[i] - rows[indexes]) == 1) & (abs(columns[i] - columns[indexes]) == 1))
            )
        newNeighbors = indexes[newNeighbors]
        if newNeighbors.size == 0:
            return False
        used.extend(list(newNeighbors))
    return True


def isType(char, clusterTypes):
    return np.array([True if char in clusterType else False for clusterType in clusterTypes])


def isTypes(string, clusterTypes):
    boolList = np.full(len(clusterTypes), True)
    negative = False
    for char in string:
        if char == "~":
            negative = True
            continue
        if negative:
            negative = False
            boolList = boolList & ~isType(char, clusterTypes)
            continue
        boolList = boolList & isType(char, clusterTypes)
    return boolList


def findValidLines(rows, columns):
    rows = R2MM(rows)
    columns = C2MM(columns)
    if len(np.unique(rows)) == 1:
        return np.array([])
    buffer = 25
    minRow = np.min(rows)
    maxRow = np.max(rows)
    result = linregress(rows,columns)
    y1_list = line(minRow,result.slope,result.intercept) + np.linspace(-buffer, + buffer, 5)
    y2_list = line(maxRow,result.slope,result.intercept) + np.linspace(-buffer, + buffer, 5)
    validLines = []
    for y1 in y1_list:
        for y2 in y2_list:
            if checkValidLine(
                rows, columns, minRow, maxRow, y1, y2
            ):
                m, c = lineFromTwoPoints(minRow, maxRow, y1, y2)
                if not np.isnan(m) and not np.isnan(c):
                    validLines.append((m, c))
    return np.array(validLines)

def checkValidLine(rows, columns, x1, x2, y1, y2):
    maxOverlap = 10  # micrometers
    x, y = closestPointOnLine(rows, columns, x1, x2, y1, y2)
    if np.any(abs(y - columns) > 75 + maxOverlap*2):
        return False
    if np.any(abs(x - rows) > 25 + maxOverlap):
        return False
    return True


def distanceToLine2Points(x, y, x1, x2, y1, y2):
    return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt(
        (y2 - y1) ** 2 + (x2 - x1) ** 2
    )

def distanceToLine(x,y,m,c):
    a = m
    b = -1
    return abs(a*x+b*y+c)/np.sqrt(a**2+b**2)

def closestPointOnLine(x, y, x1, x2, y1, y2):
    dx = x2 - x1
    dy = y2 - y1
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    x_closest = x1 + t * dx
    y_closest = y1 + t * dy
    return x_closest, y_closest

def findBestLine(cluster,lines):
    if len(lines) == 0:
        return (np.nan,np.nan)
    maxOverlap = 10  # micrometers
    rows = R2MM(cluster.getRows(True))
    columns = C2MM(cluster.getColumns(True))
    TSs = cluster.getTSs(True)
    relativeTSs = TSs - np.min(TSs)
    residual = np.zeros(len(lines))
    for i,(m,c) in enumerate(lines):
        distances = distanceToLine(rows,columns,m,c)
        if np.all(distances==0):
            return m,c
        x1 = np.min(rows)
        x2 = np.max(rows)
        y1 = line(x1,m,c)
        y2 = line(x2,m,c)
        x_closest, y_closest = closestPointOnLine(rows, columns, x1, x2, y1, y2)
        residual[i] += np.sum(relativeTSs>10)
        residual[i] -= np.sum((relativeTSs>10) & ((abs(x_closest-rows) > 25-maxOverlap) | (abs(y_closest-columns) > 75-maxOverlap)))
        residual[i] += np.sum(relativeTSs>15)
        residual[i] -= np.sum((relativeTSs>10) & (abs(x_closest-rows) > 25-maxOverlap) & (abs(y_closest-columns) > 75-maxOverlap))
        residual[i] -= np.sum((relativeTSs<=10) & (abs(x_closest-rows) < 25-maxOverlap) | (abs(y_closest-columns) < 75-maxOverlap))
        residual[i] += np.sum((abs(x_closest-rows)/50)[relativeTSs<=10]**2)/3
        residual[i] += np.sum((abs(y_closest-columns)/150)[relativeTSs<=10]**2)/3
        residual[i] += np.sum((abs(x_closest-rows)/50)[relativeTSs>10]**2)/10
        residual[i] += np.sum((abs(y_closest-columns)/150)[relativeTSs>10]**2)/10
    return lines[np.argmin(residual)]
