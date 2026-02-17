from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm,  # tqdm
    norm,  # scipy.stats.norm
)


def isOnePixel(cluster: clusterClass) -> bool:
    return cluster.getRows(excludeCrossTalk=True).size == 1


def isOnEdge(cluster: clusterClass) -> bool:
    return np.any((cluster.getRows(True) <= 1) | (cluster.getRows(True) >= 370)) or np.any(
        (cluster.getColumns(True) <= 1) | (cluster.getColumns(True) >= 130)
    )


def isFlat(cluster: clusterClass) -> bool:
    return np.unique(cluster.getColumns(True)).size == 1


def R2MM(rows):
    return rows * 50  # 50 micrometer per row


def MM2R(micrometer):
    return micrometer / 50  # 50 micrometer per row


def C2MM(columns):
    return columns * 150  # 150 micrometer per column


def MM2C(micrometer):
    return micrometer / 150  # 150 micrometer per column


def findConnectedSections(rows, columns):
    unused = list(np.arange(rows.size))
    used = []
    used.append(unused[0])
    sections = []
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
            sections.append(
                [
                    int(x)
                    for x in used
                    if x not in ([_ for __ in sections for _ in __] if len(sections) > 0 else [])
                ]
            )
            used.append(unused[np.argwhere(~np.isin(unused, used))[0][0]])
        used.extend(list(newNeighbors))
    sections.append(
        [
            int(x)
            for x in used
            if x not in ([_ for __ in sections for _ in __] if len(sections) > 0 else [])
        ]
    )
    return sections

def gaussianBinned(x, mu, sigma, scaler, edges):
    return (gaussianCDFFunc(edges[1:], mu, sigma) - gaussianCDFFunc(edges[:-1], mu, sigma)) * scaler


def gaussianCDFFunc(x, mu, sig):
    return norm.cdf((x - mu) / sig)


def gaussianFunc(x, mu, sig, scaler):
    return norm.pdf(x, mu, sig) * scaler


def filterForTemplate(x, y, estimate, spread):
    index = (x < len(estimate)) & (x >= 0)
    return y[index], estimate[x[index]], spread[x[index]]


def scaleTemplate(estimate, spread, angleScaler, flatScaler):
    if np.isnan(flatScaler) or np.isnan(flatScaler):
        return np.nan, np.nan
    if flatScaler == 1 and angleScaler == 1:
        return estimate, spread
    flatCutOff = np.where(estimate[1:] > 0.5)[0][0] + 1
    midPoint = (flatCutOff - 1) * flatScaler
    endPoint = midPoint + (len(estimate) - flatCutOff) * angleScaler
    flatIndexes = np.linspace(1, midPoint, flatCutOff - 1)
    angleIndexes = np.linspace(midPoint, endPoint, len(estimate) - flatCutOff + 1)[1:]
    Indexes = np.concatenate([[0], flatIndexes, angleIndexes])
    try:
        newEstimate = np.interp(np.arange(int(np.floor(Indexes[-1])) + 1), Indexes, estimate)
    except:
        print(angleScaler)
        print(flatScaler)
        print(midPoint)
        print(endPoint)
        input(f"{Indexes}")
    newSpread = np.interp(np.arange(int(np.floor(Indexes[-1])) + 1), Indexes, spread)
    return newEstimate, newSpread


def scaleOnGaussian(x, mu, sig):
    return (x - mu) / sig
