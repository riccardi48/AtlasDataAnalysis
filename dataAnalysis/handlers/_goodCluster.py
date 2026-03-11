from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm,  # tqdm
)
from ._genericClusterFuncs import isFlat, isOnePixel, isOnEdge


def isGoodCluster(cluster,minExpectedClusterSize=10,lowTS = 1):
    if not isFlat(cluster):
        return False, False
    if isOnePixel(cluster):
        return False, False
    if isOnEdge(cluster):
        return False, False
    relativeRows = cluster.getRows(True) - np.min(cluster.getRows(True))
    if np.ptp(relativeRows) < minExpectedClusterSize+1:
        return False, False
    relativeTS = cluster.getTSs(True) - np.min(cluster.getTSs(True))
    if np.all(relativeRows[1:-1]<=lowTS):
        return False, False
    #if np.mean(np.diff(relativeTS[cluster.section][1:-1])) < 0:
    #    return False
    sortIndexes = np.argsort(relativeRows)
    relativeRows = relativeRows[sortIndexes]
    relativeTS = relativeTS[sortIndexes]
    if np.sum(relativeTS>lowTS)<4:
        return False, False
    if np.sum(relativeTS<=lowTS)<minExpectedClusterSize*1.5:
        return False, False
    #if np.any((np.diff(relativeTS)>10)|(np.diff(relativeTS)<-10)):
    #    return False, False
    flipped = False
    lengthOfFlatSections = int(minExpectedClusterSize-1)
    FirstLength = relativeTS[
        (relativeRows > np.min(relativeRows))
        & (relativeRows <= np.min(relativeRows) + lengthOfFlatSections)
    ]
    LastLength = relativeTS[
        (relativeRows < np.max(relativeRows))
        & (relativeRows >= np.max(relativeRows) - lengthOfFlatSections)
    ]
    if np.any(FirstLength <= lowTS) and np.any(LastLength <= lowTS):
        return False, False
    if (np.all(FirstLength <= lowTS) + np.all(LastLength <= lowTS))%2 == 1:
        flipped = np.all(LastLength <= lowTS)
        return True, flipped
    return False, flipped
