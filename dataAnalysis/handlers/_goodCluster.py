from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm,  # tqdm
)
from ._genericClusterFuncs import isFlat, isOnePixel, isOnEdge


def isGoodCluster(cluster):
    if not isFlat(cluster):
        return False, False
    if isOnePixel(cluster):
        return False, False
    if isOnEdge(cluster):
        return False, False
    relativeRows = cluster.getRows(True) - np.min(cluster.getRows(True))
    relativeTS = cluster.getTSs(True) - np.min(cluster.getTSs(True))
    sortIndexes = np.argsort(relativeRows)
    relativeRows = relativeRows[sortIndexes]
    relativeTS = relativeTS[sortIndexes]
    flipped = False
    lengthOfFlatSections = 6
    FirstLength = relativeTS[
        (relativeRows > np.min(relativeRows))
        & (relativeRows < np.min(relativeRows) + lengthOfFlatSections)
    ]
    LastLength = relativeTS[
        (relativeRows < np.max(relativeRows))
        & (relativeRows > np.max(relativeRows) - lengthOfFlatSections)
    ]
    lowTS = 3
    if (np.all(FirstLength <= lowTS) + np.all(LastLength <= lowTS))%2 == 1:
        flipped = np.all(LastLength <= lowTS)
        return True, flipped
    return False, flipped
