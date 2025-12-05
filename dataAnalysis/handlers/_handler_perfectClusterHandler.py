from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm,  # tqdm
    curve_fit,  # scipy.optimize.curve_fit
)
from dataAnalysis._fileReader import calcDataFileManager
from ._handler_dataFrameHandler import dataFrameHandler
from ._handler_clusterHandler import clusterHandler
from ._goodCluster import isGoodCluster
from ._genericClusterFuncs import gaussianBinned
from ._perfectCluster import isPerfectCluster


class perfectClusterHandler:
    def __init__(
        self,
        calcFileManager: calcDataFileManager,
        dataFrameHandler: dataFrameHandler,
        clusterHandler: clusterHandler,
    ):
        self.calcFileManager = calcFileManager
        self.dataFrameHandler = dataFrameHandler
        self.clusterHandler = clusterHandler

    def getTimeStampTemplate(
        self, maxRow=25, layers: Optional[list[int]] = None, excludeCrossTalk: bool = True
    ):
        clusters = self.clusterHandler.getClusters(layers=layers, excludeCrossTalk=excludeCrossTalk)
        relativeRowList, relativeTSList = getRelativeRowTS(clusters)
        TSRange = 40
        RowRange = maxRow
        array, yedges, xedges = np.histogram2d(
            relativeRowList,
            relativeTSList,
            range=((-0.5, RowRange + 0.5), (-0.5, TSRange + 0.5)),
            bins=(RowRange + 1, TSRange + 1),
        )
        estimate, spread = calcTemplate(array)
        return estimate, spread

    def getPerfectClusterIndexes(
        self,
        estimate,
        spread,
        minPval=0.5,
        layers: Optional[list[int]] = None,
        excludeCrossTalk: bool = True,
    ):
        clusters = self.clusterHandler.getClusters(layers=layers, excludeCrossTalk=excludeCrossTalk)
        return [
            cluster.getIndex()
            for cluster in tqdm(clusters, desc="Finding perfect clusters")
            if isPerfectCluster(
                cluster, estimate, spread, minPval=minPval, excludeCrossTalk=excludeCrossTalk
            )
        ]
    
       


def getRelativeRowTS(clusters: clusterArray):
    relativeRowList = []
    relativeTSList = []
    for cluster in tqdm(clusters, desc="Calculating perfect cluster template"):
        goodCluster, flipped = isGoodCluster(cluster)
        Timestamps = cluster.getTSs(True)
        relativeTS = Timestamps - np.min(Timestamps)
        if not goodCluster:
            continue
        relativeRows = cluster.getRows(True) - np.min(cluster.getRows(True))
        relativeTS = cluster.getTSs(True) - np.min(cluster.getTSs(True))
        sortIndexes = np.argsort(relativeRows)
        relativeRows = relativeRows[sortIndexes]
        relativeTS = relativeTS[sortIndexes]
        if flipped:
            relativeRows = np.max(relativeRows) - relativeRows
        relativeRowList.extend(relativeRows)
        relativeTSList.extend(relativeTS)
    return np.array(relativeRowList), np.array(relativeTSList)


def calcTemplate(array):
    estimate = []
    spread = []
    for i, x in enumerate(array):
        # print(x)
        bounds = ((-5, 0, 0), (30, 30, np.sum(x) * 2))
        if i > 15:
            popt, pcov = curve_fit(
                gaussianBinned, np.arange(x.size)[2:], x[2:], maxfev=5000, bounds=bounds
            )
        else:
            popt, pcov = curve_fit(gaussianBinned, np.arange(x.size), x, maxfev=5000, bounds=bounds)
        if popt[1] < 1:
            popt[1] = 1
        if popt[0] < 0.5:
            popt[0] = 0.5
        if i == 0:
            popt[1] = 5
        estimate.append(popt[0])
        spread.append(popt[1])
    return np.array(estimate), np.array(spread)
