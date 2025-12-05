from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm,  # tqdm
    minimize,  # scipy.optimize.minimize
)
from dataAnalysis._fileReader import calcDataFileManager
from ._handler_dataFrameHandler import dataFrameHandler
from ._handler_clusterHandler import clusterHandler
from ._goodCluster import isGoodCluster
from ._genericClusterFuncs import findConnectedSections
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
        self, maxRow=25, layers: Optional[list[int]] = None, excludeCrossTalk: bool = True,minExpectedClusterSize=8,numberOfClustersUsed=100,TSRange=30
    ):
        clusters = self.clusterHandler.getClusters(layers=layers, excludeCrossTalk=excludeCrossTalk)
        relativeRowList, relativeTSList = getRelativeRowTS(clusters,maxRow=maxRow,minExpectedClusterSize=minExpectedClusterSize,numberOfClustersUsed=numberOfClustersUsed,TSRange=TSRange)
        estimate, spread = calcTemplate(relativeRowList[relativeRowList<=maxRow],relativeTSList[relativeRowList<=maxRow])
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
    
       


def getRelativeRowTS(clusters: clusterArray,minExpectedClusterSize=8,numberOfClustersUsed=100,TSRange=30,maxRow=25):
    relativeRowList = []
    relativeTSList = []
    k = 0
    for cluster in clusters:
        goodCluster, flipped = isGoodCluster(cluster,minExpectedClusterSize=minExpectedClusterSize)
        Timestamps = cluster.getTSs(True)
        relativeTS = Timestamps - np.min(Timestamps)
        if not goodCluster:
            continue
        if np.any(relativeTS>=TSRange):
            continue
        if len(findConnectedSections(cluster.getRows(True),cluster.getColumns(True))) != 1:
            continue
        relativeRows = cluster.getRows(True) - np.min(cluster.getRows(True))
        if np.ptp(relativeRows)>=maxRow*1.25:
            continue
        relativeTS = cluster.getTSs(True) - np.min(cluster.getTSs(True))
        sortIndexes = np.argsort(relativeRows)
        relativeRows = relativeRows[sortIndexes]
        relativeTS = relativeTS[sortIndexes]
        if flipped:
            relativeRows = np.max(relativeRows) - relativeRows
        relativeRowList.extend(relativeRows)
        relativeTSList.extend(relativeTS)
        k += 1
        if k >= numberOfClustersUsed:
            break
    return np.array(relativeRowList), np.array(relativeTSList)


def calcTemplate(relativeRowList,relativeTSList):
    estimate = []
    spread = []
    for i in np.sort(np.unique(relativeRowList)):
        TSsToFit = relativeTSList[np.where(relativeRowList==i)[0]]
        if len(estimate)>5 and estimate[-1]>2:
            TSsToFit=TSsToFit[TSsToFit>1]
        mu, sigma, mu_e, sigma_e = fitGaussian(TSsToFit)
        """
        if i > 15:
            popt, pcov = curve_fit(
                gaussianBinned, np.arange(x.size)[2:], x[2:], maxfev=5000, bounds=bounds
            )
        else:
            popt, pcov = curve_fit(gaussianBinned, np.arange(x.size), x, maxfev=5000, bounds=bounds)
        """
        if sigma < 1:
            sigma = 1
        if mu < 0.2:
            mu = 0.2
        #if i == 0:
        #    mu = 5
        estimate.append(mu)
        spread.append(sigma)
    return np.array(estimate), np.array(spread)

def nll(data,params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        return np.sum( np.log(sigma*np.sqrt(2*np.pi)) + (data - mu)**2/(2*sigma**2) )

def fitGaussian(data):
    mu0 = np.mean(data)
    sigma0 = np.std(data)
    func = lambda params:nll(data,params)
    result = minimize(func, x0=[mu0, sigma0], method='L-BFGS-B', bounds=[(None, None), (1e-9, None)])
    mu, sigma = result.x
    if result.hess_inv is not None:
        cov = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else result.hess_inv
        mu_e = np.sqrt(cov[0,0])
        sigma_e = np.sqrt(cov[1,1])
    else:
        mu_e = np.nan
        sigma_e = np.nan
    return mu, sigma, mu_e, sigma_e

