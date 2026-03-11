from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm,  # tqdm
    minimize,  # scipy.optimize.minimize
    curve_fit,  # scipy.optimize.curve_fit
)
from dataAnalysis._fileReader import calcDataFileManager
from ._handler_dataFrameHandler import dataFrameHandler
from ._handler_clusterHandler import clusterHandler
from ._goodCluster import isGoodCluster
from ._genericClusterFuncs import findConnectedSections,gaussianBinned,logGaussianBinned,logGaussian
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
        self, maxRow=25, layers: Optional[list[int]] = None, excludeCrossTalk: bool = True,minExpectedClusterSize=8,numberOfClustersUsed=1000,TSRange=30
    ):
        clusters = self.clusterHandler.getClusters(layers=layers, excludeCrossTalk=excludeCrossTalk)
        relativeRowList, relativeTSList = getRelativeRowTS(clusters,maxRow=maxRow,minExpectedClusterSize=minExpectedClusterSize,numberOfClustersUsed=numberOfClustersUsed,TSRange=TSRange)
        estimate, spread, estimate_e, spread_e = calcTemplate(relativeRowList[relativeRowList<=maxRow],relativeTSList[relativeRowList<=maxRow])
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
    for cluster in clusters[1000:]:
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
    estimate_e = []
    spread_e = []

    for i in np.sort(np.unique(relativeRowList)):
        TSsToFit = relativeTSList[np.where(relativeRowList==i)[0]]
        if i == 0:
            maxNumber = len(TSsToFit)
        if len(TSsToFit) < maxNumber/10:
            break
        if np.sum(TSsToFit<=1)/TSsToFit.size > 0.95:
            sigma = 0
            mu = 0
            mu_e = 0
            sigma_e = 0
        else:
            mu, sigma, mu_e, sigma_e = fitHistLogGaussian(TSsToFit)
        """
        if i > 15:
            popt, pcov = curve_fit(
                gaussianBinned, np.arange(x.size)[2:], x[2:], maxfev=5000, bounds=bounds
            )
        else:
            popt, pcov = curve_fit(gaussianBinned, np.arange(x.size), x, maxfev=5000, bounds=bounds)
        """
        estimate.append(mu)
        spread.append(sigma)
        estimate_e.append(mu_e)
        spread_e.append(sigma_e)

    return np.array(estimate), np.array(spread),np.array(estimate_e), np.array(spread_e)

def nll(data,params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        return np.sum( np.log(sigma*np.sqrt(2*np.pi)) + (data - mu)**2/(2*sigma**2) )

def _fitGaussian(data):
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

def fitGaussian(data):
    bins = np.max(data)+1
    _range = [-0.5,np.max(data)+0.5]
    height, x = np.histogram(data, bins=bins, range=_range)
    binCentres = (x[:-1] + x[1:]) / 2
    cut = np.mean(binCentres[np.where(height==np.max(height))[0]])+2
    popt, pcov = fit(height, x, cut, binCentres)
    mu,sigma = popt
    mu_e,sigma_e = np.sqrt(np.diag(pcov))
    if mu < cut-1 and cut-1 > 2:
        bins = (bins-np.max(data)%2)/2+0.5
        _range = [_range[0],_range[1]+np.max(data+1)%2]
        height, x = np.histogram(data, bins=int(bins), range=_range)
        binCentres = (x[:-1] + x[1:]) / 2
        cut = np.mean(binCentres[np.where(height==np.max(height))[0]])+2
        popt, pcov = fit(height, x, cut, binCentres)
        mu,sigma = popt
        mu_e,sigma_e = np.sqrt(np.diag(pcov))
    return mu, sigma, mu_e, sigma_e

def fit(height, x, cut, binCentres):
    func = lambda _x,mu,sigma : gaussianBinned(_x, mu, sigma, np.sum(height),x[x<=cut+((x[1]-x[0])/2)])
    popt, pcov = curve_fit(
        func,
        binCentres[binCentres<=cut],
        height[binCentres<=cut],
    )
    return popt,pcov

def fitHistLogGaussian(data):
    binWidth = 1
    bins = int(np.ptp(data)/binWidth+1)
    data = data + 0.5
    _range = (0,np.max(data)+0.5)
    height,edges = np.histogram(data[np.invert(np.isnan(data))],bins=bins,range=_range)
    func = lambda x, mu, sigma: logGaussianBinned(
        x, mu, sigma, np.sum(height), edges
    )

    binCentres = (edges[:-1] + edges[1:]) / 2
    bounds = ((0, 0.2), (100, 10))
    popt, pcov = curve_fit(
        func,
        binCentres,
        height,
        maxfev=10000,
        bounds=bounds
    )
    mu,sigma = popt
    mu_e,sigma_e = np.sqrt(np.diag(pcov))
    return mu, sigma, mu_e, sigma_e