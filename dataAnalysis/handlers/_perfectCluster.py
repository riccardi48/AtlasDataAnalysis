from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm,  # tqdm
    chi2, # scipy.stats.chi2
)
from ._genericClusterFuncs import isFlat, isOnePixel, isOnEdge, findConnectedSections, filterForTemplate
from itertools import combinations

def isPerfectCluster(cluster: clusterClass,estimate,spread,minPval=0.2,excludeCrossTalk=True):
    if not isFlat(cluster):
        return False
    if isOnePixel(cluster):
        return False
    if isOnEdge(cluster):
        return False
    if cluster.getSize(excludeCrossTalk=excludeCrossTalk) <= 3:
        return False
    addClusterValues(cluster,estimate,spread,minPval=0.5,excludeCrossTalk=True)
    relativeTS = abs(cluster.getTSs(excludeCrossTalk=excludeCrossTalk) - np.max(cluster.getTSs(excludeCrossTalk=excludeCrossTalk)))
    if cluster.pVal < minPval:
        return False
    if np.all(relativeTS[cluster.section][1:-1]<=3):
        return False
    relativeRows = abs(cluster.getRows(excludeCrossTalk=excludeCrossTalk)[cluster.section] - np.max(cluster.getRows(excludeCrossTalk=excludeCrossTalk)[cluster.section]))
    if np.ptp(relativeRows) < np.where(estimate[1:]>0.5)[0][0]+1:
        return False
    if not cluster.flipped:
        return False
    return True

def addClusterValues(cluster,estimate,spread,minPval=0.5,excludeCrossTalk=True):
    sections = findConnectedSections(cluster.getRows(excludeCrossTalk=excludeCrossTalk), cluster.getColumns(excludeCrossTalk=excludeCrossTalk))
    pVal,flipped,perm = findBestSections(cluster,sections,estimate,spread,minPval=minPval,excludeCrossTalk=excludeCrossTalk)
    section = []
    for i in perm:
        section.extend(sections[int(i)])
    cluster.pVal = pVal
    cluster.flipped = flipped
    cluster.perm = perm
    cluster.section = section

def findBestSections(cluster,sections,estimate,spread,minPval=0.2,excludeCrossTalk=True):
    if len(sections) == 1:
        pVal,flipped = pValOfSection(cluster,sections[0],estimate,spread,excludeCrossTalk=excludeCrossTalk)
        perm = (0,)
    else:
        max_pVal = -1
        max_flipped = False
        max_perm = ()
        for l in np.arange(1,len(sections)+1):
            for perm in combinations(range(0,len(sections)),l):
                section = []
                for i in perm:
                    section.extend(sections[int(i)])
                if len(section) <= 3:
                    continue
                pVal,flipped = pValOfSection(cluster,section,estimate,spread,excludeCrossTalk=excludeCrossTalk)
                if (pVal > max_pVal and len(perm)==len(max_perm)) or (len(perm)>len(max_perm) and pVal > minPval and pVal != max_pVal):
                    max_pVal = pVal
                    max_flipped = flipped
                    max_perm = perm
        pVal = max_pVal
        flipped = max_flipped
        perm = max_perm
    return pVal,flipped,perm

def pValOfSection(cluster,section,estimate,spread,excludeCrossTalk=True):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)[section]
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)[section]
    pVals = []
    for _flipped in [False,True]:
        x,y = convertRowsForFit(Rows,Timestamps,flipped=_flipped)
        y, _estimate, _spread = filterForTemplate(x,y,estimate,spread)
        _y = np.zeros(len(y))
        _y[_estimate!=0] = scaleOnLogGaussian(y[_estimate!=0]+0.5, _estimate[_estimate!=0], _spread[_estimate!=0])
        _y[_estimate==0] = scaleOnGaussian(y[_estimate==0]+0.5, 1, 1)
        if np.sum((x<len(estimate))&(x>=0)) <= 3 or np.sum((x>len(estimate))|(x<0)) > 30 or len(_y[_estimate!=0]) < 3:
            pVals.append(0)
        #elif np.sum(_y[_estimate!=0]>0)>len(_y[_estimate!=0])*0.75 or np.sum(_y[_estimate!=0]<0)>len(_y[_estimate!=0])*0.75:
        #    pVals.append(0)
        else:
            pVals.append(gaussian_loglike_pval(_y[_estimate!=0],df=len(_y[_estimate!=0])-1))
    pVal = np.max([pVals])
    flipped = bool(np.argmax([pVals]))
    return pVal,flipped

def convertRowsForFit(Rows,Timestamps,flipped):
    x = Rows - np.min(Rows)
    y = Timestamps - np.min(Timestamps)
    sortIndexes = np.argsort(x)
    x = x[sortIndexes]
    y = y[sortIndexes]
    y = y - np.min(y)
    x = x - np.min(x)
    if flipped:
        x = -x+x[-1]
    return x,y

def gaussian_loglike_pval(data,df=None):
    n = len(data)
    S_obs = np.sum(data**2)
    if df is None:
        df = n-1
    pval = 1 - chi2.cdf(S_obs, df=n-1)  # exact p-value
    return pval

def scaleOnLogGaussian(x, mu, sig):
    return (np.log(x) - np.log(mu)) / sig

def scaleOnGaussian(x, mu, sig):
    return (x - mu) / sig
