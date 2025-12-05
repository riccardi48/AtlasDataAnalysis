import sys

sys.path.append("..")

import numpy as np
from scipy.stats import chi2
from itertools import combinations
from dataAnalysis._fileReader import calcDataFileManager
import scipy.optimize as optimize


def R2MM(rows):
    return rows * 50  # 50 micrometer per row


def MM2R(micrometer):
    return micrometer / 50  # 50 micrometer per row


def C2MM(columns):
    return columns * 150  # 150 micrometer per column


def MM2C(micrometer):
    return micrometer / 150  # 150 micrometer per column


def isFlat(cluster):
    return np.unique(cluster.getColumns(True)).size == 1


def gaussianBinned(x, mu, sigma, scaler):
    width = x[1]-x[0]
    edges = np.append(x - width/2,x[-1] + width/2)
    return (gaussianCDFFunc(edges[1:],mu,sigma)-gaussianCDFFunc(edges[:-1],mu,sigma))*scaler

from scipy.stats import norm

def gaussianCDFFunc(x,mu,sig):
    return norm.cdf((x-mu)/sig)

def gaussianFunc(x,mu,sig,scaler):
    return norm.pdf(x,mu,sig)*scaler

def addPerfectClusterInfo(dataFile,config,layer=4):
    pVals,flippeds,perms,sections = loadSectionsOrCalc(dataFile,config,layer=4)
    clusters = dataFile.get_clusters(excludeCrossTalk=True,layer=layer)
    for i,cluster in enumerate(clusters):
        cluster.pVal = pVals[i]
        cluster.flipped = flippeds[i]
        cluster.perm = perms[i]
        cluster.section = sections[i]

def loadSectionsOrCalc(dataFile,config,layer=4):
    estimate,spread = getTemplate(config)

    calcFileManager = calcDataFileManager(config["pathToCalcData"], "PRS", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"{dataFile.fileName}",
    )
    fileCheck = calcFileManager.fileExists(calcFileName=calcFileName)
    if not fileCheck:
        clusters = dataFile.get_clusters(excludeCrossTalk=True,layer=layer)
        clusterList = np.array([isPerfectCluster(cluster,estimate,spread,minPval=0.5,excludeCrossTalk=True) for cluster in clusters])
        pValsList = np.array([cluster.pVal for cluster in clusters[clusterList]])
        flippedList = np.array([cluster.flipped for cluster in clusters[clusterList]])
        permList = np.array([cluster.perm for cluster in clusters[clusterList]])
        sectionList = np.array([cluster.section for cluster in clusters[clusterList]],dtype=object)
        calcFileManager.saveFile(calcFileName=calcFileName,array=np.array([pValsList,flippedList,permList,sectionList],dtype=object))
    else:
        pValsList,flippedList,permList,sectionList = calcFileManager.loadFile(calcFileName=calcFileName)
    return pValsList,flippedList,permList,sectionList

def getTemplate(config):
    calcFileManager = calcDataFileManager(config["pathToCalcData"], "TSParams", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"angle6_4Gev_kit_2",
    )
    estimate,spread = calcFileManager.loadFile(calcFileName=calcFileName)
    return estimate,spread

def isPerfectCluster(cluster,estimate,spread,minPval=0.5,excludeCrossTalk=True):
    cluster.pVal = 0.0
    cluster.flipped = False
    cluster.perm = ()
    cluster.section = []
    if not isFlat(cluster):
        return False
    if isOnePixel(cluster):
        return False
    if isOnEdge(cluster):
        return False
    if cluster.getSize(excludeCrossTalk=excludeCrossTalk) <= 8:
        return False
    addClusterValues(cluster,estimate,spread,minPval=0.5,excludeCrossTalk=True)
    relativeTS = abs(cluster.getTSs(excludeCrossTalk=excludeCrossTalk) - np.max(cluster.getTSs(excludeCrossTalk=excludeCrossTalk)))
    if np.all(relativeTS<=4):
        return False
    if cluster.pVal < minPval:
        return False
    if np.all(relativeTS[cluster.section]<=4):
        return False
    relativeRows = abs(cluster.getRows(excludeCrossTalk=excludeCrossTalk)[cluster.section] - np.max(cluster.getRows(excludeCrossTalk=excludeCrossTalk)[cluster.section]))
    if np.ptp(relativeRows) < np.where(estimate[1:]>0.5)[0][0]+1:
        return False
    return True

def isOnePixel(cluster):
    return cluster.getRows(excludeCrossTalk=True).size == 1

def isOnEdge(cluster):
    return np.any((cluster.getRows(True) <= 0) | (cluster.getRows(True) >= 371)) or np.any((cluster.getColumns(True) <= 0) | (cluster.getColumns(True) >= 131))

def addClusterValues(cluster,estimate,spread,minPval=0.5,excludeCrossTalk=True):
    cluster.pVal = 0.0
    cluster.flipped = False
    cluster.perm = ()
    cluster.section = []
    sections = findSections(cluster,excludeCrossTalk=excludeCrossTalk)
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
        pVal,flipped = logLikeOfSection(cluster,sections[0],estimate,spread,excludeCrossTalk=excludeCrossTalk)
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
                if len(section) <= 5:
                    continue
                if np.all((cluster.getTSs(excludeCrossTalk=True)[section]-np.min(cluster.getTSs(excludeCrossTalk=True)[section]))<=4):
                    continue
                pVal,flipped = logLikeOfSection(cluster,section,estimate,spread,excludeCrossTalk=excludeCrossTalk)
                if (pVal > max_pVal and len(perm)==len(max_perm)) or (len(perm)>len(max_perm) and pVal > minPval and pVal != max_pVal):
                    max_pVal = pVal
                    max_flipped = flipped
                    max_perm = perm
        pVal = max_pVal
        flipped = max_flipped
        perm = max_perm
    return pVal,flipped,perm


def findSections(cluster,excludeCrossTalk=True):
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)
    Columns = cluster.getColumns(excludeCrossTalk=excludeCrossTalk)
    sections = findConnectedSections(Rows,Columns)
    return sections

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
            sections.append([int(x) for x in used if x not in ([_ for __ in sections for _ in __] if len(sections) > 0 else [])])
            used.append(unused[np.argwhere(~np.isin(unused,used))[0][0]])
        used.extend(list(newNeighbors))
    sections.append([int(x) for x in used if x not in ([_ for __ in sections for _ in __] if len(sections) > 0 else [])])
    return sections

def logLikeOfSection(cluster,section,estimate,spread,excludeCrossTalk=True):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)[section]
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)[section]
    x,y = convertRowsForFit(Rows,Timestamps,flipped=False)
    pVal1 = gaussian_loglike_pval(scaleOnGaussian(*filterForTemplate(x,y,estimate,spread)))
    if np.sum((x<len(estimate))&(x>=0)) <= 5 or np.sum((x>len(estimate))|(x<0)) > 3:
        pVal1 = 0
    x,y = convertRowsForFit(Rows,Timestamps,flipped=True)
    pVal2 = gaussian_loglike_pval(scaleOnGaussian(*filterForTemplate(x,y,estimate,spread)))
    if np.sum((x<len(estimate))&(x>=0)) <= 5 or np.sum((x>len(estimate))|(x<0)) > 3:
        pVal2 = 0
    pVal = np.max([pVal1,pVal2])
    flipped = pVal2>pVal1
    return pVal,flipped

def scaleOnGaussian(x,mu,sig):
    return (x-mu)/sig

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
        x = np.flip(x)
        y = np.flip(y)
    return x,y

def convertToRelative(Rows,values,flipped):
    x = Rows - np.min(Rows)
    sortIndexes = np.argsort(x)
    x = x[sortIndexes]
    values = values[sortIndexes]
    x = x - np.min(x)
    if flipped:
        x = -x+x[-1]
        x = np.flip(x)
        values = np.flip(values)
    return x,values

def gaussian_loglike_pval(data):
    n = len(data)
    S_obs = np.sum(data**2)
    pval = 1 - chi2.cdf(S_obs, df=n-1)  # exact p-value
    return pval

def logLike_gaussian(data, mu0=0, sigma0=1):
    n = len(data)
    ss = np.sum((data - mu0)**2)
    return -0.5*n*np.log(2*np.pi) - n*np.log(sigma0) - 0.5*ss/(sigma0**2)

def fitTemplate(cluster,section,estimate,spread,excludeCrossTalk=True):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)[section]
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)[section]
    x,y = convertRowsForFit(Rows,Timestamps,flipped=False)
    func = getFuncForMinimize(x,y,estimate,spread)
    initial_guess = [0.5, 0.5]
    bounds = ((0.2,2),(0.2,2))
    #result1 = optimize.differential_evolution(func,bounds=bounds)
    p1,f1,dict1 = optimize.fmin_l_bfgs_b(func, initial_guess,bounds=bounds, approx_grad=True)#, epsilon=0.1,pgtol=1e-07)
    #if dict1["task"] == 'ABNORMAL: ':
    #    f1 = 1
    x,y = convertRowsForFit(Rows,Timestamps,flipped=True)
    func = getFuncForMinimize(x,y,estimate,spread)
    #result2 = optimize.differential_evolution(func,bounds=bounds)
    p2,f2,dict2 = optimize.fmin_l_bfgs_b(func, initial_guess,bounds=bounds, approx_grad=True)#, epsilon=0.1,pgtol=1e-07)
    #if dict2["task"] == 'ABNORMAL: ':
    #    f2 = 1
    #if result1.fun>result2.fun:
    #    return result2.x,True
    if f1>f2:
        return p2,True
    return p1,False

def getFuncForMinimize(x,y,estimate,spread):
    func = lambda params:funcForMinimizing(params,x,y,estimate,spread)
    return func

def funcForMinimizing(params,x,y,estimate,spread):
    newEstimate,newSpread = scaleTemplate(estimate,spread,params[0],params[1])
    y_filtered,estimate_filtered,spread_filtered = filterForTemplate(x,y,newEstimate,newSpread)
    if len(y_filtered) <= 5:
        return 2**16-1
    return -logLike_gaussian(scaleOnGaussian(y_filtered,estimate_filtered,spread_filtered))/(len(y_filtered)-4)

def filterForTemplate(x,y,estimate,spread):
    index = (x<len(estimate))&(x>=0)
    return y[index],estimate[x[index]],spread[x[index]]

def scaleTemplate(estimate,spread,angleScaler,flatScaler):
    if np.isnan(flatScaler) or np.isnan(flatScaler):
        return np.nan,np.nan
    if flatScaler == 1 and angleScaler == 1:
        return estimate,spread
    flatCutOff = np.where(estimate[1:]>0.5)[0][0]+1
    midPoint = (flatCutOff-1)*flatScaler
    endPoint = midPoint+(len(estimate)-flatCutOff)
    flatIndexes = np.linspace(1,midPoint,flatCutOff-1)
    angleIndexes = np.linspace(midPoint,endPoint,len(estimate)-flatCutOff+1)[1:]
    Indexes = np.concatenate([[0],flatIndexes, angleIndexes])*angleScaler
    try:
        newEstimate = np.interp(np.arange(int(np.floor(Indexes[-1]))+1),Indexes,estimate)
    except:
        print(angleScaler)
        print(flatScaler)
        print(midPoint)
        print(endPoint)
        input(f"{Indexes}")
    newSpread = np.interp(np.arange(int(np.floor(Indexes[-1]))+1),Indexes,spread)
    return newEstimate,newSpread

def loadOrCalcMPV(dataFile,config):
    calcFileManager = calcDataFileManager(config["pathToCalcData"], "MPV_Params", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"{dataFile.fileName}",
    )
    fileCheck = calcFileManager.fileExists(calcFileName=calcFileName)
    if not fileCheck:
        raise NotImplementedError
    else:
        paramsList = calcFileManager.loadFile(calcFileName=calcFileName)
    return paramsList

def fitTemplate2(cluster,estimate,spread,excludeCrossTalk=True,minPval=0.2,angleList=np.linspace(0.1,1.4,27),flatList=np.linspace(0.2,1.4,13)):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)
    x,y = convertRowsForFit(Rows,Timestamps,flipped=False)
    sections = findConnectedSections(cluster.getRows(excludeCrossTalk=excludeCrossTalk),cluster.getColumns(excludeCrossTalk=excludeCrossTalk))
    angleScalerList = []
    flatScalerList = []
    pValList = []
    flippedList = []
    permList = []
    for angleScaler in angleList:
        for flatScaler in flatList:
            newEstimate,newSpread = scaleTemplate(estimate,spread,angleScaler,flatScaler)
            pVal,flipped,perm = findBestSections(cluster,sections,newEstimate,newSpread,minPval=minPval,excludeCrossTalk=excludeCrossTalk)
            if pVal > minPval:
                flatScalerList.append(flatScaler)
                angleScalerList.append(angleScaler)
                pValList.append(pVal)
                flippedList.append(flipped)
                permList.append(perm)
    return np.array(flatScalerList),np.array(angleScalerList),np.array(pValList),np.array(flippedList),permList
