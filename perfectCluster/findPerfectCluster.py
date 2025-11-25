import sys
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from scipy.optimize import curve_fit
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
from dataAnalysis._fileReader import calcDataFileManager
from scipy.stats import norm
from scipy.stats import kstest
from scipy.stats import chi2
from itertools import combinations
from pixelCharges.plotCluster import clusterPlotter
from matplotlib.colors import LogNorm


def quad(x,a,b,c):
    return a*x**2+b*x+c

def TSFunc(x,angleScaler):
    z = [0.06053199,-0.005201812,0.4074863964]
    return quad(x*angleScaler,*z)

def perfectClusterTSFunc(x,angleScaler,endPoint,firstPixel):
    x = np.reshape(x,np.shape(x))
    y = np.zeros(np.shape(x))
    y[x<=endPoint] = 0.5
    y[x==0] = firstPixel
    y[x>endPoint] = TSFunc(x[x>endPoint]-endPoint,angleScaler)
    return y

def flipIfNeeded():
    sortIndexes = np.argsort(relativeRows)
    relativeRows = relativeRows[sortIndexes]
    relativeTS = relativeTS[sortIndexes]
    gaps = np.diff(relativeRows)
    if np.any(gaps>5):
        gap = np.where(gaps>5)[0][0] + 1
        if gap > relativeRows.size/2:
            relativeTS = relativeTS[:gap]
            relativeRows = relativeRows[:gap]
        else:
            relativeTS = relativeTS[gap:]
            relativeRows = relativeRows[gap:]
    if np.all(relativeTS[-5:-1]<=2) and not np.all(relativeTS[1:5]<=2):
        relativeTS = np.flip(relativeTS)
        relativeRows = np.flip(relativeRows)

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

def isFlat(cluster):
    return np.unique(cluster.getColumns(True)).size == 1

def isOnePixel(cluster):
    return cluster.getRows(excludeCrossTalk=True).size == 1

def isOnEdge(cluster):
    return np.any((cluster.getRows(True) <= 0) | (cluster.getRows(True) >= 371)) or np.any((cluster.getColumns(True) <= 0) | (cluster.getColumns(True) >= 131))

def isFlipped(TS):
    return np.all(TS[-4:-2] <= 1) and not np.all(TS[1:3] <= 1)

def logLike_gaussian(data, mu0=0, sigma0=1):
    n = len(data)
    ss = np.sum((data - mu0)**2)
    return -0.5*n*np.log(2*np.pi) - n*np.log(sigma0) - 0.5*ss/(sigma0**2)

def gaussian_loglike_pval(data):
    n = len(data)
    S_obs = np.sum(data**2)
    pval = 1 - chi2.cdf(S_obs, df=n-1)  # exact p-value
    return pval

def gaussianCDFFunc(x,mu,sig):
    return norm.cdf((x-mu)/sig)

def gaussianFunc(x,mu,sig,scaler):
    return norm.pdf(x,mu,sig)*scaler

def findSections(cluster,excludeCrossTalk=True):
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)
    Columns = cluster.getColumns(excludeCrossTalk=excludeCrossTalk)
    sections = findConnectedSections(Rows,Columns)
    return sections
def logLikeOfSection(cluster,section,estimate,spread,excludeCrossTalk=True):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)[section]
    y = Timestamps - np.min(Timestamps)
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)[section]
    x = Rows - np.min(Rows)
    sortIndexes = np.argsort(x)
    x = x[sortIndexes]
    y = y[sortIndexes]
    y = y - np.min(y)
    x = x - np.min(x)
    index = (x<len(estimate))&(x>=0)#&(y>1)
    temp_y = y[index]
    temp_estimate = estimate[x[index]]
    temp_spread = spread[x[index]]
    pVal1 = gaussian_loglike_pval((temp_y-temp_estimate)/temp_spread)
    if np.sum((x<len(estimate))&(x>=0)) <= 5 or np.sum((x>len(estimate))|(x<0)) > 3:
        pVal1 = 0
    x = -x+x[-1]
    index = (x<len(estimate))&(x>=0)#&(y>1)
    temp_y = y[index]
    temp_estimate = estimate[x[index]]
    temp_spread = spread[x[index]]
    pVal2 = gaussian_loglike_pval((temp_y-temp_estimate)/temp_spread)
    if np.sum((x<len(estimate))&(x>=0)) <= 5 or np.sum((x>len(estimate))|(x<0)) > 3:
        pVal2 = 0
    pVal = np.max([pVal1,pVal2])
    flipped = pVal2>pVal1
    return pVal,flipped

def getPvalue(cluster,section,estimate,spread,flipped,excludeCrossTalk = True):
    if len(section) == 0:
        return 0
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)[section]
    y = Timestamps - np.min(Timestamps)
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)[section]
    x = Rows - np.min(Rows)
    sortIndexes = np.argsort(x)
    x = x[sortIndexes]
    y = y[sortIndexes]
    y = y - np.min(y)
    x = x - np.min(x)
    if flipped:
        x = -x+x[-1]
    return gaussian_loglike_pval((y[x<len(estimate)]-estimate[x[x<len(estimate)]])/spread[x[x<len(estimate)]])

def graphTSonRows(cluster,section,estimate,spread,flipped,path,excludeCrossTalk=True):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)
    plot = plotClass(path)
    axs = plot.axs
    relativeRows = Rows-np.min(Rows)
    relativeTS = Timestamps - np.min(Timestamps)
    mask = np.ones(len(relativeRows), dtype=bool)
    mask[section,] = False
    x = relativeRows
    if flipped:
        x = abs(x-np.max(x))
    axs.scatter(x[~mask], relativeTS[~mask], color=plot.colorPalette[2], marker="x",label="In Fitted Sections")
    axs.scatter(x[mask], relativeTS[mask], color=plot.colorPalette[8], marker="x",label="Not in Fitted Sections")
    if len(section) != 0:
        if flipped:
            anchor = np.argmax(relativeRows[section])
            rowsForFunc = relativeRows[section][anchor]-relativeRows
        else:
            anchor = np.argmin(relativeRows[section])
            rowsForFunc = relativeRows-relativeRows[section][anchor]
        sortIndexes = np.argsort(relativeRows)
        rowsForFunc = rowsForFunc[sortIndexes]
        relativeRows = relativeRows[sortIndexes]
        rowsToGraph = np.linspace(relativeRows[rowsForFunc==0][0],relativeRows[rowsForFunc==0][0]+len(estimate)+(len(estimate)*-int(flipped)*2),len(estimate))
        if flipped:
            rowsToGraph = abs(rowsToGraph-np.max(rowsToGraph))
        axs.plot(rowsToGraph,estimate,color=plot.colorPalette[0],linestyle="dashed",label=f"Expected TS\np = {getPvalue(cluster,section,estimate,spread,flipped,excludeCrossTalk=excludeCrossTalk):.4f}")
        axs.fill_between(rowsToGraph,estimate-spread,estimate+spread, alpha=0.2,color=plot.colorPalette[0])
    plot.set_config(axs,
        title="Relative TS in cluster",
        xlabel="Relative Row",
        ylabel="Relative TS",
        legend=True,
        ylim=(-0.5,np.max(relativeTS)+5),
        xlim=(np.min(relativeRows)-1,np.max(relativeRows)+1),
    )  
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1))
    plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Row_vs_RelativeTS")

def graphGaussNorm(cluster,section,estimate,spread,flipped,path,excludeCrossTalk=True):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)
    plot = plotClass(path)
    axs = plot.axs
    relativeRows = Rows-np.min(Rows)
    relativeTS = Timestamps - np.min(Timestamps)
    if flipped:
        anchor = np.argmax(relativeRows[section])
        rowsForFunc = relativeRows[section][anchor]-relativeRows
    else:
        anchor = np.argmin(relativeRows[section])
        rowsForFunc = relativeRows-relativeRows[section][anchor]
    sortIndexes = np.argsort(relativeRows)
    rowsForFunc = rowsForFunc[sortIndexes]
    relativeTS = relativeTS[sortIndexes]
    mask = np.ones(len(relativeRows), dtype=bool)
    mask[section,] = False
    index = (rowsForFunc<len(estimate))&(rowsForFunc>0)
    y = (relativeTS[index]-estimate[rowsForFunc[index]])/spread[rowsForFunc[index]]
    axs.scatter(y, np.zeros(y.size)+0.1, color=plot.colorPalette[2], marker="x",label="Cluster TS")
    axs.plot(np.linspace(-4,4,100),gaussianFunc(np.linspace(-4,4,100),0,1,1), color=plot.colorPalette[0],label="Normal")
    plot.set_config(axs,
        title="Distrobution of TS",
        xlabel="Relative TS to gaussian",
        ylabel="PDF",
        legend=True,
    )  
    plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Gauss_Test")


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
                pVal,flipped = logLikeOfSection(cluster,section,estimate,spread,excludeCrossTalk=excludeCrossTalk)
                if (pVal > max_pVal and len(perm)==len(max_perm)) or (len(perm)>len(max_perm) and pVal > minPval and pVal != max_pVal):
                    max_pVal = pVal
                    max_flipped = flipped
                    max_perm = perm
        pVal = max_pVal
        flipped = max_flipped
        perm = max_perm
    return pVal,flipped,perm

def getGoodIndexes(dataFile,config,minPval=0.2):
    dataFile.init_cluster_voltages()
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    i = 0
    calcFileManager = calcDataFileManager(config["pathToCalcData"], "TSParams", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(
        attribute=f"angle6_4Gev_kit_2",
    )
    estimate,spread = calcFileManager.loadFile(calcFileName=calcFileName)
    indexList = []
    sectionList = []
    flippedList = []
    for cluster in clusters:
        print(f"Cluster: {cluster.getIndex()}",end="\r")
        if not isFlat(cluster):
            continue
        if isOnePixel(cluster):
            continue
        if isOnEdge(cluster):
            continue
        if cluster.getSize(True) <= 10:
            continue
        sections = findSections(cluster)
        pVal,flipped,perm = findBestSections(cluster,sections,estimate,spread,minPval=minPval)
        if pVal<minPval:
            continue
        section = []
        for i in perm:
            section.extend(sections[int(i)])
        indexList.append(cluster.getIndex())
        sectionList.append(section)
        flippedList.append(flipped)
    print(f"")
    return np.array(indexList),sectionList,np.array(flippedList)

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
    if cluster.getSize(excludeCrossTalk=excludeCrossTalk) <= 10:
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

if __name__ == "__main__":

    config = configLoader.loadConfig()
    config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
    dataFiles = initDataFiles(config)

    lengthInDW = 820
    rowPitch = 50
    possibleRows = int(np.ceil(lengthInDW/50))+1
    minPval = 0.5
    for dataFile in dataFiles:
        base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
        dataFile.init_cluster_voltages()
        excludeCrossTalk = True
        clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
        i = 0
        unFixedFitting = []
        calcFileManager = calcDataFileManager(config["pathToCalcData"], "TSParams", config["maxLine"])
        calcFileName = calcFileManager.generateFileName(
            attribute=f"{dataFile.fileName}",
        )
        estimate,spread = calcFileManager.loadFile(calcFileName=calcFileName)
        flippedList = []
        pValsList = []
        permList = []
        indexList = []
        sectionList = []
        rowsList = []
        columnList = []
        relativeTSList = []
        distFromTemplate = []
        for cluster in clusters[20000:]:
            print(f"Cluster: {cluster.getIndex()}",end="\r")
            if i > 10000000:
                break
            #addClusterValues(cluster,estimate,spread,minPval=0.5,excludeCrossTalk=True)
            #if not isPerfectCluster(cluster,estimate,spread,minPval=minPval,excludeCrossTalk=True):
            #    continue
            isPerfectCluster(cluster,estimate,spread,minPval=minPval,excludeCrossTalk=True)
            if len(cluster.section) == 0:
                continue
            i += 1
            #if len(section) == 0:
            #    continue
            flippedList.append(cluster.flipped)
            pValsList.append(cluster.pVal)
            permList.append(cluster.perm)
            indexList.append(cluster.getIndex())
            sectionList.append(cluster.section)
            
            Timestamps = cluster.getTSs(True)
            Rows = cluster.getRows(excludeCrossTalk=True)
            relativeRows = Rows-np.min(Rows)
            relativeTS = Timestamps - np.min(Timestamps)
            if cluster.flipped:
                anchor = np.argmax(relativeRows[cluster.section])
                rowsForFunc = relativeRows[cluster.section][anchor]-relativeRows
            else:
                anchor = np.argmin(relativeRows[cluster.section])
                rowsForFunc = relativeRows-relativeRows[cluster.section][anchor]
            sortIndexes = np.argsort(relativeRows)
            rowsForFunc = rowsForFunc[sortIndexes]
            relativeTS = relativeTS[sortIndexes]
            Rows = Rows[sortIndexes]
            index = (rowsForFunc<len(estimate))&(rowsForFunc>0)
            relativeTS[index]-estimate[rowsForFunc[index]]
            rowsList.extend(Rows[index])
            columnList.extend(cluster.getColumns(excludeCrossTalk=True)[sortIndexes][index])
            relativeTSList.extend(relativeTS[index])
            distFromTemplate.extend(relativeTS[index]-estimate[rowsForFunc[index]])
            #continue
            print("")
            print(cluster.pVal,cluster.flipped,cluster.perm)
            print(f"p = {getPvalue(cluster,cluster.section,estimate,spread,cluster.flipped,excludeCrossTalk=excludeCrossTalk)}")
            path = base_path + f"Cluster_{cluster.getIndex()}/"
            graphTSonRows(cluster,cluster.section,estimate,spread,cluster.flipped,path,excludeCrossTalk=excludeCrossTalk)
            clusterPlotter(cluster, path, "Relative TS").finishPlot("Relative TS", cluster.getTSs(excludeCrossTalk=excludeCrossTalk) - np.min(cluster.getTSs(excludeCrossTalk=excludeCrossTalk)),excludeCrossTalk=excludeCrossTalk,cmap="plasma_r")
            clusterPlotter(cluster, path, "Cluster Map").finishPlot("Voltage", cluster.getHit_Voltages(excludeCrossTalk=excludeCrossTalk),excludeCrossTalk=excludeCrossTalk,cmap="hot")
            clusterPlotter(cluster, path, "ToT").finishPlot("ToT", cluster.getToTs(excludeCrossTalk=excludeCrossTalk),excludeCrossTalk=excludeCrossTalk)
            if cluster.pVal < minPval:
                input("")
                continue
            graphGaussNorm(cluster,cluster.section,estimate,spread,cluster.flipped,path,excludeCrossTalk=excludeCrossTalk)
            input("")
        print("")



        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
        axs = plot.axs
        height, x = np.histogram(pValsList, bins=100)
        axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
        plot.set_config(axs,
            title="Distrobution of pVals",
            xlabel="p Value",
            ylabel="Count",
        )  
        plot.saveToPDF(f"Distrobution_of_pVals")
        
        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
        axs = plot.axs
        height, x = np.histogram([len(perm) for perm in permList], bins=100)
        axs.stairs(height, x, baseline=None, color=plot.colorPalette[1])
        plot.set_config(axs,
            title="Distrobution of Section Number",
            xlabel="Number of Sections",
            ylabel="Count",
        )  
        plot.saveToPDF(f"Distrobution_of_section_number")

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
        axs = plot.axs
        array, yedges, xedges = np.histogram2d([len(perm) for perm in permList],pValsList,bins=(30,30))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
        plot.set_config(axs,
            title="p Value vs Number of Sections",
            xlabel="p Value",
            ylabel="Number of Sections",
        )  
        plot.saveToPDF(f"p_Value__N0_Sections")

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(16,8))
        axs = plot.axs
        TSRange=30
        RowRange=372
        array, yedges, xedges = np.histogram2d(relativeTSList,rowsList,range=((-0.5,TSRange+0.5),(-0.5,RowRange+0.5)),bins=(TSRange+1,RowRange+1))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],norm=LogNorm(vmin=1, vmax=np.max(array)))
        plot.set_config(axs,
            title="Row vs Relative TS",
            xlabel="Row",
            ylabel="Relative TS",
        )
        plot.saveToPDF(f"Row_Relative_TS_log")

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(16,8))
        axs = plot.axs
        array, yedges, xedges = np.histogram2d(distFromTemplate,rowsList,range=((-TSRange/2-0.5,TSRange/2+0.5),(-0.5,RowRange+0.5)),bins=(TSRange+1,RowRange+1))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],norm=LogNorm(vmin=1, vmax=np.max(array)))
        plot.set_config(axs,
            title="Row vs Distance from Template",
            xlabel="Row",
            ylabel="Distance from Template",
        )
        plot.saveToPDF(f"Row_Distance_from_Template_log")

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(16,8))
        axs = plot.axs
        TSRange=30
        RowRange=372
        array, yedges, xedges = np.histogram2d(relativeTSList,rowsList,range=((-0.5,TSRange+0.5),(-0.5,RowRange+0.5)),bins=(TSRange+1,RowRange+1))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])#,norm=LogNorm(vmin=1, vmax=np.max(array)))
        plot.set_config(axs,
            title="Row vs Relative TS",
            xlabel="Row",
            ylabel="Relative TS",
        )
        plot.saveToPDF(f"Row_Relative_TS")

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(16,8))
        axs = plot.axs
        array, yedges, xedges = np.histogram2d(distFromTemplate,rowsList,range=((-TSRange/2-0.5,TSRange/2+0.5),(-0.5,RowRange+0.5)),bins=(TSRange+1,RowRange+1))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])#,norm=LogNorm(vmin=1, vmax=np.max(array)))
        plot.set_config(axs,
            title="Row vs Distance from Template",
            xlabel="Row",
            ylabel="Distance from Template",
        )
        plot.saveToPDF(f"Row_Distance_from_Template")

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(16,8))
        axs = plot.axs
        TSRange=30
        ColumnRange=132
        array, yedges, xedges = np.histogram2d(relativeTSList,columnList,range=((-0.5,TSRange+0.5),(-0.5,ColumnRange+0.5)),bins=(TSRange+1,ColumnRange+1))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
        plot.set_config(axs,
            title="Column vs Relative TS",
            xlabel="Column",
            ylabel="Relative TS",
       )
        plot.saveToPDF(f"Column_Relative_TS")

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/",sizePerPlot=(16,8))
        axs = plot.axs
        array, yedges, xedges = np.histogram2d(distFromTemplate,columnList,range=((-TSRange/2-0.5,TSRange/2+0.5),(-0.5,ColumnRange+0.5)),bins=(TSRange+1,ColumnRange+1))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
        plot.set_config(axs,
            title="Column vs Distance from Template",
            xlabel="Column",
            ylabel="Distance from Template",
       )
        plot.saveToPDF(f"Column_Distance_from_Template")

        continue
        unFixedFitting = np.array(unFixedFitting)

        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
        axs = plot.axs
        array, yedges, xedges = np.histogram2d(unFixedFitting[:,0],unFixedFitting[:,1],range=((0,3),(0,30)),bins=(60,60))
        axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
        plot.set_config(axs,
            title="Fitting Output",
            xlabel="Last Low",
            ylabel="Angle Scaler",
        )  
        plot.saveToPDF(f"Fitting_Output")
