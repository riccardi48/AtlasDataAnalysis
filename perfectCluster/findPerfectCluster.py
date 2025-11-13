import sys
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from scipy.optimize import curve_fit
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
from dataAnalysis._fileReader import calcDataFileManager
from scipy.stats import norm

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

def residual(x,estimate,spread):
    y = (x-estimate)/spread
    return y

def gaussianCDFFunc(x,mu,sig):
    return norm.cdf((x-mu)/sig)

def gaussianFunc(x,mu,sig,scaler):
    return norm.pdf(x,mu,sig)*scaler

config = configLoader.loadConfig()
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

lengthInDW = 820
rowPitch = 50
possibleRows = int(np.ceil(lengthInDW/50))+1

for dataFile in dataFiles:
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
    dataFile.init_cluster_voltages()
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    i = 0
    unFixedFitting = []
    for cluster in clusters[40000:50000:5]:
        if i > 1000000:
            break
        if not isFlat(cluster):
            continue
        if isOnePixel(cluster):
            continue
        if isOnEdge(cluster):
            continue
        if cluster.getSize(True) <= 6:
            continue
        #if cluster.getClusterCharge(True) <= 10 or cluster.getClusterCharge(True) >= 22:
        #    continue
        Timestamps = cluster.getTSs(True)
        TS = Timestamps - np.min(Timestamps)
        Rows = cluster.getRows(excludeCrossTalk=True)
        Columns = cluster.getColumns(excludeCrossTalk=True)
        relativeRows = Rows - np.min(Rows)
        sections = findConnectedSections(Rows,Columns)
        sectionSizes = np.array([len(section) for section in sections])
        largestSectionIndex = np.argmax(sectionSizes)
        largestSection = sections[largestSectionIndex]
        if np.all(TS<=1):
            continue
        x = relativeRows[largestSection]
        y = TS[largestSection]
        sortIndexes = np.argsort(x)
        x = x[sortIndexes]
        y = y[sortIndexes]
        y = y - np.min(y)
        x = x - np.min(x)
        if np.any(y[-4:-2] <= 1) and np.any(y[1:3] <= 1):
            continue
        flipped = isFlipped(y)
        if flipped:
            index = -x+x[-1]
            x = x[index]
            x = np.flip(x)
            y = np.flip(y)
        #bounds = [(0,0,0),(5,30,10)]
        p0 = [1,16 if np.ptp(relativeRows) > 16 else np.ptp(relativeRows)-2,2]
        try:
            popt,pcov = curve_fit(perfectClusterTSFunc,x,y,p0=p0,maxfev = 1000)
            if popt[0] == p0[0] and popt[1] == p0[1]:
                continue
        except:
            continue
        #print(f"Angle Scaler: {popt[0]}")
        #print(f"End Point:    {popt[1]}")
        #print(f"First Pixel:  {popt[2]}")
        unFixedFitting.append(abs(popt))
        i += 1
        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/" + f"Cluster_{cluster.getIndex()}/")
        axs = plot.axs

        axs.scatter(cluster.getRows(True)-np.min(cluster.getRows(True)), TS, color=plot.colorPalette[2], marker="x",label="Cluster TS")
        if flipped:
            anchor = np.argmax(relativeRows[largestSection])
            rowsForFunc = abs(relativeRows[largestSection][anchor]-relativeRows)
        else:
            anchor = np.argmin(relativeRows[largestSection])
            rowsForFunc = relativeRows-relativeRows[largestSection][anchor]
        sortIndexes = np.argsort(relativeRows)
        rowsForFunc = rowsForFunc[sortIndexes]
        relativeRows = relativeRows[sortIndexes]
        axs.plot(relativeRows, perfectClusterTSFunc(rowsForFunc,*popt), color=plot.colorPalette[0],label="Fit")
        
        plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/" + f"Cluster_{cluster.getIndex()}/")
        axs = plot.axs
        calcFileManager = calcDataFileManager(config["pathToCalcData"], "TSParams", config["maxLine"])
        calcFileName = calcFileManager.generateFileName(
            attribute=f"{dataFile.fileName}",
        )
        estimate,spread = calcFileManager.loadFile(calcFileName=calcFileName)
        x = x[x<=30]
        estimate = estimate[x]
        spread = spread[x]
        axs.scatter(np.zeros(x.size)+0.1,residual(x,estimate,spread), color=plot.colorPalette[2], marker="x",label="Cluster TS")
        axs.plot(np.linspace(-1,1,100),gaussianFunc(np.linspace(-1,1,100),0,1,1), color=plot.colorPalette[0],label="Normal")

        plot.set_config(axs,
            title="Distrobution of TS",
            xlabel="Relative TS to gaussian",
            ylabel="PDF",
            legend=True,
        )  
        plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Gauss_Test")
        input("Press Any Key")

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
