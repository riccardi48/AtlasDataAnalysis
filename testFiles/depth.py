import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
import numpy as np
from plotAnalysis import plotClass
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def linearLine(a,b,x):
    m = 1/(a-b)
    y = m*(x-a)
    return y
def calcDepthsFromTSs(cluster,excludeCrossTalk=True,residual=1/np.sqrt(12)):
    residual=0.4
    rows = cluster.getRows(excludeCrossTalk)
    sortArray = np.argsort(rows)
    rows = rows[sortArray]
    TSs = cluster.getTSs(excludeCrossTalk)[sortArray]
    if np.ptp(TSs) > 500:
        TSs = (TSs+512)%1024
    relativeTSs = TSs - np.min(TSs)
    highTSs = np.where(relativeTSs>=2)[0]
    #print(highTSs)
    if highTSs.size > 0 and np.average(highTSs) < cluster.getRowWidth(excludeCrossTalk)/2:
        rightToLeft = True
    else:
        rightToLeft = False
    relativeRows = rows-np.min(rows)
    if rightToLeft:
        rows = np.flip(rows)
        relativeTSs = np.flip(relativeTSs)
        highTSs = np.where(relativeTSs>=2)[0]
    if 0 in highTSs:
        minStart = residual
        maxStart = 0.5
        startIndex = 1
    else:
        minStart = -residual
        maxStart = +residual
        startIndex = 0
    if highTSs.size > 1:
        if np.all(np.isin(np.arange(highTSs[startIndex],highTSs[-1]+1),highTSs)):
            minStop = relativeRows[highTSs[startIndex]-1]-residual
            maxStop = relativeRows[highTSs[startIndex]-1]+1**(-rightToLeft)-residual
        else:
            minStop = relativeRows[-1]-residual
            maxStop = relativeRows[-1]+1-residual
    else:
        minStop = relativeRows[-1]-residual
        maxStop = relativeRows[-1]+1-residual

    minDepth = np.zeros(relativeRows.shape)
    maxDepth = np.zeros(relativeRows.shape)
    minDepth[rows<maxStop] = linearLine(minStart,minStop,relativeRows[rows<maxStop])
    minDepth[rows>=maxStop]  = linearLine(maxStart,minStop,relativeRows[rows>=maxStop])
    maxDepth[rows<maxStop] = linearLine(maxStart,maxStop,relativeRows[rows<maxStop])
    maxDepth[rows>=maxStop]  = linearLine(minStart,maxStop,relativeRows[rows>=maxStop])
    depth = np.average([minDepth,maxDepth],axis=0)
    error = np.ptp([minDepth,maxDepth],axis=0)/2
    if rightToLeft:
        depth = np.flip(depth)
        error = np.flip(error)
    depth = depth[np.argsort(sortArray)]
    error = error[np.argsort(sortArray)]
    cluster.depth = depth
    cluster.depthError = error
    cluster.minDepth = minDepth
    cluster.maxDepth = maxDepth


config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    indexes = []
    clusters = dataFile.get_clusters(layers=[4],excludeCrossTalk=True)
    for cluster in clusters:
        if cluster.getSize(True) > 4 and np.unique(cluster.getColumns(True)).size==1 and cluster.getSize(True) > cluster.getRowWidth(True)/2:
            #print(dataFile.get_dataFrame().iloc[cluster.getIndexes(True)])
            calcDepthsFromTSs(cluster)
            indexes.append(cluster.getIndex())
            #input()
    testAngles = np.linspace(0,90,900)
    angles = np.zeros(testAngles.shape)
    d=40
    clusters = dataFile.get_clusters(excludeCrossTalk=True)
    for cluster in clusters[indexes]:
        depths = cluster.depth*d
        depthsError = cluster.depthError*d
        rows = cluster.getRows(True)*50
        angle = np.rad2deg(np.arctan(np.ptp(rows)/np.ptp(depths)))
        angleMax = np.rad2deg(np.arctan(np.ptp(rows[depths>-d])/(np.max(depths-depthsError)-np.min((depths+depthsError)[depths>-d]))))
        angleMin = np.rad2deg(np.arctan(np.ptp(rows[depths>-d])/(np.max(depths+depthsError)-np.min((depths-depthsError)[depths>-d]))))
        angles[(testAngles>=angleMin) & (testAngles<=angleMax)] += 1/np.sum([(testAngles>=angleMin) & (testAngles<=angleMax)])
    plot = plotClass(config["pathToOutput"] + "AngleTest/")
    axs = plot.axs
    #hist, binEdges = np.histogram(angles,bins=50,range=(75,90))
    axs.stairs(
            angles,
            np.linspace(-0.05,90.05,901),
            label=f"{d:.0f} μm",
            baseline=None,
            color=plot.colorPalette[3],
        )
    axs.vlines(
        dataFile.angle, 0, axs.get_ylim()[1], colors=plot.textColor, linestyles="dashed"
    )
    axs.text(
        dataFile.angle,
        axs.get_ylim()[1],
        dataFile.angle,
        color=plot.textColor,
        fontweight="bold",
        horizontalalignment="right",
        verticalalignment="top",
    )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(60, 90),
        title="Angle Distribution",
        legend=True,
        xlabel="Equivalent Angle [Degrees]",
        ylabel="Frequency",
    )
    plot.saveToPDF(f"AngleDistribution_{dataFile.fileName}")

    testDepths = np.linspace(0,100,10000)
    depths = np.zeros(testDepths.shape)
    clusters = dataFile.get_clusters(excludeCrossTalk=True)
    angle = 86.5
    for cluster in clusters[indexes]:
        lastAbove = np.where(cluster.depth>=-1)[0][-1]
        if lastAbove<len(cluster.depth):
            dMax = lastAbove*50/(np.tan(np.deg2rad(angle))*(cluster.maxDepth[0]-cluster.minDepth[lastAbove]))
            dMin = lastAbove*50/(np.tan(np.deg2rad(angle))*(cluster.minDepth[0]-cluster.maxDepth[lastAbove]))
            #dMax = lastAbove*50/np.tan(np.deg2rad(angle))
            #dMin = (lastAbove+1)*50/np.tan(np.deg2rad(angle))
            if dMin > dMax:
                dMin,dMax = dMax,dMin
            if np.sum([(testDepths>=dMax) & (testDepths<=dMin)]) == 0:
                depths[find_nearest(testDepths,dMax)] += 1
            else:
                depths[(testDepths>=dMax) & (testDepths<=dMin)] += 1/np.sum([(testDepths>=dMax) & (testDepths<=dMin)])

    plot = plotClass(config["pathToOutput"] + "AngleTest/")
    axs = plot.axs
    #hist, binEdges = np.histogram(angles,bins=50,range=(75,90))
    axs.stairs(
            depths,
            np.linspace(-0.005,100.005,10001),
            label=f"{d:.0f} μm",
            baseline=None,
            color=plot.colorPalette[3],
        )
    axs.vlines(
        d, 0, axs.get_ylim()[1], colors=plot.textColor, linestyles="dashed"
    )
    axs.text(
        d,
        axs.get_ylim()[1],
        d,
        color=plot.textColor,
        fontweight="bold",
        horizontalalignment="right",
        verticalalignment="top",
    )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(0, 100),
        title="Angle Distribution",
        legend=True,
        xlabel="Equivalent Angle [Degrees]",
        ylabel="Frequency",
    )
    plot.saveToPDF(f"AngleDistribution_{dataFile.fileName}_2")
