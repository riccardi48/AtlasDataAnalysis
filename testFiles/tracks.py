import sys

sys.path.append("..")

from dataAnalysis import initDataFiles, configLoader
from dataAnalysis.handlers._genericClusterFuncs import R2MM,C2MM,MM2C,MM2R
from dataAnalysis.handlers._functions import TStoMS,MStoTS
from dataAnalysis._fileReader import calcDataFileManager
import numpy as np
from tqdm import tqdm


def compareTimes(time, times, tolerance):
    return bool(np.all(abs(time - times) < tolerance))


class trackClass():
    def __init__(self, cluster):
        self.clusters = {cluster.getLayer(): cluster}
        self._cache = {}

    def _invalidateCache(self):
        self._cache = {}

    def addCluster(self, cluster):
        layer = cluster.getLayer()
        if layer not in self.clusters:
            self.clusters[layer] = cluster
            self._invalidateCache()

    def getTSs(self):
        if 'TSs' not in self._cache:
            self._cache['TSs'] = np.array([cluster.getTSs(True)[0] for cluster in self.clusters.values()])
        return self._cache['TSs']

    def getExtTSs(self):
        if 'ExtTSs' not in self._cache:
            self._cache['ExtTSs'] = np.array([cluster.getEXT_TSs(True)[0] for cluster in self.clusters.values()])
        return self._cache['ExtTSs']

    def getLayers(self):
        return list(self.clusters.keys())

    def getClusters(self):
        return list(self.clusters.values())

    def checkClusterMatch(self, cluster, TSTolerance=20, EXTTolerance=2000):
        return (
            compareTimes(cluster.getTSs(True)[0], self.getTSs(True), TSTolerance) and
            compareTimes(cluster.getEXT_TSs(True)[0], self.getExtTSs(True), EXTTolerance)
        )

    def getAllIndexes(self):
        indexes = []
        for cluster in self.getClusters():
            indexes.extend(cluster.getIndexes(True))
        return indexes

    def getIndexes(self):
        return [cluster.getIndex() for cluster in self.getClusters()]

    def checkActive(self, EXT_TS, index, EXT_TSVariance=200000, indexVariance=50):
        if 'maxExtTS' not in self._cache:
            self._cache['maxExtTS'] = np.max(self.getExtTSs())
        if 'maxAllIndex' not in self._cache:
            self._cache['maxAllIndex'] = np.max(self.getAllIndexes())
        return (
            self._cache['maxExtTS'] + EXT_TSVariance >= EXT_TS or
            self._cache['maxAllIndex'] + indexVariance >= index
        )

    def fitTrack(self, offsets):
        if "trackDict" not in self.__dict__:
            z, x, y = [], [], []
            for cluster in self.getClusters():
                layer = cluster.getLayer()
                size = cluster.getSize(True)
                rows = cluster.getRows(True)
                columns = cluster.getColumns(True)
                if layer != 1:
                    off = offsets[layer - 2]
                    rows = rows - off[0]
                    columns = columns - off[1]
                z.append(np.full(size, layer))
                x.append(rows)
                y.append(columns)
            z = np.concatenate(z) * 25.4e-3
            x = R2MM(np.concatenate(x)) * 1e-6
            y = C2MM(np.concatenate(y)) * 1e-6
            self.trackDict = fit_track_3d(x, y, z)

    def checkCut(self, dxdzLimit=(-1, 1), dydzLimit=(-1, 1), residualsLimit=1):
        dxdz = self.trackDict["dxdz"]
        dydz = self.trackDict["dydz"]
        residuals = np.sum(self.trackDict["residuals"])
        return (
            dxdzLimit[0] < dxdz < dxdzLimit[1] and
            dydzLimit[0] < dydz < dydzLimit[1] and
            residuals < residualsLimit
        )


def loadOrCalcTracks(clusters):
    calcFileManager = calcDataFileManager(config["pathToCalcData"], f"{dataFile.fileName}", config["maxLine"])
    calcFileName = calcFileManager.generateFileName(attribute="Tracks")
    if not calcFileManager.fileExists(calcFileName=calcFileName):
        tracks = calcTracks(clusters)
        trackIndexes = [track.getIndexes() for track in tracks]
        calcFileManager.saveFile(calcFileName=calcFileName, array=np.array(trackIndexes, dtype=object))
    else:
        trackIndexes = calcFileManager.loadFile(calcFileName=calcFileName)
        tracks = initTracks(trackIndexes, clusters)
    return tracks


def initTracks(trackIndexes, clusters):
    tracks = []
    for trackIndex in trackIndexes:
        track = trackClass(clusters[trackIndex[0]])
        for index in trackIndex[1:]:
            track.addCluster(clusters[index])
        tracks.append(track)
    return tracks


def resetClusterIndexes(clusters):
    for i, cluster in enumerate(clusters):
        cluster.index = i


def calcTracks(clusters):
    activeTracks = []
    finishedTracks = []
    for cluster in tqdm(clusters, desc="Calculating Tracks"):
        clusterTS = cluster.getTSs(True)[0]
        clusterExtTS = cluster.getEXT_TSs(True)[0]
        clusterExtTSLast = cluster.getEXT_TSs(True)[-1]
        clusterIndexLast = cluster.getIndexes(True)[-1]

        foundTrack = False
        for track in activeTracks:
            if compareTimes(clusterTS, track.getTSs(), 20) and compareTimes(clusterExtTS, track.getExtTSs(), 2000):
                track.addCluster(cluster)
                foundTrack = True
                break
        if not foundTrack:
            activeTracks.append(trackClass(cluster))

        newFinished = [t for t in activeTracks if not t.checkActive(clusterExtTSLast, clusterIndexLast)]
        finishedTracks += newFinished
        activeTracks = [t for t in activeTracks if t.checkActive(clusterExtTSLast, clusterIndexLast)]

    finishedTracks += activeTracks
    return finishedTracks

def fit_track_3d(x, y, z):
    """
    Fit a 3D line through detector hits using PCA/SVD.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates of the hits (z = layer position).

    Returns
    -------
    dict with:
        centroid  : (3,) array — point on the fitted line
        direction : (3,) array — unit direction vector
        dxdz      : float — slope in x vs z
        dydz      : float — slope in y vs z
        residuals : (N,) array — perpendicular distance of each point from line
        predict   : callable(z_new) -> (x, y) — predicts x, y at any z
    """
    points = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    direction = Vt[0]

    # Ensure direction points in +z direction for consistency
    if direction[2] < 0:
        direction = -direction

    t_vals = (points - centroid) @ direction
    proj = centroid + np.outer(t_vals, direction)
    residuals = np.linalg.norm(points - proj, axis=1)

    dxdz = direction[0] / direction[2]
    dydz = direction[1] / direction[2]

    def predict(z_new):
        """Predict (x, y) position at a given z (e.g. a downstream layer)."""
        t = (z_new - centroid[2]) / direction[2]
        x_pred = centroid[0] + t * direction[0]
        y_pred = centroid[1] + t * direction[1]
        return x_pred, y_pred

    return {
        "centroid":  centroid,
        "direction": direction,
        "dxdz":      dxdz,
        "dydz":      dydz,
        "residuals": residuals,
        "predict":   predict,
    }

def alignment(tracks):
    layer2Offset = []
    layer3Offset = []
    layer4Offset = []
    for k,track in enumerate(tracks):
        if len(track.getLayers()) != 4:
            continue
        rows1 = np.mean(track.clusters[1].getRows(True))
        columns1 = np.mean(track.clusters[1].getColumns(True))
        layer2Offset.append([np.mean(track.clusters[2].getRows(True)-rows1),np.mean(track.clusters[2].getColumns(True)-columns1)])
        layer3Offset.append([np.mean(track.clusters[3].getRows(True)-rows1),np.mean(track.clusters[3].getColumns(True)-columns1)])
        layer4Offset.append([np.mean(track.clusters[4].getRows(True)-rows1),np.mean(track.clusters[4].getColumns(True)-columns1)])
    layer2Offset = np.median(layer2Offset,axis=0)
    layer3Offset = np.median(layer3Offset,axis=0)
    layer4Offset = np.median(layer4Offset,axis=0)
    return np.array([layer2Offset,layer3Offset,layer4Offset])

def plotTrackAttribute(tracks,offsets,attribute,plotGen,path,_range=None):
    plot = plotGen.newPlot(path)
    listOFAttributes = []
    for k,track in enumerate(tracks):
        if len(track.getLayers()) != 4:
            continue
        track.fitTrack(offsets)
        z = []
        x = []
        y = []
        for cluster in track.getClusters():
            z.extend(list(np.repeat(cluster.getLayer(),cluster.getSize(True))))
            rows = cluster.getRows(True)
            columns = cluster.getColumns(True)
            if cluster.getLayer() != 1:
                rows = rows-offsets[cluster.getLayer()-2,0]
                columns = columns-offsets[cluster.getLayer()-2,1]
            x.extend(list(rows))
            y.extend(list(columns))
        z = np.array(z)*25.4*1e-3
        x = R2MM(np.array(x))*1e-6
        y = C2MM(np.array(y))*1e-6
        if attribute == "residuals":
            listOFAttributes.append(np.sum(track.trackDict[attribute]))
        else:
            listOFAttributes.append(track.trackDict[attribute])
    height, x = np.histogram(
        listOFAttributes,
        bins=200,
        range=_range,
    )
    plot.axs.stairs(
        height, x, color=plot.colorPalette[1], baseline=None, label=f"{attribute}"
    )
    plot.set_config(
            plot.axs,
            title=f"{attribute}",
            xlabel=f"{attribute}",
            ylabel="Counts",
            legend=True,
            ylim=(0, None),
        )
    plot.saveToPDF(f"{attribute}")

config = configLoader.loadConfig()
config["filterDict"] = {"fileName":"angle6_4GeV_lancs_3"}
#config["filterDict"] = {"fileName":"angle6_6GeV_lancs_5"}
dataFiles = initDataFiles(config)

from plotFiles.plotClass import plotGenerator


plotGen = plotGenerator(f"{config["pathToOutput"]}Tracks/")
for dataFile in dataFiles:
    path = f"{dataFile.fileName}/"
    clusters = dataFile.get_clusters(excludeCrossTalk = True)
    resetClusterIndexes(clusters)
    df = dataFile.get_dataFrame()
    tracks = loadOrCalcTracks(clusters)
    print(len([True for track in tracks if len(track.getLayers()) == 4]))
    print(len(tracks))
    offsets = alignment(tracks)
    print(offsets)
    for track in tracks:
        if len(track.getLayers()) != 4:
            continue
        track.fitTrack(offsets)
    #plotTrackAttribute(tracks,offsets,"dxdz",plotGen,path,_range=(-0.01,0.01))
    #plotTrackAttribute(tracks,offsets,"dydz",plotGen,path,_range=(-0.01,0.01))
    #plotTrackAttribute(tracks,offsets,"residuals",plotGen,path,_range=(0,0.002))
    for k,track in enumerate(tracks):
        break
        if len(track.getLayers()) != 4:
            continue
        print(df.iloc[track.getAllIndexes(),:5])
        plot = plotGen.newPlot(path)
        colors = ["red","green","blue","purple"]
        for i in range(1,5):
            cluster = track.clusters[i]
            rows = cluster.getRows(True)
            columns = cluster.getColumns(True)
            if i != 1:
                rows = rows-offsets[cluster.getLayer()-2,0]
                columns = columns-offsets[cluster.getLayer()-2,1]
            plot.axs.scatter(rows,columns,marker="x",color=colors[i-1],label=f"Layer {i}")
        plot.set_config(
            plot.axs,
            title=f"Scatter of Track",
            xlabel="Rows",
            ylabel="Columns",
            legend=True,
            xlim=(0,372),
            ylim=(0,132),
        )
        plot.saveToPDF(f"track_{k}")

        print(track.trackDict)
        input()
    tracks4Layer = [track for track in tracks if len(track.getLayers()) == 4]
    goodTracks = np.array([track for track in tracks4Layer if track.checkCut(dxdzLimit=(-0.0025,0.0025),dydzLimit=(-0.0025,0.0025),residualsLimit=0.002)])
    print(len(goodTracks))
    #for track in goodTracks:
    #    print(df.iloc[track.getAllIndexes(),:5])
    #    input()
    times = [np.min(track.getExtTSs()) for track in goodTracks]
    firstTime = np.min(times[:100])
    firstTime = clusters[0].getEXT_TSs(True)[0]
    times = times - firstTime
    _range = (0, 500000)
    bins = int(np.ptp(_range) / 1000)
    height, x = np.histogram(TStoMS(times), bins=bins, range=_range)
    plot = plotGen.newPlot(path)
    plot.axs.stairs(height, x, baseline=None, color=plot.colorPalette[1], label=f"{dataFile.fileName}")
    plot.set_config(
        plot.axs,
        ylim=(0, None),
        xlim=_range,
        title="Clusters Count Over Time",
        legend=False,
        xlabel="Time [ms]",
        ylabel="Count",
    )
    plot.saveToPDF(f"GoodTrackTimes_{dataFile.fileName}")

config = configLoader.loadConfig()
config["filterDict"] = {"fileName":["angle6_4Gev_kit_2"]}
#config["filterDict"] = {"fileName":["angle6_6Gev_kit_4"]}
dataFiles = initDataFiles(config)

print(dataFiles[0].fileName)
clusters1 = dataFiles[0].get_clusters(excludeCrossTalk=True,layers=config["layers"])
resetClusterIndexes(clusters1)
clusters3 = dataFiles[0].get_perfectClusters(excludeCrossTalk=True,layers=config["layers"])
times1 = np.array([np.min(cluster.getEXT_TSs(True)) for cluster in clusters1])
times3 = np.array([np.min(cluster.getEXT_TSs(True)) for cluster in clusters3])

times2 = np.array([np.min(track.clusters[4].getEXT_TSs(True)) for track in goodTracks])

bins1 = int(np.ptp(TStoMS(times1))/1000)
print(bins1)
height1, x1 = np.histogram(times1, bins=bins1)
print(height1[:20])
bins2 = int(np.ptp(TStoMS(times2))/1000)

height2, x2 = np.histogram(times2, bins=bins2)
height2 = height2[10:]
x2 = x2[10:]
peak1 = int(x1[np.where(height1>np.max(height1)/2)[0][0]])-(x1[1]-x1[0])
print(np.where(height1>np.max(height1)/2)[0][0])
print(height1[np.where(height1>10)[0][0]:np.where(height1>10)[0][0]+10])
print(x1[np.where(height1>10)[0][0]:np.where(height1>10)[0][0]+10])

peak2 = int(x2[np.where(height2>np.max(height2)/2)[0][0]])-(x2[1]-x2[0])
print(np.where(height2>np.max(height2)/2)[0][0])
print(height2[np.where(height2>10)[0][0]:np.where(height2>10)[0][0]+10])
print(x2[np.where(height2>10)[0][0]:np.where(height2>10)[0][0]+10])

print(peak1)
print(peak2)

timeLength = 3000000000
#timeLength = 20000000000
tolerance = 2000

syncTimes1 = times1[(times1>peak1)&(times1<peak1+timeLength)&(times1<np.max(times2))]-peak1
syncTimes2 = times2[(times2>peak2)&(times2<peak2+timeLength)&(times2<np.max(times1))]-peak2
syncTimes3 = times3[(times3>peak1)&(times3<peak1+timeLength)&(times3<np.max(times2))]-peak1

print(len(syncTimes1))
print(len(syncTimes2))
print(len(syncTimes3))

from _track import sync_sensors, match_hits, DriftModel, MatchResult


sub_result = sync_sensors(syncTimes1,syncTimes2,max_offset_ts=3e7,coincidence_window_ts=100,n_drift_chunks=6)

syncTimes1 = times1[(times1>peak1)&(times1<np.max(times2))]-peak1
syncTimes2 = times2[(times2>peak2)&(times2<np.max(times1))]-peak2
syncTimes3 = times3[(times3>peak1)&(times3<np.max(times2))]-peak1

result = match_hits(syncTimes1, syncTimes2, drift_model=sub_result.drift_model, coincidence_window_ts=500)
print(result.summary())

plot = plotGen.newPlot(f"{dataFiles[0].fileName}",sizePerPlot=(50,10))
timeLengthPlot=1e8
times2_corrected = sub_result.drift_model.correct_TS(times2-peak2)
plotTimes1 = times1[(times1>peak1)&(times1<peak1+timeLengthPlot)]-peak1
plotTimes2 = times2_corrected[(times2_corrected>0)&(times2_corrected<timeLengthPlot)]
plotTimes3 = times3[(times3>peak1)&(times3<peak1+timeLengthPlot)]-peak1
plot.axs.vlines(plotTimes1,0,50,color="red",linewidth=1)
plot.axs.vlines(plotTimes3,0,50,color="green",linewidth=1)
plot.axs.vlines(plotTimes2,50,100,color="blue",linewidth=1)
plot.set_config(
        plot.axs,
        ylim=(0, 100),
    )
plot.saveToPDF("Times")

plot = plotGen.newPlot(f"{dataFiles[0].fileName}",sizePerPlot=(50,5))
timeLengthPlot=2e7
plotTimes1 = times1[(times1>peak1)&(times1<peak1+timeLengthPlot)]-peak1
plotTimes2 = times2_corrected[(times2_corrected>0)&(times2_corrected<timeLengthPlot)]
plotTimes3 = times3[(times3>peak1)&(times3<peak1+timeLengthPlot)]-peak1
plot.axs.vlines(plotTimes1,0,50,color="red",linewidth=1)
plot.axs.vlines(plotTimes3,0,50,color="green",linewidth=1)
plot.axs.vlines(plotTimes2,50,100,color="blue",linewidth=1)
plot.set_config(
        plot.axs,
        ylim=(0, 100),
    )
plot.saveToPDF("Times_zoomed")

def compress_array(arr):
    """Run-length encode a numpy array."""
    if len(arr) == 0:
        return []
    
    # Find where values change
    changes = np.where(np.diff(arr) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(arr)]))
    
    runs = [(arr[s], e - s) for s, e in zip(starts, ends)]
    return runs

def print_compressed(arr):
    runs = compress_array(arr)
    unique_vals, counts = np.unique(arr, return_counts=True)
    
    print(f"Original length : {len(arr)}")
    print(f"Compressed runs : {len(runs)}")
    print(f"Unique values   : {dict(zip(unique_vals, counts))}")
    print(f"Space saved     : {(1 - len(runs)/len(arr))*100:.1f}%")
    print()
    print("Compressed view (value x count):")
    print("-" * 40)
    
    line = ""
    for value, count in runs:
        entry = f"{value}x{count}  "
        line += entry
        if len(line) > 80:  # wrap long lines
            print(line)
            line = ""
    if line:
        print(line)


synced = [np.sum(abs((pTime)-(times2_corrected))<tolerance) for pTime in times3-peak1]
print(np.unique(synced,return_counts=True))
#print_compressed(synced)
#synced = [np.sum(abs((pTime)-(times2_corrected))<tolerance) for pTime in times1-peak1]
#print(np.unique(synced,return_counts=True))
#print_compressed(synced)
#synced = [np.sum(abs((pTime)-(times2-offset))<tolerance*10) for pTime in times1]
#synced = [np.sum(abs((pTime-peak1)-(times2-peak2))<tolerance) for pTime in times3]
#synced = [np.sum(abs((pTime)-(times2-offset))<tolerance*10) for pTime in times1]
#print(np.unique(synced,return_counts=True))

#print_compressed(synced)

plot = plotGen.newPlot(f"{dataFiles[0].fileName}")
height, x = np.histogram(
    [cluster.getRowWidth(True) for cluster in clusters1[(times1>peak1)&(times1<np.max(times2))][result.matched_a]],
    bins=40,
    range=(0.5,40.5),
)
plot.axs.stairs(
    height/np.sum(height), x, color=plot.colorPalette[1], baseline=None, label="Matched",
)
height, x = np.histogram(
    [cluster.getRowWidth(True) for cluster in clusters1[(times1>peak1)&(times1<np.max(times2))]],
    bins=40,
    range=(0.5,40.5),
)
plot.axs.stairs(
    height/np.sum(height), x, color=plot.colorPalette[2], baseline=None, label="All"
)
plot.set_config(
        plot.axs,
        ylabel="Counts",
        ylim=(0, None),
        legend=True,
    )
plot.saveToPDF(f"RowWidths")
