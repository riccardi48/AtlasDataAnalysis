from typing import Optional, Any
from dataAnalysis._types import dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    tqdm, # tqdm
)
from dataAnalysis._fileReader import calcDataFileManager
from ._functions import TStoMS
from ._crossTalkFinder import crossTalkFinder
from ._clusters import calcClusters
from ._clusterClass import clusterClass
from ._handler_dataFrameHandler import dataFrameHandler

class clusterHandler:
    def __init__(self, calcFileManager: calcDataFileManager, dataFrameHandler: dataFrameHandler):
        self.haveCrossTalk = False
        self.haveClusters = False
        self.crossTalkFinder = crossTalkFinder()
        self.calcFileManager = calcFileManager
        self.dataFrameHandler = dataFrameHandler
        self._clusters: clusterArray = None

    def getClusters(self, layers: Optional[list[int]] = None, recalc: bool = False,returnIndexes:bool = False,excludeCrossTalk: bool = False) -> clusterArray:
        clusters = self._loadOrCalculateClusters(recalc)
        indexes = np.arange(len(clusters))
        if layers is not None:
            filter = self.layerFilter(clusters, layers=layers)
        else:
            filter = np.full(clusters.size,True)
        if excludeCrossTalk:
            filter &= [cluster.getSize(excludeCrossTalk=True)>0 for cluster in clusters]
        if returnIndexes:
            return (clusters[filter], indexes[filter])
        else:
            return clusters[filter]


    def _loadOrCalculateClusters(self, recalc: bool = False) -> clusterArray:
        if not recalc and self._clusters is not None:
            return self._clusters
        if not recalc and self.calcFileManager.fileExists("clusters"):
            try:
                return self._loadClustersFromFile()
            except Exception as e:
                print(f"Warning: Failed to load clusters from file: {e}")
                print("Recalculating clusters...")
        return self._calculateNewClusters()

    def _loadClustersFromFile(self) -> clusterArray:
        rawClusters = self.calcFileManager.loadFile("clusters")
        self._clusters = self.initClusters(rawClusters)
        return self._clusters

    def _calculateNewClusters(self) -> clusterArray:
        Layers = self.dataFrameHandler.readDataFrameAttr("Layer")
        TriggerIDs = self.dataFrameHandler.readDataFrameAttr("TriggerID")
        TSs = self.dataFrameHandler.readDataFrameAttr("TS")
        print(f"Calculating Clusters")
        rawClusters = calcClusters(Layers, TriggerIDs, TSs)
        print(f"{len(rawClusters)} clusters found")
        self.calcFileManager.saveFile(rawClusters, attribute="clusters")
        self._clusters = self.initClusters(rawClusters, layers=Layers)
        return self._clusters

    def initClusters(
        self,
        clusters: clusterArray,
        layers: Optional[npt.NDArray[np.int_]] = None,
        columns: Optional[npt.NDArray[np.int_]] = None,
        rows: Optional[npt.NDArray[np.int_]] = None,
        ToTs: Optional[npt.NDArray[np.int_]] = None,
        TSs: Optional[npt.NDArray[np.int_]] = None,
    ) -> clusterArray:
        print(f"Initializing Clusters")
        self._clusters = np.empty(len(clusters), dtype=object)
        if layers is None:
            layers = self.dataFrameHandler.readDataFrameAttr("Layer")
        if columns is None:
            columns = self.dataFrameHandler.readDataFrameAttr("Column")
        if rows is None:
            rows = self.dataFrameHandler.readDataFrameAttr("Row")
        if ToTs is None:
            ToTs = self.dataFrameHandler.getToT()
        if TSs is None:
            TSs = self.dataFrameHandler.readDataFrameAttr("ext_TS")
        for i in range(len(clusters)):
            layer = layers[clusters[i][0]]
            self._clusters[i] = clusterClass(
                i,
                clusters[i],
                layer,
                columns[clusters[i]],
                rows[clusters[i]],
                ToTs[clusters[i]],
                TSs[clusters[i]],
            )
        self.haveClusters = True
        #self.dataFrameHandler.dropDataIfRamUsageHigh()
        return self._clusters

    def initClusterVoltages(
        self, Hit_Voltages: npt.NDArray[np.float64], Hit_VoltageErrors: npt.NDArray[np.float64]
    ) -> None:
        if not self.haveClusters:
            self.getClusters()
        for cluster in self._clusters:
            cluster.setHit_Voltage(
                Hit_Voltages[cluster.indexes], Hit_VoltageErrors[cluster.indexes]
            )
        #self.dataFrameHandler.dropDataIfRamUsageHigh()

    def setCalcCrossTalk(self, crosstalk: npt.NDArray[np.bool_]) -> None:
        clusters = self.getClusters()
        for cluster in clusters:
            cluster.setCrossTalk(crosstalk[cluster.indexes])
        self.haveCrossTalk = True

    def calcCrossTalk(self,recalc:bool=False) -> npt.NDArray[np.bool_]:
        clusters = self.getClusters(recalc=recalc)
        print(f"Finding Cross Talk")
        crossTalk = np.full(self.dataFrameHandler.getDataLength(), False, dtype=bool)
        for cluster in tqdm(clusters,desc="Calculating CrossTalk"):
            cluster.setCrossTalk(self.crossTalkFinder.findCrossTalk_OneCluster(cluster))
            crossTalk[cluster.getIndexes()] = cluster.crossTalk
        self.haveCrossTalk = True
        #self.dataFrameHandler.dropDataIfRamUsageHigh()
        return crossTalk

    def getClusterAttr(
        self,
        attribute: str,
        excludeCrossTalk: bool = False,
        returnIndexes: bool = False,
        layers: Optional[list[int]] = None,
    ) -> tuple[npt.NDArray[Any], Optional[npt.NDArray[np.int_]]]:
        toBeReturned: npt.NDArray[Any]
        if attribute == "Sizes":
            toBeReturned = np.array(
                [cluster.getSize(excludeCrossTalk) for cluster in self._clusters]
            )
            filter = self.layerFilter(self._clusters, layers=layers)
        elif attribute == "ColumnWidths":
            toBeReturned = np.array(
                [cluster.getColumnWidth(excludeCrossTalk) for cluster in self._clusters]
            )
            filter = self.layerFilter(self._clusters, layers=layers)
        elif attribute == "RowWidths":
            toBeReturned = np.array(
                [cluster.getRowWidth(excludeCrossTalk) for cluster in self._clusters]
            )
            filter = self.layerFilter(self._clusters, layers=layers)
        elif attribute == "TSs":
            toBeReturned = np.array(
                [np.min(cluster.getEXT_TSs(excludeCrossTalk)) for cluster in self._clusters]
            )
            filter = self.layerFilter(self._clusters, layers=layers)
        elif attribute == "Times":
            toBeReturned = np.array(
                [np.min([cluster.getEXT_TSs(excludeCrossTalk)]) for cluster in self._clusters]
            )
            toBeReturned = TStoMS(toBeReturned - np.min(toBeReturned[toBeReturned>0]))
            filter = self.layerFilter(self._clusters, layers=layers)
        if returnIndexes:
            indexes = np.arange(len(self._clusters))
            return (toBeReturned[filter], indexes[filter])
        else:
            return (toBeReturned[filter], None)

    def layerFilter(
        self, clusters: clusterArray, layers: Optional[list[int]] = None
    ) -> npt.NDArray[np.bool_]:
        if layers == None:
            return np.full(len(clusters), True, bool)
        else:
            return np.isin([cluster.layer for cluster in clusters], layers)

    def getFlatClusters(
        self, width, excludeCrossTalk: bool = False, layers: Optional[list[int]] = None
    ) -> clusterArray:
        self._widthDict = self._loadOrMakeWidthDict(excludeCrossTalk)
        clusters = self.getClusters()
        filter = self.layerFilter(self._clusters, layers=layers)
        return clusters[self._widthDict[width]][filter[self._widthDict[width]]]

    def _loadOrMakeWidthDict(self, excludeCrossTalk: bool = False):
        if "_widthDict" in self.__dict__:
            return self._widthDict
        clusters = self.getClusters()
        cluster: clusterClass
        self._widthDict = {}
        for cluster in clusters:
            width = cluster.getRowWidth(excludeCrossTalk)
            if (
                cluster.getColumnWidth(excludeCrossTalk) != 1
                or cluster.getSize(excludeCrossTalk) < width / 3
            ):
                continue
            if width in self._widthDict:
                self._widthDict[width].append(cluster.getIndex())
                continue
            self._widthDict[width] = [cluster.getIndex()]
        return self._widthDict

    def clearData(self) -> None:
        self._clusters = None
        self.haveClusters = False
        self.haveCrossTalk = False