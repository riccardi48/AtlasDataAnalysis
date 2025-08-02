from typing import Optional, Any
from dataAnalysis._types import clusterClass,dataAnalysis,clusterArray
from dataAnalysis._dependencies import (
    pd,                 # pandas
    np,                 # numpy
    npt,                # numpy.typing
)
from dataAnalysis._fileReader import rawDataFileManager, calcDataFileManager
from ._functions import readFileName,calcToT,trueTimeStamps,TStoMS
from ._hit_voltage import calcHit_Voltage,calcHit_VoltageError
from ._crossTalkFinder import crossTalkFinder
from ._clusters import calcClusters
from ._clusterClass import clusterClass

class dataHandler:
    def __init__(self, pathToData: str, pathToCalcData: str, maxLine: Optional[int] = None):
        self.fileName, self.voltage, self.angle, self.telescope = readFileName(pathToData)
        self.baseAttrNames = [
            "PackageID",
            "Layer",
            "Column",
            "Row",
            "TS",
            "TS1",
            "TS2",
            "TriggerTS",
            "TriggerID",
            "ext_TS",
            "ext_TS2",
            "FIFO_overflow",
        ]
        self.dataFrameHandler = dataFrameHandler(
            pathToData,self.baseAttrNames, maxLine=maxLine, telescope=self.telescope
        )
        self.calcFileManager = calcDataFileManager(pathToCalcData, self.fileName, maxLine)
        self.clusterHandler = clusterHandler(self.calcFileManager, self.dataFrameHandler)

    def getDataFrame(self) -> pd.DataFrame:
        return self.dataFrameHandler.data

    def baseAttr(
        self,
        attribute: str,
        excludeCrossTalk: bool = False,
        layers: Optional[list] = None,
        returnIndexes: bool = False,
        recalc=False,
    ) -> tuple[npt.NDArray[Any], Optional[npt.NDArray[Any]]]:
        if attribute in self.baseAttrNames:
            toBeReturned = self.dataFrameHandler.readDataFrameAttr(attribute)
        else:
            if attribute in self.__dict__ and not recalc:
                toBeReturned = getattr(self, attribute)
            elif self.calcFileManager.fileExists(attribute) and not recalc:
                toBeReturned = self.calcFileManager.loadFile(attribute)
                getattr(self, attribute, toBeReturned)
            else:
                attr = getattr(self.dataFrameHandler, "get" + attribute)
                if attribute in ["Hit_Voltage", "Hit_VoltageError"]:
                    toBeReturned = attr(ToTs=self.baseAttr("ToT")[0])
                else:
                    toBeReturned = attr()
                getattr(self, attribute, toBeReturned)
                self.calcFileManager.saveFile(toBeReturned, attribute=attribute)
        toBeReturned, indexes = self.layerCrosstalkFilter(toBeReturned, excludeCrossTalk, layers)
        if returnIndexes:
            return toBeReturned, indexes
        else:
            return (toBeReturned, None)

    def getClusters(self, excludeCrossTalk: bool = False, **kwargs) -> clusterArray:
        if excludeCrossTalk:
            self.getCrossTalk(**kwargs)
        return self.clusterHandler.getClusters(**kwargs)

    def getClustersAttr(
        self,
        attribute: str,
        excludeCrossTalk: bool = False,
        returnIndexes: bool = False,
        layers: Optional[list[int]] = None,
    ) -> tuple[npt.NDArray[Any], Optional[npt.NDArray[np.int_]]]:
        toBeReturned: npt.NDArray[Any]
        indexes: Optional[npt.NDArray[np.int_]]
        if self.calcFileManager.fileExists(attribute, cut=excludeCrossTalk, layers=layers) and (
            not returnIndexes
            or self.calcFileManager.fileExists(
                "clusterIndexes", cut=excludeCrossTalk, layers=layers
            )
        ):
            toBeReturned = self.calcFileManager.loadFile(
                attribute, cut=excludeCrossTalk, layers=layers, suppressText=True
            )
            if returnIndexes:
                indexes = self.calcFileManager.loadFile(
                    "clusterIndexes", cut=excludeCrossTalk, layers=layers, suppressText=True
                )
        else:
            if not self.clusterHandler.haveClusters:
                self.getClusters()
            if excludeCrossTalk and not self.clusterHandler.haveCrossTalk:
                self.getCrossTalk()
            if returnIndexes:
                toBeReturned, indexes = self.clusterHandler.getClusterAttr(
                    attribute,
                    excludeCrossTalk=excludeCrossTalk,
                    returnIndexes=returnIndexes,
                    layers=layers,
                )
                self.calcFileManager.saveFile(
                    toBeReturned, attribute=attribute, cut=excludeCrossTalk, layers=layers
                )
                if indexes is not None:
                    self.calcFileManager.saveFile(
                        indexes, "clusterIndexes", cut=excludeCrossTalk, layers=layers
                    )
            else:
                toBeReturned = self.clusterHandler.getClusterAttr(
                    attribute,
                    excludeCrossTalk=excludeCrossTalk,
                    returnIndexes=returnIndexes,
                    layers=layers,
                )[0]
                self.calcFileManager.saveFile(
                    toBeReturned, attribute=attribute, cut=excludeCrossTalk, layers=layers
                )
        if returnIndexes:
            return (toBeReturned, indexes)
        else:
            return (toBeReturned, None)

    def initClusterVoltages(self) -> None:
        self.clusterHandler.initClusterVoltages(
            self.baseAttr("Hit_Voltage")[0], self.baseAttr("Hit_VoltageError")[0]
        )

    def notCrossTalk(self) -> npt.NDArray[np.bool_]:
        return np.invert(self.getCrossTalk())

    def getCrossTalk(
        self, recalc: bool = False, layers: Optional[list[int]] = None, initClusters: bool = True
    ) -> npt.NDArray[np.bool_]:
        if "crossTalk" in self.__dict__ and not recalc:
            toBeReturned = self.crossTalk
            if initClusters:
                self.clusterHandler.setCalcCrossTalk(self.crossTalk)
        elif self.calcFileManager.fileExists("crossTalk") and not recalc:
            self.crossTalk: npt.NDArray[np.bool_] = self.calcFileManager.loadFile("crossTalk")
            if self.telescope == "lancs":
                row = 248
                self.dataFrameHandler.cutSensor(row)
            toBeReturned = self.crossTalk
            if initClusters:
                self.clusterHandler.setCalcCrossTalk(self.crossTalk)
        else:
            self.crossTalk = self.clusterHandler.calcCrossTalk()
            if self.telescope == "lancs":
                row = 248
                self.crossTalk = self.crossTalk[np.where(self.baseAttr("Row")[0] < row)[0]]
                self.dataFrameHandler.cutSensor(row)
                self.getClusters(recalc=True)
                self.baseAttr("ToT", recalc=True)
                self.baseAttr("Hit_Voltage", recalc=True)
                self.clusterHandler.setCalcCrossTalk(self.crossTalk)
            self.calcFileManager.saveFile(self.crossTalk, attribute="crossTalk")
            toBeReturned = self.crossTalk
        toBeReturned, indexes = self.layerCrosstalkFilter(toBeReturned, False, layers)
        return toBeReturned

    def cutArrayCrossTalkFilter(
        self, array: npt.NDArray[Any], excludeCrossTalk: bool = True
    ) -> npt.NDArray[np.bool_]:
        if excludeCrossTalk:
            includedIndexes = self.notCrossTalk()
        else:
            includedIndexes = np.full(len(array), True, dtype=bool).astype(np.bool_)
        return includedIndexes

    def layerFilter(
        self, array: npt.NDArray[Any], layers: Optional[list[int]] = None
    ) -> npt.NDArray[np.bool_]:
        if layers is None:
            includedIndexes = np.full(len(array), True, dtype=bool).astype(np.bool_)
        else:
            layersArray = self.baseAttr("Layer")[0]
            includedIndexes = np.isin(layersArray, layers, assume_unique=True, kind="table")
        return includedIndexes

    def layerCrosstalkFilter(
        self,
        array: npt.NDArray[Any],
        excludeCrossTalk: bool = False,
        layers: Optional[list[int]] = None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        filter = (self.cutArrayCrossTalkFilter(array, excludeCrossTalk)) & (
            self.layerFilter(array, layers)
        )
        array = array[filter]
        indexes = np.arange(len(filter))[filter]
        return array, indexes

    def save_nonCrossTalk_to_csv(self, path, name) -> None:
        self.dataFrameHandler.saveNonCrossTalkToCSV(
            self.getCrossTalk(), path, name, self.getClusters()
        )


class dataFrameHandler:
    # Stores the data frame
    # Retrieves calculated data for values on a per hit basis i.e. one value per line in data frame e.g. ToT, Voltage
    def __init__(
        self,
        pathToData: str,
        columnNames:list[str],
        maxLine: Optional[int] = None,
        telescope: str = "kit",
    ):
        self.dataFileManager = rawDataFileManager(pathToData,columnNames, maxLine=maxLine)
        self.telescope = telescope

    def loadDataIfNotLoaded(self) -> None:
        if not self.checkLoadedData():
            self.data = self.dataFileManager.readFile()
            self.dataLength:int = len(self.data)

    def cutSensor(self, row: int) -> pd.DataFrame:
        self.loadDataIfNotLoaded()
        positions = np.where(self.data["Row"].to_numpy() >= row)[0]
        index_labels = self.data.index[positions]
        self.data = self.data.drop(index=index_labels)
        self.data.reset_index(drop=True, inplace=True)
        return self.data

    def checkLoadedData(self) -> bool:
        if "data" in self.__dict__.keys():
            return True
        else:
            return False

    def readDataFrameAttr(self, attribute) -> npt.NDArray[np.int_]:
        self.loadDataIfNotLoaded()
        toBeReturned = self.data[attribute].to_numpy()
        return toBeReturned

    def getToT(self) -> npt.NDArray[np.int_]:
        self.loadDataIfNotLoaded()
        toBeReturned = calcToT(self.data["TS"].to_numpy(), self.data["TS2"].to_numpy())
        return toBeReturned

    def getHit_Voltage(
        self, ToTs: Optional[npt.NDArray[np.int_]] = None
    ) -> npt.NDArray[np.float64]:
        self.loadDataIfNotLoaded()
        if ToTs is None:
            ToTs = self.getToT()
        rows = self.data["Row"].to_numpy()
        columns = self.data["Column"].to_numpy()
        Layers = self.data["Layer"].to_numpy()
        toBeReturned = calcHit_Voltage(rows, columns, ToTs, Layers)
        return toBeReturned

    def getHit_VoltageError(
        self, ToTs: Optional[npt.NDArray[np.int_]] = None
    ) -> npt.NDArray[np.float64]:
        self.loadDataIfNotLoaded()
        if ToTs is None:
            ToTs = self.getToT()
        rows = self.data["Row"].to_numpy()
        columns = self.data["Column"].to_numpy()
        Layers = self.data["Layer"].to_numpy()
        toBeReturned = calcHit_VoltageError(rows, columns, ToTs, Layers)
        return toBeReturned

    def getDataLength(self) -> int:
        self.loadDataIfNotLoaded()
        if self.dataLength > 0:
            return self.dataLength
        else:
            self.dataLength = len(self.data)
            return self.dataLength

    def saveNonCrossTalkToCSV(
        self, crosstalk: npt.NDArray[np.bool_], path: str, name: str, clusters: clusterArray
    ) -> None:
        outputDF = self.data
        outputDF["ext_TS"] = trueTimeStamps(clusters, outputDF["ext_TS"].to_numpy())
        positions = np.where(crosstalk)[0]
        index_labels = self.data.index[positions]
        outputDF = outputDF.drop(index=index_labels)
        self.dataFileManager.saveFile(outputDF, path, name)


class clusterHandler:
    def __init__(self, calcFileManager: calcDataFileManager, dataFrameHandler: dataFrameHandler):
        self.haveCrossTalk = False
        self.haveClusters = False
        self.crossTalkFinder = crossTalkFinder()
        self.calcFileManager = calcFileManager
        self.dataFrameHandler = dataFrameHandler
        self._clusters: clusterArray = None

    def getClusters(self, layers: Optional[list[int]] = None, recalc: bool = False) -> clusterArray:
        clusters = self._loadOrCalculateClusters(recalc)
        if layers is not None:
            filter = self.layerFilter(self.clusters, layers=layers)
            return clusters[filter]
        return clusters
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
        rawClusters = calcClusters(
            Layers, 
            TriggerIDs, 
            TSs
        )
        print(f"{len(rawClusters)} clusters found")
        self.calcFileManager.saveFile(rawClusters, attribute="clusters")
        self._clusters = self.initClusters(
            rawClusters, 
            layers=Layers, 
            TSs=TSs
        )
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

    def setCalcCrossTalk(self, crosstalk: npt.NDArray[np.bool_]) -> None:
        clusters = self.getClusters()
        for cluster in clusters:
            cluster.setCrossTalk(crosstalk[cluster.indexes])
        self.haveCrossTalk = True

    def calcCrossTalk(self) -> npt.NDArray[np.bool_]:
        clusters = self.getClusters()
        print(f"Finding Cross Talk")
        crossTalk = np.full(self.dataFrameHandler.getDataLength(), False, dtype=bool)
        for i, cluster in enumerate(clusters):
            cluster.setCrossTalk(self.crossTalkFinder.findCrossTalk_OneCluster(cluster))
            crossTalk[cluster.getIndexes()] = cluster.crossTalk
            if i % 1000 == 0:
                print(f"{i/clusters.size*100:.2f}%", end="\r")
        print("100.00%")
        self.haveCrossTalk = True
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
            filter = self.layerFilter(self._clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "ColumnWidths":
            toBeReturned = np.array(
                [cluster.getColumnWidth(excludeCrossTalk) for cluster in self._clusters]
            )
            filter = self.layerFilter(self._clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "RowWidths":
            toBeReturned = np.array(
                [cluster.getRowWidth(excludeCrossTalk) for cluster in self._clusters]
            )
            filter = self.layerFilter(self._clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "TSs":
            toBeReturned = np.array(
                [np.min(cluster.getTSs(excludeCrossTalk)) for cluster in self._clusters]
            )
            filter = self.layerFilter(self._clusters, layers=layers)
        elif attribute == "Times":
            toBeReturned = np.array(
                [np.min([cluster.getTSs(excludeCrossTalk)]) for cluster in self._clusters]
            )
            toBeReturned = TStoMS(toBeReturned.astype(int) - int(np.min(toBeReturned)))
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