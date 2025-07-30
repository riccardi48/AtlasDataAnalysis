from lowLevelFunctions import (
    isFiltered,
    readFileName,
    print_mem_usage,
    calcToT,
    calcHit_VoltageError,
    calcHit_Voltage,
    TStoMS,
    calcClusters,
    trueTimeStamps,
)
from clusterClass import clusterClass
import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from glob import glob
from typing import Optional, Any, TypeAlias
import configLoader
from collections import defaultdict


clusterArray: TypeAlias = npt.NDArray[np.object_]


class dataAnalysis:
    def __init__(self, pathToData: str, pathToCalcData: str, maxLine: Optional[int] = None) -> None:
        self.dataHandler = dataHandler(pathToData, pathToCalcData, maxLine=maxLine)
        self.pathToData = pathToData

    def get_voltage(self) -> int:
        return self.dataHandler.voltage

    def get_angle(self) -> float:
        return self.dataHandler.angle

    def get_fileName(self) -> str:
        return self.dataHandler.fileName

    def get_telescope(self) -> str:
        return self.dataHandler.telescope

    def check_if_filtered(self, filterDict) -> bool:
        return isFiltered(self, filter_dict=filterDict)

    def get_dataFrame(self) -> pd.DataFrame:
        return self.dataHandler.getDataFrame()

    def get_base_attr(
        self, attribute: str, **kwargs
    ) -> tuple[npt.NDArray[Any], Optional[npt.NDArray[Any]]]:
        if "layer" in kwargs:
            kwargs["layers"] = [kwargs["layer"]]
            kwargs.pop("layer")
        return self.dataHandler.baseAttr(attribute, **kwargs)

    def get_clusters(self, **kwargs) -> clusterArray:
        if "layer" in kwargs:
            if kwargs["layer"] is None:
                kwargs["layers"] = None
                kwargs.pop("layer")
            else:
                kwargs["layers"] = [kwargs["layer"]]
                kwargs.pop("layer")
        return self.dataHandler.getClusters(**kwargs)

    def get_cluster_attr(
        self, attribute, excludeCrossTalk: bool = False, **kwargs
    ) -> tuple[npt.NDArray[Any], Optional[npt.NDArray[np.int_]]]:
        if "layer" in kwargs:
            kwargs["layers"] = [kwargs["layer"]]
            kwargs.pop("layer")
        return self.dataHandler.getClustersAttr(
            attribute, excludeCrossTalk=excludeCrossTalk, **kwargs
        )

    def get_crossTalk(self, recalc: bool = False) -> npt.NDArray[np.bool_]:
        return self.dataHandler.getCrossTalk(recalc=recalc)

    def init_cluster_voltages(self) -> None:
        self.dataHandler.initClusterVoltages()

    def save_nonCrossTalk_to_csv(self, path) -> None:
        self.dataHandler.save_nonCrossTalk_to_csv(path, self.get_fileName())


class calcDataFileManager:
    def __init__(self, pathToCalcData: str, fileName: str, maxLine: Optional[int] = None):
        self.pathToCalcData = pathToCalcData
        self.fileName = fileName
        self.maxLine = maxLine
        os.makedirs(f"{self.pathToCalcData}", exist_ok=True)

    def generateFileName(
        self,
        attribute: str,
        cut: bool = False,
        layers: Optional[list[int]] = None,
        name: str = "",
        file: str = "",
    ) -> str:
        pathToFile = (
            f"{self.pathToCalcData}"
            + f"{self.fileName}/"
            + f"{file}"
            + f"{attribute}"
            + f"{"_cut" if cut else ""}"
            + f"{f"_{self.maxLine}" if self.maxLine is not None else ""}"
            + f"{"_"+"".join(str(x) for x in layers) if layers is not None else ""}"
            + f"{name}"
            + f".npy"
        )
        return pathToFile

    def fileExists(
        self, attribute: str = "", cut: bool = False, calcFileName: Optional[str] = None, **kwargs
    ) -> bool:
        if calcFileName is None:
            calcFileName = self.generateFileName(attribute, cut, **kwargs)
        if os.path.isfile(calcFileName):
            return True
        return False

    def loadFile(
        self,
        attribute: str = "",
        cut: bool = False,
        suppressText: bool = False,
        calcFileName: Optional[str] = None,
        **kwargs,
    ) -> npt.NDArray[Any]:
        if calcFileName is None:
            calcFileName = self.generateFileName(attribute, cut, **kwargs)
        if not suppressText:
            print(f"Loaded: {calcFileName}")
        return np.load(calcFileName, allow_pickle=True)

    def saveFile(
        self,
        array: npt.NDArray[Any],
        attribute: str = "",
        cut: bool = False,
        suppressText: bool = False,
        calcFileName: Optional[str] = None,
        **kwargs,
    ) -> None:
        if calcFileName is None:
            calcFileName = self.generateFileName(attribute, cut, **kwargs)
        self.makeDirIfNeeded(calcFileName)
        np.save(calcFileName, array)
        if not suppressText:
            print(f"Saved: {calcFileName}")

    def makeDirIfNeeded(self, fileName) -> None:
        dirName = "/".join(fileName.split("/")[:-1])
        if not os.path.isdir(dirName):
            os.makedirs(dirName)


class rawDataFileManager:
    def __init__(self, pathToData: str, columns: list[str], maxLine: Optional[int] = None):
        self.pathToData = pathToData
        self.columns = columns
        self.maxLine = maxLine

    def readFile(self) -> pd.DataFrame:
        print(f"Opening: {self.pathToData}")
        dtypes = {
            "PackageID": int,
            "Layer": int,
            "Column": int,
            "Row": int,
            "TS": int,
            "TS1": int,
            "TS2": int,
            "TriggerTS": int,
            "TriggerID": int,
            "ext_TS": int,
            "ext_TS2": int,
            "FIFO_overflow": bool,
        }
        return pd.read_csv(
            self.pathToData,
            delimiter="\t",
            names=self.columns,
            dtype=dtypes,
            header=0,
            nrows=self.maxLine,
        )

    def saveFile(self, dataFrame: pd.DataFrame, path: str, name: str) -> None:
        self.makeDirIfNeeded(path)
        with open(f"{path}{name}_decode.dat", "w") as file:
            file.write(
                "# PackageID; Layer; Column; Row; TS; TS1; TS2; TriggerTS; TriggerID; ext. TS; ext. TS2; FIFO overflow\n"
            )
            dataFrame.astype(int).to_csv(file, sep="\t", header=False, index=False)
            print(f"Save to csv: {path}{name}_decode.dat")

    def makeDirIfNeeded(self, fileName: str) -> None:
        dirName = "/".join(fileName.split("/")[:-1])
        if not os.path.isdir(dirName):
            os.makedirs(dirName)


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
            pathToData, self.baseAttrNames, maxLine=maxLine, telescope=self.telescope
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
            self.getCrossTalk()
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

    def getCrossTalk(self, recalc: bool = False) -> npt.NDArray[np.bool_]:
        if "crossTalk" in self.__dict__ and not recalc:
            toBeReturned = self.crossTalk
            self.clusterHandler.setCalcCrossTalk(self.crossTalk)
        elif self.calcFileManager.fileExists("crossTalk") and not recalc:
            self.crossTalk: npt.NDArray[np.bool_] = self.calcFileManager.loadFile("crossTalk")
            if self.telescope == "lancs":
                row = 248
                self.dataFrameHandler.cutSensor(row)
            toBeReturned = self.crossTalk
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
        print_mem_usage()
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
        baseAttrNames: list[str],
        maxLine: Optional[int] = None,
        telescope: str = "kit",
    ):
        self.dataFileManager = rawDataFileManager(pathToData, baseAttrNames, maxLine=maxLine)
        self.telescope = telescope
        self.dataLength: int = 0

    def loadDataIfNotLoaded(self) -> None:
        if not self.checkLoadedData():
            self.data = self.dataFileManager.readFile()

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
        self.haveClusters = False
        self.haveCrossTalk = False
        self.crossTalkFinder = crossTalkFinder()
        self.calcFileManager = calcFileManager
        self.dataFrameHandler = dataFrameHandler
        self.clusters: clusterArray

    def foundClusters(self) -> bool:
        return self.haveClusters

    def getClusters(self, layers: Optional[list[int]] = None, recalc: bool = False) -> clusterArray:
        if "clusters" in self.__dict__ and not recalc:
            toBeReturned = self.clusters
        elif self.calcFileManager.fileExists("clusters") and not recalc:
            clusters = self.calcFileManager.loadFile("clusters")
            self.clusters = self.initClusters(clusters)
            toBeReturned = self.clusters
        else:
            Layers = self.dataFrameHandler.readDataFrameAttr("Layer")
            TriggerIDs = self.dataFrameHandler.readDataFrameAttr("TriggerID")
            TSs = self.dataFrameHandler.readDataFrameAttr("TS")
            print(f"Finding clusters in {len(Layers)} lines")
            clusters = calcClusters(Layers, TriggerIDs, TSs)
            self.calcFileManager.saveFile(clusters, attribute="clusters")
            ext_TSs = self.dataFrameHandler.readDataFrameAttr("ext_TS")
            self.clusters = self.initClusters(clusters, layers=Layers, TSs=ext_TSs)
            toBeReturned = self.clusters
        filter = self.layerFilter(self.clusters, layers=layers)
        print_mem_usage()
        return toBeReturned[filter]

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
        self.clusters = np.empty(len(clusters), dtype=object)
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
            self.clusters[i] = clusterClass(
                i,
                clusters[i],
                layer,
                columns[clusters[i]],
                rows[clusters[i]],
                ToTs[clusters[i]],
                TSs[clusters[i]],
            )
        self.haveClusters = True
        return self.clusters

    def initClusterVoltages(
        self, Hit_Voltages: npt.NDArray[np.float64], Hit_VoltageErrors: npt.NDArray[np.float64]
    ) -> None:
        if not self.haveClusters:
            self.getClusters()
        for cluster in self.clusters:
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
                [cluster.getSize(excludeCrossTalk) for cluster in self.clusters]
            )
            filter = self.layerFilter(self.clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "ColumnWidths":
            toBeReturned = np.array(
                [cluster.getColumnWidth(excludeCrossTalk) for cluster in self.clusters]
            )
            filter = self.layerFilter(self.clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "RowWidths":
            toBeReturned = np.array(
                [cluster.getRowWidth(excludeCrossTalk) for cluster in self.clusters]
            )
            filter = self.layerFilter(self.clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "TSs":
            toBeReturned = np.array(
                [np.min(cluster.getTSs(excludeCrossTalk)) for cluster in self.clusters]
            )
            filter = self.layerFilter(self.clusters, layers=layers)
        elif attribute == "Times":
            toBeReturned = np.array(
                [np.min(cluster.getTSs(excludeCrossTalk)) for cluster in self.clusters]
            )
            toBeReturned = TStoMS(toBeReturned.astype(int) - int(np.min(toBeReturned)))
            filter = self.layerFilter(self.clusters, layers=layers)
        if returnIndexes:
            indexes = np.arange(len(self.clusters))
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


class crossTalkFinder:
    def __init__(self) -> None:
        self.crossTalkDict: dict[int, npt.NDArray[np.int_]]

    def findCrossTalk_OneCluster(self, cluster: clusterClass) -> npt.NDArray[np.bool_]:
        crossTalk = np.full(len(cluster.getShortIndexes()), False, dtype=bool)
        ToTs = cluster.getToTs()
        shortIndexes = cluster.getShortIndexes()
        columns = cluster.getColumns()
        rows = cluster.getRows()
        for pixel in shortIndexes:
            if not crossTalk[pixel]:
                expectedCrossTalk = self.crossTalkFunction()[rows[pixel]]
                for cross in expectedCrossTalk:
                    clashPixels = shortIndexes[
                        (columns == columns[pixel]) & (rows == cross[0]) & (np.invert(crossTalk))
                    ]
                    for clash in clashPixels:
                        if ToTs[pixel] < 30 and ToTs[clash] < 30:
                            pass
                        elif ToTs[clash] >= 255 and ToTs[pixel] >= 30 and ToTs[pixel] < 255:
                            crossTalk[clash] = True
                        elif ToTs[pixel] >= ToTs[clash] and ToTs[pixel] < 255 and ToTs[clash] < 30:
                            crossTalk[clash] = True
        return crossTalk

    def crossTalkFunction(self) -> dict[int, npt.NDArray[np.int_]]:
        if "crossTalkDict" in self.__dict__:
            return self.crossTalkDict
        else:
            crossTalkArray = self.calcCrossTalkArray()
            self.crossTalkDict = {}
            for i in range(372):
                expectedCrossTalk = crossTalkArray[np.where(crossTalkArray[:, :] == i)[0]]
                for expected in expectedCrossTalk:
                    for j in range(len(expected)):
                        if i in self.crossTalkDict:
                            if (
                                expected[j] not in [i - 2, i - 1, i, i + 1, i + 2]
                                and expected[j] != -1
                            ):
                                self.crossTalkDict[i] = np.append(
                                    self.crossTalkDict[i],
                                    [
                                        [
                                            expected[j],
                                            i,
                                        ]
                                    ],
                                    axis=0,
                                )
                        else:
                            if (
                                expected[j] not in [i - 2, i - 1, i, i + 1, i + 2]
                                and expected[j] != -1
                            ):
                                self.crossTalkDict[i] = np.array(
                                    [
                                        [
                                            expected[j],
                                            i,
                                        ]
                                    ]
                                )
                if i not in self.crossTalkDict:
                    self.crossTalkDict[i] = np.array([])
            return self.crossTalkDict

    def calcCrossTalkArray(self) -> npt.NDArray[np.int_]:
        # These numbers are all manually set based on the RowRow correlation graph
        up_to = 124
        crossTalkArray = np.full((up_to, 9), -1)
        crossTalkArray[:, 0] = np.arange(0, up_to)
        for row in range(1, up_to):
            expectedCrossTalk = []
            if row <= 18:
                expectedCrossTalk.append(248 - row + 17)
                expectedCrossTalk.append(248 - row + 18)
                expectedCrossTalk.append(248 - row + 19)
            if row >= 18 and row <= 52:
                expectedCrossTalk.append(row + 247)
                expectedCrossTalk.append(row + 248)
                expectedCrossTalk.append(row + 249)
            elif row >= 53 and row <= 104:
                expectedCrossTalk.append(row + 247)
                expectedCrossTalk.append(row + 248)
                expectedCrossTalk.append(row + 249)
            elif row >= 105 and row <= 123:
                expectedCrossTalk.append(104 - row + 372)
                expectedCrossTalk.append(104 - row + 371)
                if row >= 106:
                    expectedCrossTalk.append(104 - row + 373)
            if row <= 12:
                expectedCrossTalk.append(198 - row)
                expectedCrossTalk.append(197 - row)
            elif row >= 13 and row <= 51:
                expectedCrossTalk.append(row + 185)
                expectedCrossTalk.append(row + 186)
            elif row >= 54 and row <= 61:
                expectedCrossTalk.append(row + 185)
                expectedCrossTalk.append(row + 186)
            elif row >= 62 and row <= 104:
                expectedCrossTalk.append(row + 80)
                expectedCrossTalk.append(row + 81)
                expectedCrossTalk.append(row + 82)
            elif row >= 106 and row <= 122:
                expectedCrossTalk.append(row + 19)
            elif row == 123:
                expectedCrossTalk.append(row + 62)
            for i in range(len(expectedCrossTalk)):
                crossTalkArray[row, i + 1] = expectedCrossTalk[i]
        crossTalkArray[np.where(crossTalkArray[:, :] > 372)] = -1
        crossTalkArray = crossTalkArray.astype(int)
        crossTalkArray = np.append(
            crossTalkArray, np.append([[248, 267]], np.full((1, 7), -1), axis=1), axis=0
        )
        return crossTalkArray


def filterDataFiles(allDataFiles: list[dataAnalysis], filterDict: dict = {}) -> list[dataAnalysis]:
    dataFiles = []
    for dataFile in allDataFiles:
        boolList = []
        for f in filterDict.keys():
            attr = getattr(dataFile, "get_" + f)
            try:
                boolList.append((np.isin(attr(), filterDict[f])))
            except:
                boolList.append((attr() == filterDict[f]))
        if np.all(boolList):
            dataFiles.append(dataFile)
    sortIndex = np.argsort(
        np.array([dataFile.get_angle() * 1000 + dataFile.get_voltage() for dataFile in dataFiles])
    )
    returnDataFiles = np.array(dataFiles)[sortIndex]
    for dataFile in returnDataFiles:
        if dataFile.get_telescope() == "lancs":
            dataFile.dataHandler.getCrossTalk()
    return list(np.flip(returnDataFiles))


def initDataFiles(config: dict = {}) -> list[dataAnalysis]:
    if config == {}:
        config = configLoader.defaultConfig()
    files = glob(f"{config["pathToData"]}{config["fileFormate"]}")
    allDataFiles = [
        dataAnalysis(pathToDataFile, config["pathToCalcData"], maxLine=config["maxLine"])
        for pathToDataFile in files
    ]
    dataFiles = filterDataFiles(
        allDataFiles,
        filterDict=config["filterDict"],
    )
    return dataFiles


if __name__ == "__main__":
    pathToData = "/home/atlas/rballard/for_magda/data/Cut/202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
    pathToOutput = "/home/atlas/rballard/AtlasDataAnalysis/output"
    pathToCalcData = "/home/atlas/rballard/AtlasDataAnalysis/calculatedData"
    dataFile = dataAnalysis(pathToData, pathToCalcData, maxLine=100000)
    print(dataFile.get_angle())
    print(dataFile.get_base_attr("ToT")[0].shape)
    print(dataFile.get_base_attr("Hit_Voltage", returnIndexes=True, layers=[4])[0])
    print(dataFile.get_base_attr("ToT", excludeCrossTalk=True)[0].shape)
    print(dataFile.get_cluster_attr("Sizes")[0].shape)
    print(dataFile.get_cluster_attr("RowWidths", excludeCrossTalk=True)[0].shape)
    print(dataFile.get_cluster_attr("ColumnWidths", excludeCrossTalk=True, layers=[4])[0].shape)
    print(dataFile.get_cluster_attr("Sizes", excludeCrossTalk=True)[0].shape)
    print(dataFile.get_cluster_attr("Sizes", excludeCrossTalk=True, layers=[4])[0].shape)
