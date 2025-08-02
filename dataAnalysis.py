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
from numba import njit, types
from numba.typed import Dict
import numba
import configLoader

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
            if kwargs["layer"] is None:
                kwargs["layers"] = None
                kwargs.pop("layer")
            else:
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
            if kwargs["layer"] is None:
                kwargs["layers"] = None
                kwargs.pop("layer")
            else:
                kwargs["layers"] = [kwargs["layer"]]
                kwargs.pop("layer")
        return self.dataHandler.getClustersAttr(
            attribute, excludeCrossTalk=excludeCrossTalk, **kwargs
        )

    def get_crossTalk(self, recalc: bool = False, **kwargs) -> npt.NDArray[np.bool_]:
        if "layer" in kwargs:
            if kwargs["layer"] is None:
                kwargs["layers"] = None
                kwargs.pop("layer")
            else:
                kwargs["layers"] = [kwargs["layer"]]
                kwargs.pop("layer")
        return self.dataHandler.getCrossTalk(recalc=recalc, **kwargs)

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
                [np.min([cluster.getTSs(excludeCrossTalk)]) for cluster in self.clusters]
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
        # Precompute the cross-talk dictionary once
        raw_dict = self._build_crosstalk_dict()
        # Convert to numba-compatible format for JIT compilation
        self.crossTalkDict = self._convert_to_numba_dict(raw_dict)

    def findCrossTalk_OneCluster(self, cluster) -> npt.NDArray[np.bool_]:
        shortIndexes = cluster.getShortIndexes()
        ToTs = cluster.getToTs()
        columns = cluster.getColumns()
        rows = cluster.getRows()
        
        # Use JIT-compiled function for the heavy lifting
        return self._find_crosstalk_jit(
            np.ascontiguousarray(ToTs, dtype=np.int32),
            np.ascontiguousarray(columns, dtype=np.int32),
            np.ascontiguousarray(rows, dtype=np.int32),
            self.crossTalkDict
        )

    @staticmethod
    @njit(cache=True)
    def _find_crosstalk_jit(ToTs, columns, rows, crossTalkDict):
        n_pixels = len(ToTs)
        crossTalk = np.zeros(n_pixels, dtype=numba.boolean)
        
        for pixel_idx in range(n_pixels):
            if crossTalk[pixel_idx]:
                continue
                
            pixel_row = rows[pixel_idx]
            pixel_col = columns[pixel_idx]
            pixel_tot = ToTs[pixel_idx]
            
            # Skip if no cross-talk expected for this row
            if pixel_row not in crossTalkDict:
                continue
                
            expected_crosstalk = crossTalkDict[pixel_row]
            if len(expected_crosstalk) == 0:
                continue
            
            # Check each expected cross-talk row
            for cross_talk_idx in range(len(expected_crosstalk)):
                cross_row = expected_crosstalk[cross_talk_idx][0]
                
                # Find clash pixels: same column, target row, not already marked
                for clash_idx in range(n_pixels):
                    if (crossTalk[clash_idx] or 
                        clash_idx == pixel_idx or 
                        columns[clash_idx] != pixel_col or 
                        rows[clash_idx] != cross_row):
                        continue
                    
                    clash_tot = ToTs[clash_idx]
                    
                    # Apply cross-talk detection logic
                    if not (pixel_tot < 30 and clash_tot < 30):
                        if ((clash_tot >= 255 and 30 <= pixel_tot < 255) or
                            (pixel_tot >= clash_tot and pixel_tot < 255 and clash_tot < 30)):
                            crossTalk[clash_idx] = True
        
        return crossTalk

    def _convert_to_numba_dict(self, raw_dict):
        """Convert Python dict to numba-compatible typed dict"""
        # Create numba typed dict
        nb_dict = Dict.empty(
            key_type=types.int32,
            value_type=types.int32[:, :]
        )
        
        for key, value in raw_dict.items():
            if value.size > 0:
                nb_dict[key] = value.astype(np.int32)
            else:
                nb_dict[key] = np.empty((0, 2), dtype=np.int32)
        
        return nb_dict

    def _build_crosstalk_dict(self) -> dict[int, npt.NDArray[np.int_]]:
        """Build cross-talk dictionary more efficiently"""
        crossTalkArray = self.calcCrossTalkArray()
        crossTalkDict = {}
        
        # Pre-allocate lists for better performance
        for i in range(372):
            pairs = []
            
            # Find all occurrences of i in the array
            rows_with_i, cols_with_i = np.where(crossTalkArray == i)
            
            for row_idx in rows_with_i:
                row_data = crossTalkArray[row_idx]
                for val in row_data:
                    if (val != -1 and 
                        not (i - 2 <= val <= i + 2)):
                        pairs.append([val, i])
            
            crossTalkDict[i] = np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
        
        return crossTalkDict

    def calcCrossTalkArray(self) -> npt.NDArray[np.int_]:
        """Optimized array calculation with vectorization where possible"""
        up_to = 124
        crossTalkArray = np.full((up_to, 9), -1, dtype=np.int32)
        crossTalkArray[:, 0] = np.arange(0, up_to)
        
        # Vectorize some of the repetitive calculations
        rows = np.arange(1, up_to)
        
        # Process ranges more efficiently
        for row in rows:
            col_idx = 1
            
            # Group conditions and batch process
            if row <= 18:
                vals = [248 - row + 17, 248 - row + 18, 248 - row + 19]
                crossTalkArray[row, col_idx:col_idx+3] = vals
                col_idx += 3
                
            if 18 <= row <= 104:
                vals = [row + 247, row + 248, row + 249]
                crossTalkArray[row, col_idx:col_idx+3] = vals
                col_idx += 3
            elif 105 <= row <= 123:
                vals = [104 - row + 372, 104 - row + 371]
                if row >= 106:
                    vals.append(104 - row + 373)
                crossTalkArray[row, col_idx:col_idx+len(vals)] = vals
                col_idx += len(vals)
            
            # Additional conditions
            if row <= 12:
                vals = [198 - row, 197 - row]
                crossTalkArray[row, col_idx:col_idx+2] = vals
                col_idx += 2
            elif 13 <= row <= 51 or 54 <= row <= 61:
                vals = [row + 185, row + 186]
                crossTalkArray[row, col_idx:col_idx+2] = vals
                col_idx += 2
            elif 62 <= row <= 104:
                vals = [row + 80, row + 81, row + 82]
                crossTalkArray[row, col_idx:col_idx+3] = vals
                col_idx += 3
            elif 106 <= row <= 122:
                crossTalkArray[row, col_idx] = row + 19
            elif row == 123:
                crossTalkArray[row, col_idx] = row + 62
        
        # Vectorized cleanup
        crossTalkArray[crossTalkArray > 372] = -1
        
        # Add final row efficiently
        final_row = np.full((1, 9), -1, dtype=np.int32)
        final_row[0, :2] = [248, 267]
        
        return np.vstack([crossTalkArray, final_row])

    # Fallback method without numba (if numba not available)
    def findCrossTalk_OneCluster_fallback(self, cluster) -> npt.NDArray[np.bool_]:
        shortIndexes = cluster.getShortIndexes()
        n_pixels = len(shortIndexes)
        crossTalk = np.zeros(n_pixels, dtype=bool)
        ToTs = cluster.getToTs()
        columns = cluster.getColumns()
        rows = cluster.getRows()
        
        # Create column-to-indices mapping for O(1) lookup
        col_to_indices = {}
        for idx, col in enumerate(columns):
            if col not in col_to_indices:
                col_to_indices[col] = []
            col_to_indices[col].append(idx)
        
        for pixel_idx in range(n_pixels):
            if crossTalk[pixel_idx]:
                continue
                
            pixel_row = rows[pixel_idx]
            pixel_col = columns[pixel_idx]
            pixel_tot = ToTs[pixel_idx]
            
            # Get expected cross-talk (from precomputed dict)
            expected_crosstalk = self.crossTalkDict.get(pixel_row)
            if expected_crosstalk is None or len(expected_crosstalk) == 0:
                continue
            
            # Only check pixels in the same column
            same_col_indices = col_to_indices[pixel_col]
            
            for cross_row, _ in expected_crosstalk:
                for clash_idx in same_col_indices:
                    if (crossTalk[clash_idx] or 
                        clash_idx == pixel_idx or 
                        rows[clash_idx] != cross_row):
                        continue
                    
                    clash_tot = ToTs[clash_idx]
                    
                    # Optimized conditions
                    if not (pixel_tot < 30 and clash_tot < 30):
                        if ((clash_tot >= 255 and 30 <= pixel_tot < 255) or
                            (pixel_tot >= clash_tot and pixel_tot < 255 and clash_tot < 30)):
                            crossTalk[clash_idx] = True
        
        return crossTalk

    # Keep original interface
    def crossTalkFunction(self) -> dict[int, npt.NDArray[np.int_]]:
        # Convert back to original format if needed
        result = {}
        for key in self.crossTalkDict:
            result[key] = np.array(self.crossTalkDict[key])
        return result

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
    files = glob(f"{config["pathToData"]}{config["fileFormat"]}")
    allDataFiles = [
        dataAnalysis(pathToDataFile, config["pathToCalcData"], maxLine=config["maxLine"])
        for pathToDataFile in files
    ]
    dataFiles = filterDataFiles(
        allDataFiles,
        filterDict=config["filterDict"],
    )
    return dataFiles
