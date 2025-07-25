from lowLevelFunctions import isFiltered,readFileName,print_mem_usage,calcToT,calcHit_VoltageError,calcHit_Voltage,TStoMS,calcClusters,trueTimeStamps
import os
import pandas as pd
import numpy as np
from glob import glob
import configLoader
class dataAnalysis:
    def __init__(self, pathToData: str, pathToCalcData: str, maxLine: int = None):
        self.dataHandler = dataHandler(pathToData, pathToCalcData, maxLine=maxLine)
        self.pathToData = pathToData

    def get_voltage(self):
        return self.dataHandler.voltage

    def get_angle(self):
        return self.dataHandler.angle

    def get_fileName(self):
        return self.dataHandler.fileName

    def get_telescope(self):
        return self.dataHandler.telescope

    def check_if_filtered(self, filterDict):
        return isFiltered(self, filter_dict=filterDict)

    def get_dataFrame(self):
        return self.dataHandler.getDataFrame()

    def get_base_attr(self, attribute: str, **kwargs) -> np.ndarray:
        if "layer" in kwargs:
            kwargs["layers"] = [kwargs["layer"]]
            kwargs.pop("layer")
        return self.dataHandler.baseAttr(attribute, **kwargs)

    def get_clusters(self, **kwargs) -> np.ndarray:
        if "layer" in kwargs:
            if kwargs["layer"] is None:
                kwargs["layers"] = None
                kwargs.pop("layer")
            else:
                kwargs["layers"] = [kwargs["layer"]]
                kwargs.pop("layer")
        return self.dataHandler.getClusters(**kwargs)

    def get_cluster_attr(self, attribute, excludeCrossTalk: bool = False, **kwargs) -> np.ndarray:
        if "layer" in kwargs:
            kwargs["layers"] = [kwargs["layer"]]
            kwargs.pop("layer")
        return self.dataHandler.getClustersAttr(attribute, excludeCrossTalk=excludeCrossTalk, **kwargs)

    def get_crossTalk(self,recalc:bool=False):
        return self.dataHandler.getCrossTalk(recalc=recalc)

    def init_cluster_voltages(self):
        self.dataHandler.initClusterVoltages()

    def save_nonCrossTalk_to_csv(self, path):
        self.dataHandler.save_nonCrossTalk_to_csv(path,self.get_fileName())

class calcDataFileManager:
    def __init__(self, pathToCalcData: str, fileName: str, maxLine: int = None):
        self.pathToCalcData = pathToCalcData
        self.fileName = fileName
        self.maxLine = maxLine
        os.makedirs(f"{self.pathToCalcData}", exist_ok=True)

    def generateFileName(self, attribute: str, cut: bool = False, layers: list[int] = None, name: str = "", file: str = ""):
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

    def fileExists(self, attribute: str = None, cut: bool = False, calcFileName: str = None, **kwargs):
        if calcFileName is None:
            calcFileName = self.generateFileName(attribute, cut, **kwargs)
        if os.path.isfile(calcFileName):
            return True
        return False

    def loadFile(self, attribute: str = None, cut: bool = False, suppressText: bool = False, calcFileName: str = None, **kwargs):
        if calcFileName is None:
            calcFileName = self.generateFileName(attribute, cut, **kwargs)
        if not suppressText:
            print(f"Loaded: {calcFileName}")
        return np.load(calcFileName, allow_pickle=True)

    def saveFile(self, array: np.ndarray, attribute: str = None, cut: bool = False, suppressText: bool = False, calcFileName: str = None, **kwargs):
        if calcFileName is None:
            calcFileName = self.generateFileName(attribute, cut, **kwargs)
        self.makeDirIfNeeded(calcFileName)
        np.save(calcFileName, array)
        if not suppressText:
            print(f"Saved: {calcFileName}")

    def makeDirIfNeeded(self, fileName):
        dirName = "/".join(fileName.split("/")[:-1])
        if not os.path.isdir(dirName):
            os.makedirs(dirName)

class dataHandler:
    def __init__(self, pathToData: str, pathToCalcData: str, maxLine: int = None):
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
        self.dataFrameHandler = dataFrameHandler(pathToData, self.baseAttrNames, maxLine=maxLine, telescope=self.telescope)
        self.calcFileManager = calcDataFileManager(pathToCalcData, self.fileName, maxLine)
        self.clusterHandler = clusterHandler(self.calcFileManager, self.dataFrameHandler)

    def getDataFrame(self):
        return self.dataFrameHandler.data

    def baseAttr(self, attribute: str, excludeCrossTalk: bool = False, layers: list = None, returnIndexes: bool = False, recalc=False) -> "np.array":
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
                    toBeReturned = attr(ToTs=self.baseAttr("ToT"))
                else:
                    toBeReturned = attr()
                getattr(self, attribute, toBeReturned)
                self.calcFileManager.saveFile(toBeReturned, attribute=attribute)
        toBeReturned = self.layerCrosstalkFilter(toBeReturned, excludeCrossTalk, layers, returnIndexes=returnIndexes)
        if returnIndexes:
            (toBeReturned, indexes) = toBeReturned
            return toBeReturned, indexes
        else:
            return toBeReturned

    def getClusters(self, excludeCrossTalk: bool = False, **kwargs):
        if excludeCrossTalk:
            self.getCrossTalk()
        return self.clusterHandler.getClusters(excludeCrossTalk=excludeCrossTalk, **kwargs)

    def getClustersAttr(self, attribute, excludeCrossTalk: bool = False, returnIndexes: bool = False, layers: list[int] = None):
        if self.calcFileManager.fileExists(attribute, cut=excludeCrossTalk, layers=layers) and (
            not returnIndexes or self.calcFileManager.fileExists("clusterIndexes", cut=excludeCrossTalk, layers=layers)
        ):
            toBeReturned = self.calcFileManager.loadFile(attribute, cut=excludeCrossTalk, layers=layers, suppressText=True)
            if returnIndexes:
                indexes = self.calcFileManager.loadFile("clusterIndexes", cut=excludeCrossTalk, layers=layers, suppressText=True)
        else:
            if not self.clusterHandler.haveClusters:
                self.getClusters()
            if excludeCrossTalk and not self.clusterHandler.haveCrossTalk:
                self.getCrossTalk()
            if returnIndexes:
                toBeReturned, indexes = self.clusterHandler.getClusterAttr(
                    attribute, excludeCrossTalk=excludeCrossTalk, returnIndexes=returnIndexes, layers=layers
                )
                self.calcFileManager.saveFile(toBeReturned, attribute=attribute, cut=excludeCrossTalk, layers=layers)
                self.calcFileManager.saveFile(indexes, "clusterIndexes", cut=excludeCrossTalk, layers=layers)
            else:
                toBeReturned = self.clusterHandler.getClusterAttr(
                    attribute, excludeCrossTalk=excludeCrossTalk, returnIndexes=returnIndexes, layers=layers
                )
                self.calcFileManager.saveFile(toBeReturned, attribute=attribute, cut=excludeCrossTalk, layers=layers)
        if returnIndexes:
            return toBeReturned, indexes
        else:
            return toBeReturned

    def initClusterVoltages(self):
        self.clusterHandler.initClusterVoltages(self.baseAttr("Hit_Voltage"), self.baseAttr("Hit_VoltageError"))

    def notCrossTalk(self):
        return np.invert(self.getCrossTalk())

    def getCrossTalk(self,recalc:bool=False):
        if "crossTalk" in self.__dict__ and not recalc:
            toBeReturned = self.crossTalk
            self.clusterHandler.setCalcCrossTalk(self.crossTalk)
        elif self.calcFileManager.fileExists("crossTalk") and not recalc:
            self.crossTalk = self.calcFileManager.loadFile("crossTalk")
            if self.telescope == "lancs":
                row = 248
                self.dataFrameHandler.cutSensor(row)
            toBeReturned = self.crossTalk
            self.clusterHandler.setCalcCrossTalk(self.crossTalk)
        else:
            self.crossTalk = self.clusterHandler.calcCrossTalk()
            if self.telescope == "lancs":
                row = 248
                self.crossTalk = self.crossTalk[np.where(self.baseAttr("Row") < row)[0]]
                self.dataFrameHandler.cutSensor(row)
                self.getClusters(recalc=True)
                self.baseAttr("ToT", recalc=True)
                self.baseAttr("Hit_Voltage", recalc=True)
                self.clusterHandler.setCalcCrossTalk(self.crossTalk)
            self.calcFileManager.saveFile(self.crossTalk, attribute="crossTalk")
            toBeReturned = self.crossTalk
        print_mem_usage()
        return toBeReturned

    def cutArrayCrossTalkFilter(self, array: list, excludeCrossTalk: bool = True):
        if excludeCrossTalk:
            includedIndexes = self.notCrossTalk()
        else:
            includedIndexes = np.full(len(array), True, dtype=bool)
        return includedIndexes

    def layerFilter(self, array: list, layers: list = None):
        if layers is None:
            includedIndexes = np.full(len(array), True, dtype=bool)
        else:
            layersArray = self.baseAttr("Layer")
            includedIndexes = np.isin(layersArray, layers, assume_unique=True, kind="table")
        return includedIndexes

    def layerCrosstalkFilter(self, array: list, excludeCrossTalk: bool = False, layers: list = None, returnIndexes: bool = False):
        if layers is not None or excludeCrossTalk is not None:
            filter = (self.cutArrayCrossTalkFilter(array, excludeCrossTalk)) & (self.layerFilter(array, layers))
        else:
            filter = np.full(len(array), True, dtype=bool)
        array = array[filter]
        if returnIndexes:
            indexes = np.arange(len(filter))[filter]
            array = (array, indexes)
        return array

    def save_nonCrossTalk_to_csv(self, path,name):
        self.dataFrameHandler.saveNonCrossTalkToCSV(self.getCrossTalk(), path,name,self.getClusters())


class dataFrameHandler:
    # Stores the data frame
    # Retrieves calculated data for values on a per hit basis i.e. one value per line in data frame e.g. ToT, Voltage
    def __init__(self, pathToData: str, baseAttrNames: list, maxLine: int = None, telescope: str = "kit"):
        self.dataFileManager = rawDataFileManager(pathToData, baseAttrNames, maxLine=maxLine)
        self.telescope = telescope

    def loadDataIfNotLoaded(self):
        if not self.checkLoadedData():
            self.data = self.dataFileManager.readFile()

    def cutSensor(self, row):
        self.loadDataIfNotLoaded()
        self.data = self.data.drop(index=np.where(self.data["Row"].values >= row)[0])
        self.data.index = np.arange(len(self.data))

    def checkLoadedData(self):
        if "data" in self.__dict__.keys():
            return True
        else:
            return False

    def readDataFrameAttr(self, attribute):
        self.loadDataIfNotLoaded()
        toBeReturned = self.data[attribute].values
        return toBeReturned

    def getToT(self):
        self.loadDataIfNotLoaded()
        toBeReturned = calcToT(self.data["TS"].values, self.data["TS2"].values)
        return toBeReturned

    def getHit_Voltage(self, ToTs=None):
        self.loadDataIfNotLoaded()
        if ToTs is None:
            ToTs = self.getToT()
        rows = self.data["Row"].values
        columns = self.data["Column"].values
        Layers = self.data["Layer"].values
        toBeReturned = calcHit_Voltage(rows, columns, ToTs, Layers)
        return toBeReturned

    def getHit_VoltageError(self, ToTs=None):
        self.loadDataIfNotLoaded()
        if ToTs is None:
            ToTs = self.getToT()
        rows = self.data["Row"].values
        columns = self.data["Column"].values
        Layers = self.data["Layer"].values
        toBeReturned = calcHit_VoltageError(rows, columns, ToTs, Layers)
        return toBeReturned

    def getDataLength(self):
        self.loadDataIfNotLoaded()
        if "dataLength" in self.__dict__:
            return self.dataLength
        else:
            self.dataLength = len(self.data)
            return self.dataLength

    def saveNonCrossTalkToCSV(self, crosstalk, path,name,clusters):
        outputDF = self.data
        outputDF["ext_TS"] = trueTimeStamps(clusters,outputDF["ext_TS"].values)
        outputDF = outputDF.drop(index=np.where(crosstalk)[0])
        self.dataFileManager.saveFile(outputDF, path,name)


class clusterHandler:
    def __init__(self, calcFileManager : calcDataFileManager, dataFrameHandler : dataFrameHandler):
        self.haveClusters = False
        self.haveCrossTalk = False
        self.crossTalkFinder = crossTalkFinder()
        self.calcFileManager = calcFileManager
        self.dataFrameHandler = dataFrameHandler

    def foundClusters(self):
        return self.haveClusters

    def getClusters(self, excludeCrossTalk: bool = False, returnIndexes: bool = False, layers: list[int] = None, recalc: bool = False):
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
        if returnIndexes:
            indexes = np.arange(len(self.clusters))
            return toBeReturned[filter], indexes[filter]
        else:
            return toBeReturned[filter]

    def initClusters(self, clusters, layers=None, columns=None, rows=None, ToTs=None, TSs=None):
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
            self.clusters[i] = clusterClass(i, clusters[i], layer, columns[clusters[i]], rows[clusters[i]], ToTs[clusters[i]], TSs[clusters[i]])
        self.haveClusters = True
        return self.clusters

    def initClusterVoltages(self, Hit_Voltages, Hit_VoltageErrors):
        if not self.haveClusters:
            self.getClusters()
        for cluster in self.clusters:
            cluster.setHit_Voltage(Hit_Voltages[cluster.indexes], Hit_VoltageErrors[cluster.indexes])

    def setCalcCrossTalk(self, crosstalk):
        clusters = self.getClusters()
        for cluster in clusters:
            cluster.setCrossTalk(crosstalk[cluster.indexes])
        self.haveCrossTalk = True

    def calcCrossTalk(self):
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

    def getClusterAttr(self, attribute, excludeCrossTalk: bool = False, returnIndexes: bool = False, layers: list[int] = None):
        if attribute == "Sizes":
            toBeReturned = np.array([cluster.getSize(excludeCrossTalk) for cluster in self.clusters])
            filter = self.layerFilter(self.clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "ColumnWidths":
            toBeReturned = np.array([cluster.getColumnWidth(excludeCrossTalk) for cluster in self.clusters])
            filter = self.layerFilter(self.clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "RowWidths":
            toBeReturned = np.array([cluster.getRowWidth(excludeCrossTalk) for cluster in self.clusters])
            filter = self.layerFilter(self.clusters, layers=layers) & (toBeReturned > 0)
        elif attribute == "TSs":
            toBeReturned = np.array([np.min(cluster.getTSs(excludeCrossTalk)) for cluster in self.clusters])
            filter = self.layerFilter(self.clusters, layers=layers)
        elif attribute == "Times":
            toBeReturned = np.array([np.min(cluster.getTSs(excludeCrossTalk)) for cluster in self.clusters])
            toBeReturned = TStoMS(toBeReturned - np.min(toBeReturned))
            filter = self.layerFilter(self.clusters, layers=layers)
        if returnIndexes:
            indexes = np.arange(len(self.clusters))
            return toBeReturned[filter], indexes[filter]
        else:
            return toBeReturned[filter]

    def layerFilter(self, clusters, layers: list[int] = None):
        if layers == None:
            return np.full(len(clusters), True, bool)
        else:
            return np.isin([cluster.layer for cluster in clusters], layers)


class clusterClass:
    def __init__(self, index: int, indexes: list[int], layer: int, columns: list[int], rows: list[int], ToTs: list[int], TSs: list[int]):
        self.indexes = indexes
        self.layer = layer
        self.columns = columns
        self.rows = rows
        self.ToTs = ToTs
        self.TSs = TSs
        self.index = index

    def setHit_Voltage(self, Hit_Voltages: list[float], Hit_VoltageErrors: list[float]):
        self.Hit_Voltages = Hit_Voltages
        self.Hit_VoltageErrors = Hit_VoltageErrors

    def setCrossTalk(self, crossTalk: list[bool]):
        self.crossTalk = crossTalk
        self.notCrossTalk = np.invert(crossTalk)

    def getIndex(self, excludeCrossTalk: bool = False) -> int:
        return self.index

    def getIndexes(self, excludeCrossTalk: bool = False) -> list[int]:
        if excludeCrossTalk:
            return self.indexes[self.notCrossTalk]
        else:
            return self.indexes

    def getLayer(self, excludeCrossTalk: bool = False) -> int:
        return self.layer

    def getShortIndexes(self, excludeCrossTalk: bool = False) -> list[int]:
        if excludeCrossTalk:
            return np.arange(self.getSize())[self.notCrossTalk]
        else:
            return np.arange(self.getSize())
    def getSize(self, excludeCrossTalk: bool = False) -> int:
        if excludeCrossTalk:
            return np.sum(self.notCrossTalk)
        else:
            return len(self.indexes)

    def getRowWidth(self, excludeCrossTalk: bool = False) -> int:
        if excludeCrossTalk:
            if self.rows[self.notCrossTalk].size == 0:
                return 0
            else:
                return np.ptp(self.rows[self.notCrossTalk]) + 1
        else:
            if self.rows.size == 0:
                return 0
            else:
                return np.ptp(self.rows) + 1

    def getColumnWidth(self, excludeCrossTalk: bool = False) -> int:
        if excludeCrossTalk:
            if self.columns[self.notCrossTalk].size == 0:
                return 0
            else:
                return np.ptp(self.columns[self.notCrossTalk]) + 1
        else:
            if self.columns.size == 0:
                return 0
            else:
                return np.ptp(self.columns) + 1

    def getColumns(self, excludeCrossTalk: bool = False) -> list[int]:
        if excludeCrossTalk:
            return self.columns[self.notCrossTalk]
        else:
            return self.columns

    def getRows(self, excludeCrossTalk: bool = False) -> list[int]:
        if excludeCrossTalk:
            return self.rows[self.notCrossTalk]
        else:
            return self.rows

    def getHit_Voltages(self, excludeCrossTalk: bool = False) -> list[float]:
        if excludeCrossTalk:
            return self.Hit_Voltages[self.notCrossTalk]
        else:
            return self.Hit_Voltages

    def getHit_VoltageErrors(self, excludeCrossTalk: bool = False) -> list[float]:
        if excludeCrossTalk:
            return self.Hit_VoltageErrors[self.notCrossTalk]
        else:
            return self.Hit_VoltageErrors

    def getToTs(self, excludeCrossTalk: bool = False) -> list[int]:
        if excludeCrossTalk:
            return self.ToTs[self.notCrossTalk]
        else:
            return self.ToTs

    def getToTErrors(self, excludeCrossTalk: bool = False) -> list[int]:
        if excludeCrossTalk:
            return np.full(self.ToTs.shape, 2)[self.notCrossTalk]
        else:
            return np.full(self.ToTs.shape, 2)

    def getTSs(self, excludeCrossTalk: bool = False) -> list[int]:
        if excludeCrossTalk:
            return self.TSs[self.notCrossTalk]
        else:
            return self.TSs


class rawDataFileManager:
    def __init__(self, pathToData: str, columns: list, maxLine: int = None):
        self.pathToData = pathToData
        self.columns = columns
        self.maxLine = maxLine

    def readFile(self):
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
        return pd.read_csv(self.pathToData, delimiter="\t", names=self.columns, dtype=dtypes, header=0, nrows=self.maxLine)

    def saveFile(self, dataFrame, path, name):
        self.makeDirIfNeeded(path)
        with open(f"{path}{name}_decode.dat", 'w') as file:
            file.write('# PackageID; Layer; Column; Row; TS; TS1; TS2; TriggerTS; TriggerID; ext. TS; ext. TS2; FIFO overflow\n')
            dataFrame.astype(int).to_csv(file, sep="\t", header=False, index=False)
            print(f"Save to csv: {path}{name}_decode.dat")

    def makeDirIfNeeded(self, fileName):
        dirName = "/".join(fileName.split("/")[:-1])
        if not os.path.isdir(dirName):
            os.makedirs(dirName)



class crossTalkFinder:
    def findCrossTalk_OneCluster(self, cluster: clusterClass) -> list[bool]:
        crossTalk = np.full(len(cluster.getShortIndexes()), False, dtype=bool)
        ToTs = cluster.getToTs()
        shortIndexes = cluster.getShortIndexes()
        columns = cluster.getColumns()
        rows = cluster.getRows()
        for pixel in shortIndexes:
            if not crossTalk[pixel]:
                expectedCrossTalk = self.crossTalkFunction(rows[pixel])
                for cross in expectedCrossTalk:
                    clashPixels = shortIndexes[(columns == columns[pixel]) & (rows == cross[0]) & (np.invert(crossTalk))]
                    for clash in clashPixels:
                        if ToTs[pixel] < 30 and ToTs[clash] < 30:
                            pass
                        elif ToTs[clash] >= 255 and ToTs[pixel] >= 30 and ToTs[pixel] < 255:
                            crossTalk[clash] = True
                        elif ToTs[pixel] >= ToTs[clash] and ToTs[pixel] < 255 and ToTs[clash] < 30:
                            crossTalk[clash] = True
        return crossTalk

    def crossTalkFunction(self, row: int, returnDict=False) -> list[int]:
        if "crossTalkDict" in self.__dict__:
            if returnDict:
                return self.crossTalkDict
            return self.crossTalkDict[int(row)]
        else:
            crossTalkArray = self.calcCrossTalkArray()
            self.crossTalkDict = {}
            for i in range(372):
                expectedCrossTalk = crossTalkArray[np.where(crossTalkArray[:, :] == i)[0]]
                for expected in expectedCrossTalk:
                    for j in range(len(expected)):
                        if i in self.crossTalkDict:
                            if expected[j] not in [i-2,i-1,i,i+1,i+2] and expected[j] != -1:
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
                            if expected[j] not in [i-2,i-1,i,i+1,i+2] and expected[j] != -1:
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
            if returnDict:
                return self.crossTalkDict
            return self.crossTalkDict[int(row)]
    def calcCrossTalkArray(self) -> list[list[int]]:
        # These numbers are all manually set based on the RowRow correlation graph
        up_to = 124
        crossTalkArray = np.full((up_to,9), -1)
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
        crossTalkArray = np.append(crossTalkArray, np.append([[248, 267]],np.full((1,7),-1),axis=1), axis=0)
        return crossTalkArray

def filterDataFiles(allDataFiles: list[dataAnalysis], filterDict: dict = {}):
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
    dataFiles = np.array(dataFiles)[np.argsort([dataFile.get_angle() * 1000 + dataFile.get_voltage() for dataFile in dataFiles])]
    for dataFile in dataFiles:
        if dataFile.get_telescope() == "lancs":
            dataFile.dataHandler.getCrossTalk()
    return np.flip(dataFiles)

def initDataFiles(config:dict={})->list[dataAnalysis]:
    if config == {}:
        config = configLoader.defaultConfig()
    files = glob(f"{config["pathToData"]}{config["fileFormate"]}")
    allDataFiles = [dataAnalysis(pathToDataFile, config["pathToCalcData"], maxLine=config["maxLine"]) for pathToDataFile in files]
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
    print(dataFile.get_base_attr("ToT").shape)
    print(dataFile.get_base_attr("Hit_Voltage", returnIndexes=True, layers=[4]))
    print(dataFile.get_base_attr("ToT", excludeCrossTalk=True).shape)
    print(dataFile.get_cluster_attr("Sizes").shape)
    print(dataFile.get_cluster_attr("RowWidths", excludeCrossTalk=True).shape)
    print(dataFile.get_cluster_attr("ColumnWidths", excludeCrossTalk=True, layers=[4]).shape)
    print(dataFile.get_cluster_attr("Sizes", excludeCrossTalk=True).shape)
    print(dataFile.get_cluster_attr("Sizes", excludeCrossTalk=True, layers=[4]).shape)
