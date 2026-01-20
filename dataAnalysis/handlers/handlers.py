from typing import Optional, Any
from dataAnalysis._types import dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    pd,  # pandas
    np,  # numpy
    npt,  # numpy.typing
)
from dataAnalysis._fileReader import calcDataFileManager
from ._functions import readFileName
from ._clusterClass import clusterClass
from ._handler_dataFrameHandler import dataFrameHandler
from ._handler_clusterHandler import clusterHandler
from ._handler_perfectClusterHandler import perfectClusterHandler


class dataHandler:
    def __init__(
        self,
        pathToData: str,
        pathToCalcData: str,
        maxLine: Optional[int] = None,
        baseAttrNames: list[str] = [
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
        ],
    ):
        self.fileName, self.voltage, self.angle, self.telescope = readFileName(pathToData)
        self.baseAttrNames = baseAttrNames
        self.dataFrameHandler = dataFrameHandler(
            pathToData, self.baseAttrNames, maxLine=maxLine, telescope=self.telescope
        )
        self.calcFileManager = calcDataFileManager(pathToCalcData, self.fileName, maxLine)
        self.clusterHandler = clusterHandler(self.calcFileManager, self.dataFrameHandler)
        self.perfectClusterHandler = perfectClusterHandler(
            self.calcFileManager, self.dataFrameHandler, self.clusterHandler
        )

    def getDataFrame(self) -> pd.DataFrame:
        if "data" not in self.dataFrameHandler.__dict__:
            self.dataFrameHandler.loadDataIfNotLoaded()
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
                setattr(self, attribute, toBeReturned)
            else:
                attr = getattr(self.dataFrameHandler, "get" + attribute)
                if attribute in ["Hit_Voltage", "Hit_VoltageError"]:
                    toBeReturned = attr(ToTs=self.baseAttr("ToT"))
                else:
                    toBeReturned = attr()
                setattr(self, attribute, toBeReturned)
                self.calcFileManager.saveFile(toBeReturned, attribute=attribute)
        toBeReturned, indexes = self.layerCrosstalkFilter(toBeReturned, excludeCrossTalk, layers)
        if returnIndexes:
            return toBeReturned, indexes
        else:
            return toBeReturned

    def getClusters(self, excludeCrossTalk: bool = False, **kwargs) -> clusterArray:
        if excludeCrossTalk:
            self.getCrossTalk(**kwargs)
        return self.clusterHandler.getClusters(excludeCrossTalk=excludeCrossTalk, **kwargs)

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
            self.baseAttr("Hit_Voltage"), self.baseAttr("Hit_VoltageError")
        )

    def notCrossTalk(self) -> npt.NDArray[np.bool_]:
        return np.invert(self.getCrossTalk())

    def getCrossTalk(
        self,
        recalc: bool = False,
        layers: Optional[list[int]] = None,
        initClusters: bool = True,
        returnIndexes: bool = False,
    ) -> npt.NDArray[np.bool_]:
        if "crossTalk" in self.__dict__ and not recalc:
            toBeReturned = self.crossTalk
            if initClusters:
                self.clusterHandler.setCalcCrossTalk(self.crossTalk)
        elif self.calcFileManager.fileExists("crossTalk") and not recalc:
            self.crossTalk: npt.NDArray[np.bool_] = self.calcFileManager.loadFile("crossTalk")
            toBeReturned = self.crossTalk
            if initClusters:
                self.clusterHandler.setCalcCrossTalk(self.crossTalk)
        else:
            self.crossTalk = self.clusterHandler.calcCrossTalk(recalc=recalc)
            self.calcFileManager.saveFile(self.crossTalk, attribute="crossTalk")
            toBeReturned = self.crossTalk
        toBeReturned, indexes = self.layerCrosstalkFilter(toBeReturned, False, layers)
        if returnIndexes:
            return (toBeReturned, indexes)
        else:
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
            layersArray = self.baseAttr("Layer")
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
        # self.getCrossTalk(recalc=True)
        self.dataFrameHandler.saveNonCrossTalkToCSV(
            self.getCrossTalk(), path, name, self.getClusters()
        )

    def getFlatClusters(self, width, excludeCrossTalk: bool = False, **kwargs) -> clusterArray:
        if excludeCrossTalk:
            self.getCrossTalk(**kwargs)
        return self.clusterHandler.getFlatClusters(
            width, excludeCrossTalk=excludeCrossTalk, **kwargs
        )

    def getTimeStampTemplate(
        self, maxRow=25, layers: Optional[list[int]] = None, excludeCrossTalk: bool = True, recalc=False, **kwargs
    ):
        calcFileName = self.calcFileManager.generateFileName(
            attribute="perfectClusterTemplate",
            layers=layers,
            name=f"_rows_{maxRow}",
        )
        fileCheck = self.calcFileManager.fileExists(calcFileName=calcFileName)

        if fileCheck and not recalc:
            estimate, spread = self.calcFileManager.loadFile(calcFileName=calcFileName)
        else:
            estimate, spread = self.perfectClusterHandler.getTimeStampTemplate(
                maxRow=maxRow, layers=layers, excludeCrossTalk=excludeCrossTalk,**kwargs
            )
            self.calcFileManager.saveFile([estimate, spread], calcFileName=calcFileName)
        return estimate, spread

    def getPerfectClusterIndexes(
        self,
        estimate,
        spread,
        minPval=0.5,
        layers: Optional[list[int]] = None,
        excludeCrossTalk: bool = True,
    ):
        if "perfectClusterIndexes" in self.__dict__:
            return self.perfectClusterIndexes
        self.initPerfectClusters(
            estimate,
            spread,
            minPval=minPval,
            layers = layers,
            excludeCrossTalk = excludeCrossTalk,
        )
        return self.perfectClusterIndexes
    
    def initPerfectClusters(
        self,
        estimate,
        spread,
        minPval=0.5,
        layers: Optional[list[int]] = None,
        excludeCrossTalk: bool = True,
    ):
        name = f"_minPval_{minPval}_rows_{len(estimate)-1}"
        calcFileName1 = self.calcFileManager.generateFileName(
            attribute="perfectClusterIndexes",
            layers=layers,
            name=name,
        )
        calcFileName2 = self.calcFileManager.generateFileName(
            attribute="perfectClusterData",
            layers=layers,
            name=name,
        )
        clusters = self.clusterHandler.getClusters(
            layers=layers, excludeCrossTalk=excludeCrossTalk
        )
        fileCheck1 = self.calcFileManager.fileExists(calcFileName=calcFileName1)
        fileCheck2 = self.calcFileManager.fileExists(calcFileName=calcFileName2)
        if fileCheck1 and fileCheck2:
            self.perfectClusterIndexes = self.calcFileManager.loadFile(calcFileName=calcFileName1)
            pVals,flippeds,perm,section = self.calcFileManager.loadFile(calcFileName=calcFileName2)
            for i,cluster in enumerate(clusters):
                cluster.pVal = pVals[i]
                cluster.flipped = flippeds[i]
                cluster.perm = perm[i]
                cluster.section = section[i]
        else:
            self.perfectClusterIndexes = self.perfectClusterHandler.getPerfectClusterIndexes(
                estimate, spread, minPval=minPval, layers=layers, excludeCrossTalk=excludeCrossTalk
            )
            self.calcFileManager.saveFile(
                self.perfectClusterIndexes, calcFileName=calcFileName1
            )
            pVals = np.array([cluster.pVal for cluster in clusters])
            flippeds = np.array([cluster.flipped for cluster in clusters])
            perm = np.array([cluster.perm for cluster in clusters],dtype=object)
            section = np.array([cluster.section for cluster in clusters],dtype=object)
            self.calcFileManager.saveFile(
                np.array([pVals,flippeds,perm,section],dtype=object), calcFileName=calcFileName2
            )
        return None
    
    def getPerfectClusters(
        self,
        estimate,
        spread,
        minPval=0.5,
        layers: Optional[list[int]] = None,
        excludeCrossTalk: bool = True,
    ):
        clusters = self.clusterHandler.getClusters(
            excludeCrossTalk=False
        )
        if "perfectClusterIndexes" in self.__dict__:
            return clusters[self.perfectClusterIndexes]
        else:
            self.initPerfectClusters(
                estimate,
                spread,
                minPval=minPval,
                layers = layers,
                excludeCrossTalk = excludeCrossTalk,
            )
        return clusters[self.perfectClusterIndexes]
    
    def clearData(self) -> None:
        self.dataFrameHandler.clearData()
        self.clusterHandler.clearData()
        if "crossTalk" in self.__dict__:
            del self.crossTalk
        if "perfectClusterIndexes" in self.__dict__:
            del self.perfectClusterIndexes