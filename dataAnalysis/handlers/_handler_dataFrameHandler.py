from typing import Optional, Any
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
)
from dataAnalysis._fileReader import rawDataFileManager
from dataAnalysis._memCheck import usage
from ._functions import calcToT, trueTimeStamps
from ._hit_voltage import calcHit_Voltage, calcHit_VoltageError

class dataFrameHandler:
    # Stores the data frame
    # Retrieves calculated data for values on a per hit basis i.e. one value per line in data frame e.g. ToT, Voltage
    def __init__(
        self,
        pathToData: str,
        columnNames: list[str],
        maxLine: Optional[int] = None,
        telescope: str = "kit",
    ):
        self.dataFileManager = rawDataFileManager(pathToData, columnNames, maxLine=maxLine)
        self.telescope = telescope

    def loadDataIfNotLoaded(self) -> None:
        if not self.checkLoadedData():
            self.data = self.dataFileManager.readFile()
            self.dataLength: int = len(self.data)

    def dropDataIfRamUsageHigh(self,ramLimit:int = 10000) -> bool:
        if usage() > ramLimit:
            del self.data
            return True
        return False
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
        hit_voltage, hit_voltageError = calcHit_VoltageError(rows, columns, ToTs, Layers)
        return hit_voltage

    def getHit_VoltageError(
        self, ToTs: Optional[npt.NDArray[np.int_]] = None
    ) -> npt.NDArray[np.float64]:
        self.loadDataIfNotLoaded()
        if ToTs is None:
            ToTs = self.getToT()
        rows = self.data["Row"].to_numpy()
        columns = self.data["Column"].to_numpy()
        Layers = self.data["Layer"].to_numpy()
        hit_voltage, hit_voltageError = calcHit_VoltageError(rows, columns, ToTs, Layers)
        return hit_voltageError

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
        print("Fixing ext_TS for non-CrossTalk hits")
        outputDF["ext_TS"] = trueTimeStamps(clusters, outputDF["ext_TS"].to_numpy())
        positions = np.where(crosstalk)[0]
        index_labels = self.data.index[positions]
        print("Removing CrossTalk hits from DataFrame")
        outputDF = outputDF.drop(index=index_labels)
        print("Saving non-CrossTalk hits to CSV")
        self.dataFileManager.saveFile(outputDF, path, name)

    def clearData(self) -> None:
        if self.checkLoadedData():
            del self.data