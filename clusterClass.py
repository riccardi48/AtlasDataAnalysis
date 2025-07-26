import numpy as np
import numpy.typing as npt

class clusterClass:
    def __init__(self, index: int, indexes: list[int], layer: int, columns: list[int], rows: list[int], ToTs: list[int], TSs: list[int]):
        self.indexes = indexes
        self.layer = layer
        self.columns = columns
        self.rows = rows
        self.ToTs = ToTs
        self.TSs = TSs
        self.index = index

    def setHit_Voltage(self, Hit_Voltages: list[float], Hit_VoltageErrors: list[float]) -> None:
        self.Hit_Voltages = Hit_Voltages
        self.Hit_VoltageErrors = Hit_VoltageErrors

    def setCrossTalk(self, crossTalk: list[bool]) -> None:
        self.crossTalk = crossTalk
        self.notCrossTalk = np.invert(crossTalk)

    def getIndex(self, excludeCrossTalk: bool = False) -> int:
        return self.index

    def getIndexes(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.int_]:
        if excludeCrossTalk:
            return self.indexes[self.notCrossTalk]
        else:
            return self.indexes

    def getLayer(self, excludeCrossTalk: bool = False) -> int:
        return self.layer

    def getShortIndexes(self, excludeCrossTalk: bool = False) -> npt.NDArray[np.int_]:
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

    def getColumns(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.int_]:
        if excludeCrossTalk:
            return self.columns[self.notCrossTalk]
        else:
            return self.columns

    def getRows(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.int_]:
        if excludeCrossTalk:
            return self.rows[self.notCrossTalk]
        else:
            return self.rows

    def getHit_Voltages(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.float64]:
        if excludeCrossTalk:
            return self.Hit_Voltages[self.notCrossTalk]
        else:
            return self.Hit_Voltages

    def getHit_VoltageErrors(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.float64]:
        if excludeCrossTalk:
            return self.Hit_VoltageErrors[self.notCrossTalk]
        else:
            return self.Hit_VoltageErrors

    def getToTs(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.int_]:
        if excludeCrossTalk:
            return self.ToTs[self.notCrossTalk]
        else:
            return self.ToTs

    def getToTErrors(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.int_]:
        if excludeCrossTalk:
            return np.full(self.ToTs.shape, 2)[self.notCrossTalk]
        else:
            return np.full(self.ToTs.shape, 2)

    def getTSs(self, excludeCrossTalk: bool = False) -> npt.ArrayLike[np.int_]:
        if excludeCrossTalk:
            return self.TSs[self.notCrossTalk]
        else:
            return self.TSs