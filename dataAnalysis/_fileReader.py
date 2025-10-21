from ._dependencies import (
    pd,  # pandas
    np,  # numpy
    npt,  # numpy.typing
)
from typing import Optional, Any
import os


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
            "PackageID": np.int32,
            "Layer": np.int16,
            "Column": np.int16,
            "Row": np.int16,
            "TS": np.int16,
            "TS1": np.int16,
            "TS2": np.int16,
            "TriggerTS": np.int64,
            "TriggerID": np.int64,
            "ext_TS": np.int64,
            "ext_TS2": np.int64,
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
