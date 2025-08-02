from typing import Optional, Any
from ._types import clusterClass, dataAnalysis, clusterArray
from .handlers import dataHandler, readFileName
from ._memCheck import printMemUsage
from ._dependencies import (
    pd,  # pandas
    np,  # numpy
    npt,  # numpy.typing
)


def _isFiltered(dataFile: dataAnalysis, filter_dict: dict = {}) -> bool:
    # Takes in an array of data_class and filters and sorts them
    # filter_dict has keys that are attributes of data_class with values that you want to filter for
    # Returns filtered list sorted by angle and then by voltage
    for f in filter_dict.keys():
        attr = getattr(dataFile, f"{f}")
        if isinstance(f, list):
            if not np.isin(attr, filter_dict[f]):
                return False
        else:
            if not attr == filter_dict[f]:
                return False
    return True


class dataAnalysis:
    def __init__(self, pathToData: str, pathToCalcData: str, maxLine: Optional[int] = None) -> None:
        self.fileName, self.voltage, self.angle, self.telescope = readFileName(pathToData)
        self.dataHandler = dataHandler(pathToData, pathToCalcData, maxLine=maxLine)
        self.pathToData = pathToData

    def check_if_filtered(self, filterDict) -> bool:
        return _isFiltered(self, filter_dict=filterDict)

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
        self.dataHandler.save_nonCrossTalk_to_csv(path, self.fileName)

    def get_flatClusters(self,width,**kwargs) -> clusterArray:
        if "layer" in kwargs:
            if kwargs["layer"] is None:
                kwargs["layers"] = None
                kwargs.pop("layer")
            else:
                kwargs["layers"] = [kwargs["layer"]]
                kwargs.pop("layer")
        return self.dataHandler.getFlatClusters(width,**kwargs)