from ._types import clusterClass, dataAnalysis, clusterArray
from .dataAnalysis import dataAnalysis
from ._dependencies import (
    np,  # numpy
)
from glob import glob


def initDataFiles(config: dict = {}) -> list[dataAnalysis]:
    if config == {}:
        import configLoader

        config = configLoader.loadConfig()
    files = glob(f"{config["pathToData"]}{config["fileFormat"]}")
    allDataFiles = [
        dataAnalysis(pathToDataFile, config["pathToCalcData"], maxLine=config["maxLine"])
        for pathToDataFile in files
    ]
    dataFiles = _filterDataFiles(
        allDataFiles,
        filterDict=config["filterDict"],
    )
    return dataFiles


def _filterDataFiles(allDataFiles: list[dataAnalysis], filterDict: dict = {}) -> list[dataAnalysis]:
    dataFiles: list[dataAnalysis] = []
    for dataFile in allDataFiles:
        if dataFile.check_if_filtered(filterDict):
            dataFiles.append(dataFile)
    sortIndex = np.flip(
        np.argsort([dataFile.angle * 1000 + dataFile.voltage for dataFile in dataFiles])
    )
    return list(np.array(dataFiles)[sortIndex])
