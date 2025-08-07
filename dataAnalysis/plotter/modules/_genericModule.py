from dataAnalysis._types import dataAnalysis
from typing import Optional, Any, Union
from dataAnalysis.configLoader import addMissingKeys
from dataAnalysis._dependencies import (
    np,
    npt,
    PdfPages,
)
class plotModule():
    def __init__(self, dataFile: dataAnalysis, config: dict, pathToOutput:str, saveToSharedPdf: bool = False, pdf:Optional[PdfPages]=None):
        self.dataFile = dataFile
        self.config = config
        self.pathToOutput = pathToOutput
        self.saveToSharedPdf = saveToSharedPdf
        self.pdf = pdf
    def configCheck(self, config: Union[dict | None]) -> dict:
        if config is not None:
            config = addMissingKeys(config, self.config)
        else:
            config = self.config
        return config
