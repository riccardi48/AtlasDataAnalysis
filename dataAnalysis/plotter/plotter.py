from dataAnalysis._types import dataAnalysis
from typing import Optional, Any, Union
from dataAnalysis.configLoader import addMissingKeys
from dataAnalysis._dependencies import (
    np,
    npt,
    PdfPages,
)
from .modules import crossTalkModule,simpleModule

class plotterClass:
    def __init__(self, dataFile: dataAnalysis, config: dict, pathToOutput, saveToSharedPdf: bool = False):
        self.dataFile = dataFile
        self.config = config
        self.pathToOutput = pathToOutput
        self.saveToSharedPdf = saveToSharedPdf
        if self.saveToSharedPdf:
            self.pdfName = "grouped"
            self.pdf = PdfPages(f"{config["pathToOutput"]}{self.dataFile.fileName}/{self.pdfName}.pdf")

    def configCheck(self, config: Union[dict | None]) -> dict:
        if config is not None:
            config = addMissingKeys(config, self.config)
        else:
            config = self.config
        return config

    def crossTalk(self):
        if "_crossTalk" not in self.__dict__:
            self._crossTalk = crossTalkModule(self.dataFile, self.config, f"{self.pathToOutput}CrossTalk/", saveToSharedPdf = self.saveToSharedPdf)
            if self.saveToSharedPdf:
                self._crossTalk.pdf = self.pdf
        return self._crossTalk
    
    def simple(self):
        if "_simple" not in self.__dict__:
            self._simple = simpleModule(self.dataFile, self.config, f"{self.pathToOutput}Simple/", saveToSharedPdf = self.saveToSharedPdf)
            if self.saveToSharedPdf:
                self._simple.pdf = self.pdf
        return self._simple
    
    def savePDF(self):
        self.pdf.savefig()
        self.pdf.close()
        print(f"Saved Plot: {self.pdfName}.pdf")


