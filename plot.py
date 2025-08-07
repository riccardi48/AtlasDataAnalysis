from dataAnalysis import initDataFiles,printMemUsage
import dataAnalysis.configLoader as configLoader
from dataAnalysis.plotter import plotterClass
import time
printMemUsage()
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    start = time.time()
    pathToOutput = f"{config["pathToOutput"]}{dataFile.fileName}/"
    plotter = plotterClass(dataFile,config,pathToOutput,saveToSharedPdf=True)
    plotter.crossTalk().CuttingComparison()
    plotter.crossTalk().CuttingComparison({"layers":[3]})
    plotter.crossTalk().CuttingComparison({"layers":[2]})
    plotter.crossTalk().CuttingComparison({"layers":[1]})
    plotter.savePDF()
    end = time.time()
    print(f"Time taken for {dataFile.fileName}: {end - start:.2f}")
    printMemUsage()
