import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,printMemUsage,configLoader
import time
printMemUsage()
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    start = time.time()
    print([cluster.getColumnWidth(True) for cluster in dataFile.get_flatClusters(20,excludeCrossTalk=True,layer=4)])
    print([cluster.getRowWidth(True) for cluster in dataFile.get_flatClusters(20,excludeCrossTalk=True,layer=4)])
    end = time.time()
    print(f"Time taken for {dataFile.fileName}: {end - start:.2f}s")
    printMemUsage()
