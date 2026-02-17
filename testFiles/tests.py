import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,printMemUsage,configLoader
import time
printMemUsage()
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    start = time.time()
    dataFile.get_clusters(excludeCrossTalk=True,layer=4)
    end = time.time()
    print(f"Time taken for {dataFile.fileName}: {end - start:.2f}s")
    printMemUsage()
