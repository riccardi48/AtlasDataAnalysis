from dataAnalysis import initDataFiles,printMemUsage
import dataAnalysis.configLoader as configLoader
import time
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    start = time.time()
    dataFile.get_clusters(excludeCrossTalk=True, recalc = False)
    end = time.time()
    print(f"Time taken for {dataFile.fileName}: {end - start:.2f}")
    printMemUsage()
