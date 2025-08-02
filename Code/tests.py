from dataAnalysis import initDataFiles
import configLoader
import time
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    start = time.time()
    dataFile.get_clusters(excludeCrossTalk=True, recalc = True)
    end = time.time()
    print(f"Time taken for {dataFile.get_fileName()}: {end - start:.2f}")
