import sys
sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import time
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    start = time.time()
    print(dataFile.get_clusters(recalc=True,excludeCrossTalk = True).size)
    end = time.time()
    clusters = dataFile.get_clusters(excludeCrossTalk = True)
    print(clusters[1000].getIndexes(excludeCrossTalk = True))
    print(f"Time taken for {dataFile.fileName}: {end - start:.2f}s")