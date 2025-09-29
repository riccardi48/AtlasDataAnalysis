import sys
sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import time
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    start = time.time()
    print(dataFile.get_clusters(recalc=True).size)
    end = time.time()
    print(f"Time taken for {dataFile.fileName}: {end - start:.2f}s")