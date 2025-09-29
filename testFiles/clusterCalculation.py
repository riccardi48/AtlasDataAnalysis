import sys
sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader

config = configLoader.loadConfig()
config["maxLine"] = 100000
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    print(dataFile.get_clusters(recalc=True)).size