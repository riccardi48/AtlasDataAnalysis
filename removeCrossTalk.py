from dataAnalysis import dataAnalysis,initDataFiles
import configLoader
import glob

configLoader.saveConfig({"filterDict":{"telescope":"kit","fileName":"angle6_4Gev_kit_2"}},path="config.json")
config = configLoader.loadConfig(path="config.json")

dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    dataFile.save_nonCrossTalk_to_csv(config["pathToDataOutput"])
