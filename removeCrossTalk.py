from dataAnalysis import initDataFiles,configLoader

if __name__ == "__main__":
    configLoader.saveConfig({"filterDict":{"telescope":"kit","fileName":"angle6_6Gev_kitHV6_kit_10"}},path="config.json")

config = configLoader.loadConfig(path="config.json")
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    dataFile.save_nonCrossTalk_to_csv(config["pathToDataOutput"])
