from dataAnalysis import initDataFiles,configLoader,printMemUsage

if __name__ == "__main__":
    configLoader.saveConfig({"filterDict":{"telescope":"kit","fileName":"long_term_6Gev_kit_01"}},path="config.json")

config = configLoader.loadConfig(path="config.json")
dataFiles = initDataFiles(config)
printMemUsage()

for dataFile in dataFiles:
    dataFile.save_nonCrossTalk_to_csv(config["pathToDataOutput"])
    printMemUsage()
