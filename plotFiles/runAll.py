import sys
from plotClass import plotGenerator
sys.path.append("..")
from dataAnalysis import configLoader,initDataFiles

from simplePlots import runSimple
from templatePlots import runTemplate
from clusterChargePlots import runCharge
from efficiencyPlots import runEfficiency
from mpvPlots import runMPV
from correlationPlots import runCorrelation

config = configLoader.loadConfig()
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
# config["filterDict"] = {"telescope":"kit","fileName":["angle6_4Gev_kit_2","angle6_6Gev_kit_4"]}
configLoader.saveConfig(config,"config.json")
config = configLoader.loadConfig("config.json")
dataFiles = initDataFiles(config)
plotGen = plotGenerator(config["pathToOutput"])


runSimple(dataFiles,plotGen,config)
runTemplate(dataFiles,plotGen,config)
runCharge(dataFiles,plotGen,config)
runEfficiency(dataFiles,plotGen,config)
runMPV(dataFiles,plotGen,config)
runCorrelation(dataFiles,plotGen,config)