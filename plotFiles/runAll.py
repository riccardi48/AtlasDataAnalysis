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
#config["filterDict"] = {"telescope": "kit", "angle": 86.5}
config["filterDict"] = {"telescope":"kit","fileName":["angle6_6Gev_kit_4"]}
#config["filterDict"] = {"telescope": "kit", "angle": 45}
#config["filterDict"] = {"angle":86.5,"voltage":48.6,"telescope":"lancs"}
#configLoader.saveConfig(config,"config.json")

config = configLoader.loadConfig("config.json")
config["filterDict"] = {
    "fileName": [
        "angle6_6Gev_kit_4",
        "angle6_6Gev_kitHV30_kit_5",
        "angle6_6Gev_kitHV20_kit_6",
        "angle6_6Gev_kitHV15_kit_7",
        "angle6_6Gev_kitHV10_kit_8",
        "angle6_6Gev_kitHV8_kit_9",
        "angle6_6Gev_kitHV6_kit_10",
        "angle6_6Gev_kitHV4_kit_12",
        "angle6_6Gev_kitHV2_kit_13",
        "angle6_6Gev_kitHV0_kit_14",
    ]
}

dataFiles = initDataFiles(config)
plotGen = plotGenerator(config["pathToOutput"])


runSimple(dataFiles,plotGen,config)
runCorrelation(dataFiles,plotGen,config)
runTemplate(dataFiles,plotGen,config)
runCharge(dataFiles,plotGen,config)
runEfficiency(dataFiles,plotGen,config)
runMPV(dataFiles,plotGen,config)
