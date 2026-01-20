import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,printMemUsage,configLoader
import numpy as np

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)
for dataFile in dataFiles:
    crosstalk = np.sum(dataFile.get_crossTalk())
    rows = dataFile.get_base_attr("Row").size
    print(f"Percent of cross talk {crosstalk/rows * 100:.2f}%.\n({crosstalk}/{rows})")