import sys

sys.path.append("..")

from dataAnalysis import initDataFiles,configLoader
import numpy as np
import pandas as pd

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 1000)  # or 1000
pd.set_option('display.max_colwidth', 1000)  # or 199
pd.set_option('display.width', 1000)
df = dataFiles[0].get_dataFrame()
df["cluster"] = np.zeros(len(df),dtype=int)
for cluster in dataFiles[0].get_clusters(excludeCrossTalk=True)[:200]:
    df.loc[cluster.getIndexes(),"cluster"] = cluster.getIndex()
df["crosstalk"] = np.zeros(len(df),dtype=bool)
for cluster in dataFiles[0].get_clusters(excludeCrossTalk=True)[:200]:
    df.loc[cluster.getIndexes(),"crosstalk"] = cluster.crossTalk
dataFiles[0].init_cluster_voltages()
df["Charge_Collected"] = np.zeros(len(df),dtype=bool)
for cluster in dataFiles[0].get_clusters(excludeCrossTalk=True)[:200]:
    df.loc[cluster.getIndexes(),"Charge_Collected"] = cluster.getHit_Voltages()
print(df[:100][~df["crosstalk"][:100]])