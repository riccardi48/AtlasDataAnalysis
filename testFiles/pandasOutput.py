import sys

sys.path.append("..")

from dataAnalysis import initDataFiles,configLoader
import numpy as np
import pandas as pd

config = configLoader.loadConfig()
config["filterDict"] = {"fileName":"angle6_4Gev_kit_2"}
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
df["Charge"] = np.zeros(len(df),dtype=float)
for cluster in dataFiles[0].get_clusters(excludeCrossTalk=True)[:200]:
    df.loc[cluster.getIndexes(),"Charge"] = cluster.getHit_Voltages()
df["Charge_E"] = np.zeros(len(df),dtype=float)
for cluster in dataFiles[0].get_clusters(excludeCrossTalk=True)[:200]:
    df.loc[cluster.getIndexes(),"Charge_E"] = cluster.getHit_VoltageErrors()
df["Charge_E_Relative"] = df["Charge_E"]/df["Charge"]
df["ToT"] = np.zeros(len(df),dtype=bool)
for cluster in dataFiles[0].get_clusters(excludeCrossTalk=True)[:200]:
    df.loc[cluster.getIndexes(),"ToT"] = cluster.getToTs()

print(df[:100][~df["crosstalk"][:100]])
