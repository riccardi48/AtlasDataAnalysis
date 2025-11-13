from funcs import characterizeCluster, typeDict, isTypes
import sys

sys.path.append("..")
from dataAnalysis import initDataFiles, configLoader
import numpy as np
from plotAnalysis import plotClass


def percentTypes(string, clusterTypes):
    return np.sum(isTypes(string, clusterTypes)) / len(clusterTypes) * 100


config = configLoader.loadConfig()
# config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
# config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)
bigPlot = plotClass(f"{config["pathToOutput"]}ClusterTracks/Collected/", sizePerPlot=(24, 6))
for k, dataFile in enumerate(dataFiles[:8]):
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/ClusterTypes_/"
    dataFile.init_cluster_voltages()
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    clusters = clusters[:30000]
    for cluster in clusters:
        characterizeCluster(cluster)
        print(f"Cluster Index:{cluster.getIndex()}", end="\r")
    print(f"")
    clusterTypes = np.array([cluster.clusterType for cluster in clusters])
    plot = plotClass(base_path, sizePerPlot=(24, 6))
    axs = plot.axs
    for key in typeDict.keys():
        axs.bar(typeDict[key], percentTypes(str(key), clusterTypes), 0.7)
    axs.bar("Perfect\nCluster (Flat)", percentTypes("239~4", clusterTypes), 0.7)
    axs.bar("Perfect\nCluster", percentTypes("39~4", clusterTypes), 0.7)
    # axs.bar(
    #    "Perfect Cluster\n\\w Gaps",
    #    percentTypes("7", clusterTypes[isTypes("39~4", clusterTypes)]),
    #    0.7,
    # )
    # axs.bar("Long Clusters", percentTypes("3b~4", clusterTypes), 0.7)
    # axs.bar("Long Clusters\n\\w Gaps", percentTypes("3b~47", clusterTypes), 0.7)
    xlim = axs.get_xlim()
    axs.hlines(
        100 - percentTypes("1", clusterTypes),
        xlim[0],
        xlim[1],
        color=plot.textColor,
        linestyle="dashed",
    )
    plot.set_config(
        axs,
        title="Percent of Cluster Types",
        xlabel="Cluster Type",
        ylabel="Percent of Clusters",
        xlim=xlim,
    )
    plot.saveToPDF(f"Cluster_Types_")

    variableArray = np.zeros((len(typeDict.keys()), len(clusters)), dtype=bool)
    for i, key in enumerate(typeDict.keys()):
        variableArray[i] = isTypes(key, clusterTypes)
    cov = np.corrcoef(variableArray.astype(int))
    print(cov)
    keys = list(typeDict.keys())
    for i, j in np.rot90(np.where((cov > 0.3) | (cov < -0.3))):
        if i > j:
            print(
                f"{typeDict[keys[i]].replace("\n","")}, {typeDict[keys[j]].replace("\n","")} = {cov[i,j]:.2f}"
            )
    labels = list(typeDict.values()) + ["Perfect\nCluster"] + ["Perfect\nCluster (Flat)"]
    sizes = (
        [percentTypes(str(key), clusterTypes) for key in typeDict.keys()]
        + [percentTypes("39~4", clusterTypes)]
        + [percentTypes("239~4", clusterTypes)]
    )
    x = np.arange(len(labels))
    width = 0.1
    offset = width * k
    bigPlot.axs.bar(
        x + offset, sizes, width, label=dataFile.fileName, color=bigPlot.colorPalette[k]
    )

bigPlot.set_config(
    bigPlot.axs,
    title="Percent of Cluster Types",
    xlabel="Cluster Type",
    ylabel="Percent of Clusters",
    legend=True,
)
bigPlot.axs.set_xticks(x + width, labels)
bigPlot.saveToPDF(f"Cluster_Types")
