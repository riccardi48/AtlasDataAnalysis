from AtlasDataAnalysis.Code.dataAnalysis.dataAnalysis import initDataFiles
from AtlasDataAnalysis.Code.lowLevelFunctions import calcDepthsFromTSs
import numpy as np
import AtlasDataAnalysis.Code.dataAnalysis.configLoader as configLoader
from plotAnalysis import plotClass

config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    indexes = []
    clusters = dataFile.get_clusters(layers=[4],excludeCrossTalk=True)
    for cluster in clusters:
        if cluster.getSize(True) > 4 and np.unique(cluster.getColumns(True)).size==1 and cluster.getSize(True) > cluster.getRowWidth(True)/2:
            #print(dataFile.get_dataFrame().iloc[cluster.getIndexes(True)])
            calcDepthsFromTSs(cluster)
            indexes.append(cluster.getIndex())
            #input()
    testAngles = np.linspace(0,90,900)
    angles = np.zeros(testAngles.shape)
    d=-1
    clusters = dataFile.get_clusters(excludeCrossTalk=True)
    for cluster in clusters[indexes]:
        angle = np.rad2deg(np.arctan(np.ptp(cluster.getRows(True))/np.ptp(cluster.depth)))
        angleMax = np.rad2deg(np.arctan(np.ptp(cluster.getRows(True)[cluster.depth>-1])/(np.max(cluster.depth-cluster.depthError)-np.min((cluster.depth+cluster.depthError)[cluster.depth>-1]))))
        angleMin = np.rad2deg(np.arctan(np.ptp(cluster.getRows(True)[cluster.depth>-1])/(np.max(cluster.depth+cluster.depthError)-np.min((cluster.depth-cluster.depthError)[cluster.depth>-1]))))
        angles[(testAngles>=angleMin) & (testAngles<=angleMax)] += 1/np.sum([(testAngles>=angleMin) & (testAngles<=angleMax)])
    plot = plotClass(config["pathToOutput"] + "AngleTest/")
    axs = plot.axs
    #hist, binEdges = np.histogram(angles,bins=50,range=(75,90))
    axs.stairs(
            angles,
            np.linspace(-0.5,90.5,901),
            label=f"{-d*50:.0f} μm",
            baseline=None,
            color=plot.colorPalette[3],
        )
    axs.vlines(
        dataFile.get_angle(), 0, axs.get_ylim()[1], colors=plot.textColor, linestyles="dashed"
    )
    axs.text(
        dataFile.get_angle(),
        axs.get_ylim()[1],
        dataFile.get_angle(),
        color=plot.textColor,
        fontweight="bold",
        horizontalalignment="right",
        verticalalignment="top",
    )
    plot.set_config(
        axs,
        ylim=(0, None),
        xlim=(75, 90),
        title="Angle Distribution",
        legend=True,
        xlabel="Equivalent Angle [Degrees]",
        ylabel="Frequency",
    )
    plot.saveToPDF(f"AngleDistribution_{dataFile.get_fileName()}")