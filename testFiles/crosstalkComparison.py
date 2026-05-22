import sys

sys.path.append("..")
from plotFiles.plotClass import plotGenerator
from dataAnalysis.handlers._genericClusterFuncs import isFlat
from dataAnalysis import initDataFiles, configLoader, printMemUsage
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from landau import landau
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

def landauFunc(
    x,
    x_mpv,
    xi,
    scaler,
    ):
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    return y

config = configLoader.loadConfig()
config["filterDict"] = {"telescope":"kit"}
#config["maxLine"] = 1000000
dataFiles = initDataFiles(config,printNames=True)
plotGen = plotGenerator(config["pathToOutput"])
attribute = "Angle"
layer = 4
path = f"Combined/"
plot = plotGen.newPlot(path,sizePerPlot=(7,5),rect=(0.05,0.04,0.995,0.96))
axs = plot.axs
attributeDict = {"Bias Voltage": 48.6, "Angle": 86.5}
cmap = plt.get_cmap("plasma")
for dataFile in []:#dataFiles:
    if attribute == "Bias Voltage":
        x = dataFile.voltage
        check = dataFile.angle == attributeDict["Angle"]
        label = f"{x}V"
    elif attribute == "Angle":
        x = dataFile.angle
        check = dataFile.voltage == attributeDict["Bias Voltage"]
        label = f"{x} Degrees"
    if check:
        crossTalkPercent = (
            np.sum(dataFile.get_crossTalk(layer=layer, initClusters=False))
            / dataFile.get_crossTalk(layer=layer, initClusters=False).size
        )
        axs.scatter(
            x,
            crossTalkPercent,
            label=dataFile.fileName.replace("_"," "),
            marker="x",
        )
    dataFile.clear_data()
    printMemUsage()
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=(-2, None),
    title=f"{attribute} vs Crosstalk Percent",
    legend=True,
    xlabel=f"{attribute}",
    ylabel=f"Crosstalk Percent of Total Hits",
    ncols=2,
    yticks=(0.05,0.01),
    yticksSig=2,
    xticks=(5,1),
)
#plot.saveToPDF(f"{attribute.replace(" ","_")}_{layer}")

plot = plotGen.newPlot(path,sizePerPlot=(7,5),rect=(0.02,0.02,0.95,0.96))
axs = plot.axs
for dataFile in dataFiles:
    ToT = dataFile.get_base_attr("ToT", layer=layer, excludeCrossTalk=True)
    binWidth = 2
    bins = np.linspace(0, np.max(ToT) + binWidth, int((np.max(ToT) + binWidth) / binWidth))
    height, x = np.histogram(ToT, bins=bins, range=(0, np.max(ToT) + binWidth))
    binCentres = (x[:-1] + x[1:]) / 2
    index = binCentres > 30
    popt, pcov = curve_fit(
            landauFunc,
            binCentres[index],
            height[index],
    )
    crossTalkPercent = (
            np.sum(dataFile.get_crossTalk(layer=layer, initClusters=False))
            / dataFile.get_crossTalk(layer=layer, initClusters=False).size
        )
    print(popt[0], crossTalkPercent)
    axs.scatter(
            popt[0],
            crossTalkPercent,
            label=dataFile.fileName.replace("_"," "),
            marker="x",
            color=cmap(((dataFile.angle+1)/(86.5+1))**0.5),
        )
    dataFile.clear_data()
    printMemUsage()

plot.set_config(
    axs,
    ylim=(0, None),
    xlim=(60, None),
    title=f"MPV vs Crosstalk Percent",
    #legend=True,
    xlabel=f"MPV of ToT Distribution [TS]",
    ylabel=f"Crosstalk Percent of Total Hits",
    ncols=2,
    yticks=(0.05,0.01),
    yticksSig=2,
    xticks=(10,2),
)


norm = plt.Normalize(np.min([dataFile.angle for dataFile in dataFiles]), np.max([dataFile.angle for dataFile in dataFiles]))
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical",label="Angle [Degrees]")
plot.saveToPDF(f"MPV_CrosstalkPercent_{layer}")
