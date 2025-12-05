import sys
from funcs import scaleTemplate,getTemplate
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass


config = configLoader.loadConfig()
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

estimate,spread = getTemplate(dataFiles[0],config)

plot = plotClass(f"/home/atlas/rballard/AtlasDataAnalysis/perfectCluster/")
axs = plot.axs
axs.scatter(np.arange(len(estimate)),estimate,marker="x",color=plot.colorPalette[0],label="Original")
axs.errorbar(
        np.arange(len(estimate)),
        estimate,
        yerr=spread,
        fmt="none",
        color=plot.colorPalette[0],
        elinewidth=1,
        capsize=3,
    )

angleScaler,flatScaler = (1,1)
newEstimate,newSpread = scaleTemplate(estimate,spread,angleScaler,flatScaler)
axs.scatter(np.arange(len(newEstimate)),newEstimate,marker="x",color=plot.colorPalette[2],label=f"{angleScaler},{flatScaler}")
axs.errorbar(
        np.arange(len(newEstimate)),
        newEstimate,
        yerr=newSpread,
        fmt="none",
        color=plot.colorPalette[2],
        elinewidth=1,
        capsize=3,
    )
angleScaler,flatScaler = (0.5,1)
newEstimate,newSpread = scaleTemplate(estimate,spread,angleScaler,flatScaler)
axs.scatter(np.arange(len(newEstimate)),newEstimate,marker="x",color=plot.colorPalette[3],label=f"{angleScaler},{flatScaler}")
axs.errorbar(
        np.arange(len(newEstimate)),
        newEstimate,
        yerr=newSpread,
        fmt="none",
        color=plot.colorPalette[3],
        elinewidth=1,
        capsize=3,
    )
angleScaler,flatScaler = (1,0.5)
newEstimate,newSpread = scaleTemplate(estimate,spread,angleScaler,flatScaler)
axs.scatter(np.arange(len(newEstimate)),newEstimate,marker="x",color=plot.colorPalette[4],label=f"{angleScaler},{flatScaler}")
axs.errorbar(
        np.arange(len(newEstimate)),
        newEstimate,
        yerr=newSpread,
        fmt="none",
        color=plot.colorPalette[4],
        elinewidth=1,
        capsize=3,
    )
angleScaler,flatScaler = (0.5,0.5)
newEstimate,newSpread = scaleTemplate(estimate,spread,angleScaler,flatScaler)
axs.scatter(np.arange(len(newEstimate)),newEstimate,marker="x",color=plot.colorPalette[5],label=f"{angleScaler},{flatScaler}")
axs.errorbar(
        np.arange(len(newEstimate)),
        newEstimate,
        yerr=newSpread,
        fmt="none",
        color=plot.colorPalette[5],
        elinewidth=1,
        capsize=3,
    )
plot.set_config(axs,
        title="Estimate Scaler Test",
        xlabel="Row",
        ylabel="Expected Relative TS",
        legend = True,
    )
plot.saveToPDF("Estimate_Scaler_Test")
