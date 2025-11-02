import sys
sys.path.append("..")
from landau import landau
from plotAnalysis import plotClass
import numpy as np
from dataAnalysis import configLoader

config = configLoader.loadConfig()
plot = plotClass(config["pathToOutput"] + "landau/")
axs = plot.axs
x = np.linspace(0, 30, 1000)
y = landau.cdf(x, 13, 1)
axs.plot(x, y, color=plot.colorPalette[1], label="Landau CDF")
y = landau.pdf(x, 13, 1)
axs.plot(x, y, color=plot.colorPalette[1], label="Landau CDF")
hlines = [0.05,0.5,0.95]
axs.hlines(hlines, x[0], x[-1], colors=plot.textColor, linestyles="dashed")
plot.set_config(
    axs,
    title="Landau CDF",
    xlabel="x",
    ylabel="CDF",
    legend=True,
)
plot.saveToPDF("landauCDF")