import sys
sys.path.append("..")
from landau import landau
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.sampling import NumericalInversePolynomial
from scipy.integrate import quad
from plotAnalysis import plotClass

class MyDist:
    def __init__(self, mpv,width):
        self.mpv = mpv
        self.width = width

    def support(self):
        return (0, np.inf)

    def pdf(self, x):
        return landau.pdf(x, self.mpv, self.width)


def linearLine(x,m,c):
    return m*x + c

def getPixelResponseSlope(rows,chargeCollected):
    rows = rows[chargeCollected>0]
    chargeCollected = chargeCollected[chargeCollected>0]
    if len(rows) <= 1:
        return np.nan
    relativeRows = rows-np.min(rows)
    popt,pcov = curve_fit(linearLine,relativeRows,chargeCollected)
    return popt[0]

def randomCluster(clusterLength,gen):
    cluster = gen.rvs(size=clusterLength)
    rows = np.arange(clusterLength)
    return rows,cluster


clusterLength = 4
mpv = 0.4
width = mpv/4
dist = MyDist(mpv,width)
gen = NumericalInversePolynomial(dist)
randomResponse = getPixelResponseSlope(*(randomCluster(clusterLength,gen)))
print(randomResponse)
clusterLengths = [4,6,8,10,14,18,22,26]
clusterLengths  = [18]
_range = (-1,1)
plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSLope/")
axs = plot.axs
for i,clusterLength in enumerate(clusterLengths):
    PRS = np.array([getPixelResponseSlope(*(randomCluster(clusterLength,gen))) for i in range(100000)])
    height, x = np.histogram(PRS, bins=210, range=_range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[i],label=f"Length {clusterLength}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=_range,
    title=f"Pixel Response Slope Histogram Generated",
    xlabel="Pixel Response Slope",
    ylabel="Frequency",
    legend=True,
    )
plot.saveToPDF(f"PixelResponseSlope_Generated")
