import sys
sys.path.append("..")
from landau import landau
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator

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

def getPixelResponseSlope(rows,chargeCollected,returnAll=False):
    rows = rows[chargeCollected>0.16]
    chargeCollected = chargeCollected[chargeCollected>0.16]
    if len(rows) <= 1:
        return np.nan
    relativeRows = rows-np.min(rows)
    popt,pcov = curve_fit(linearLine,relativeRows,chargeCollected)
    if returnAll:
        return popt,pcov
    return popt[0]

def randomCluster(clusterLength,x_mpv,xi):
    cluster = landau.sample(x_mpv, xi, clusterLength)
    rows = np.arange(clusterLength)
    return rows,cluster


def randomClusterCCE(clusterLength,x_mpv,xi,cce):
    cluster = np.zeros(clusterLength)
    for i,_cce in enumerate(cce):
        cluster[i] = landau.sample(x_mpv*_cce, xi*_cce, 1)
    rows = np.arange(clusterLength)
    rows = rows[cluster>0.16]
    cluster = cluster[cluster>0.16]
    return rows,cluster

def randomClusterCCEMany(clusterLength,x_mpv,xi,cce,n):
    cluster = np.zeros((n,clusterLength))
    for i,_cce in enumerate(cce):
        cluster[:,i] = landau.sample(x_mpv*_cce, xi*_cce, n)
    rows = np.arange(clusterLength)
    rows = np.repeat([rows],n,axis=0)
    return rows,cluster


def chargeCollectionEfficiencyFunc(
    depth,
    V_0,
    t_epi,
    edl,
    base= 0,
    GeV = None,
):
    if GeV is not None:
        if GeV == 4:
            base = 0.00
        elif GeV == 6:
            base = 0.00
    depth = np.reshape(depth, np.size(depth))
    voltage = np.zeros(depth.shape)
    voltage[depth < t_epi] = V_0
    voltage[depth >= t_epi] = np.exp(-(depth[depth >= t_epi] - t_epi) / edl) * (V_0 - base) + base
    return voltage


clusterLength = 18
mpv = 0.4
width = mpv/4
_range = (-0.2,0.2)
randomResponse = getPixelResponseSlope(*(randomCluster(clusterLength,mpv,width)))
print(randomResponse)
plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
rows,chargeCollected = randomCluster(clusterLength,mpv,width)
popt,pcov = getPixelResponseSlope(rows,chargeCollected,returnAll=True)
axs.scatter(rows,chargeCollected,color=plot.colorPalette[0],label="Generated Cluster")
axs.plot(rows,linearLine(rows,*popt),color=plot.textColor,linestyle="dashed",label=f"Fit m={popt[0]:.3} c={popt[1]:.3}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=(None,None),
    title=f"Pixel Response of one Generated Cluster with Flat MPVs",
    xlabel="Relative Row [px]",
    ylabel="Voltage [V]",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(5))
axs.xaxis.set_major_formatter("{x:.0f}")
axs.xaxis.set_minor_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(0.2))
plot.saveToPDF(f"PixelResponseOneCluster")

cce = np.full(clusterLength,1)

plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
for i,clusterLength in enumerate([4,8,12,18,25]):
    cce = np.full(clusterLength,1)
    rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce,20000)
    PRS = np.array([getPixelResponseSlope(rows[i],chargeCollected[i]) for i in range(20000)])
    height, x = np.histogram(PRS, bins=210, range=_range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[i],label=f"Length {clusterLength}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=_range,
    title=f"Pixel Response Slope Histogram Generated with Slope CCE",
    xlabel="Pixel Response Slope [V/px]",
    ylabel="Count",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(0.05))
axs.xaxis.set_major_formatter("{x:.2f}")
axs.xaxis.set_minor_locator(MultipleLocator(0.01))
axs.yaxis.set_major_locator(MultipleLocator(100))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(20))
plot.saveToPDF(f"PixelResponseSlope_Generated_Mixed")

cce = np.linspace(1,0.1,clusterLength)
plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
axs.plot(rows[0],cce,color=plot.colorPalette[0],label="Generated CCE")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=(None,None),
    title=f"Simple Charge Collection Efficiency",
    xlabel="Relative Row [px]",
    ylabel="Charge Collection Efficiency",
    )
axs.xaxis.set_major_locator(MultipleLocator(2))
axs.xaxis.set_major_formatter("{x:.0f}")
axs.xaxis.set_minor_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(0.2))
axs.yaxis.set_major_formatter("{x:.1f}")
axs.yaxis.set_minor_locator(MultipleLocator(0.1))
plot.saveToPDF(f"SlopeCCE")

plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce,1)
popt,pcov = getPixelResponseSlope(rows[0],chargeCollected[0],returnAll=True)
axs.scatter(rows[0],chargeCollected[0],color=plot.colorPalette[0],label="Generated Cluster")
axs.plot(rows[0],linearLine(rows[0],*popt),color=plot.textColor,linestyle="dashed",label=f"Fit m={popt[0]:.3} c={popt[1]:.3}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=(None,None),
    title=f"Pixel Response of one Generated Cluster with Flat MPVs",
    xlabel="Relative Row [px]",
    ylabel="Voltage [V]",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(5))
axs.xaxis.set_major_formatter("{x:.0f}")
axs.xaxis.set_minor_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(0.2))
plot.saveToPDF(f"PixelResponseOneCluster_CCE")



plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce,100000)
PRS = np.array([getPixelResponseSlope(rows[i],chargeCollected[i]) for i in range(100000)])
height, x = np.histogram(PRS, bins=210, range=_range)
axs.stairs(height, x, baseline=None, color=plot.colorPalette[0],label=f"Length {clusterLength}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=_range,
    title=f"Pixel Response Slope Histogram Generated with Slope CCE",
    xlabel="Pixel Response Slope [V/px]",
    ylabel="Count",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(0.05))
axs.xaxis.set_major_formatter("{x:.2f}")
axs.xaxis.set_minor_locator(MultipleLocator(0.01))
axs.yaxis.set_major_locator(MultipleLocator(1000))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(200))
plot.saveToPDF(f"PixelResponseSlope_Generated__SlopeCCE_{clusterLength}")

plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
cce1 = np.linspace(1,0.1,clusterLength)
rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce1,50000)
PRS1 = np.array([getPixelResponseSlope(rows[i],chargeCollected[i]) for i in range(50000)])
cce2 = np.linspace(0.1,1,clusterLength)
rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce2,50000)
PRS2 = np.array([getPixelResponseSlope(rows[i],chargeCollected[i]) for i in range(50000)])
PRS = np.concatenate((PRS1,PRS2))
height, x = np.histogram(PRS, bins=210, range=_range)
axs.stairs(height, x, baseline=None, color=plot.colorPalette[0],label=f"Length {clusterLength}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=_range,
    title=f"Pixel Response Slope Histogram Generated with Slope CCE",
    xlabel="Pixel Response Slope [V/px]",
    ylabel="Count",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(0.05))
axs.xaxis.set_major_formatter("{x:.2f}")
axs.xaxis.set_minor_locator(MultipleLocator(0.01))
axs.yaxis.set_major_locator(MultipleLocator(1000))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(200))
plot.saveToPDF(f"PixelResponseSlope_Generated__TwowayCCE_18")

clusterLength = 30

cce = chargeCollectionEfficiencyFunc(
    np.linspace(0,clusterLength,clusterLength),
    1,
    8,
    6,
)
print(cce)
plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
axs.plot(np.arange(clusterLength),cce,color=plot.colorPalette[0],label="Generated CCE")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=(None,None),
    title=f"Simple Charge Collection Efficiency",
    xlabel="Relative Row [px]",
    ylabel="Charge Collection Efficiency",
    )
axs.xaxis.set_major_locator(MultipleLocator(2))
axs.xaxis.set_major_formatter("{x:.0f}")
axs.xaxis.set_minor_locator(MultipleLocator(1))
axs.yaxis.set_major_locator(MultipleLocator(0.2))
axs.yaxis.set_major_formatter("{x:.1f}")
axs.yaxis.set_minor_locator(MultipleLocator(0.1))
plot.saveToPDF(f"ComplexCCE")

plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce,100000)
PRS = np.array([getPixelResponseSlope(rows[i],chargeCollected[i]) for i in range(100000)])
height, x = np.histogram(PRS, bins=210, range=_range)
axs.stairs(height, x, baseline=None, color=plot.colorPalette[0],label=f"Length {clusterLength}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=_range,
    title=f"Pixel Response Slope Histogram Generated with Slope CCE",
    xlabel="Pixel Response Slope [V/px]",
    ylabel="Count",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(0.05))
axs.xaxis.set_major_formatter("{x:.2f}")
axs.xaxis.set_minor_locator(MultipleLocator(0.01))
axs.yaxis.set_major_locator(MultipleLocator(1000))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(200))
plot.saveToPDF(f"PixelResponseSlope_Generated__ComplexCCE_18")

plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
cce1 = chargeCollectionEfficiencyFunc(
    np.linspace(0,clusterLength,clusterLength),
    1,
    8,
    6,
)
rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce1,50000)
PRS1 = np.array([getPixelResponseSlope(rows[i],chargeCollected[i]) for i in range(50000)])
cce2 = chargeCollectionEfficiencyFunc(
    np.linspace(clusterLength,0,clusterLength),
    1,
    8,
    6,
)
rows,chargeCollected = randomClusterCCEMany(clusterLength,mpv,width,cce2,50000)
PRS2 = np.array([getPixelResponseSlope(rows[i],chargeCollected[i]) for i in range(50000)])
PRS = np.concatenate((PRS1,PRS2))
height, x = np.histogram(PRS, bins=210, range=_range)
axs.stairs(height, x, baseline=None, color=plot.colorPalette[0],label=f"Length {clusterLength}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=_range,
    title=f"Pixel Response Slope Histogram Generated with Slope CCE",
    xlabel="Pixel Response Slope [V/px]",
    ylabel="Count",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(0.05))
axs.xaxis.set_major_formatter("{x:.2f}")
axs.xaxis.set_minor_locator(MultipleLocator(0.01))
axs.yaxis.set_major_locator(MultipleLocator(1000))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(200))
plot.saveToPDF(f"PixelResponseSlope_Generated__Complex_TwowayCCE_18")


clusterLengths = [4,6,8,10,14,18,22,26]
clusterLengths  = [18]
plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/PixelResponseSlope/Generated/")
axs = plot.axs
for i,clusterLength in enumerate(clusterLengths):
    PRS = np.array([getPixelResponseSlope(*(randomCluster(clusterLength,mpv,width))) for i in range(100000)])
    height, x = np.histogram(PRS, bins=210, range=_range)
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[i],label=f"Length {clusterLength}")
plot.set_config(
    axs,
    ylim=(0, None),
    xlim=_range,
    title=f"Pixel Response Slope Histogram Generated",
    xlabel="Pixel Response Slope [V/px]",
    ylabel="Count",
    legend=True,
    )
axs.xaxis.set_major_locator(MultipleLocator(0.05))
axs.xaxis.set_major_formatter("{x:.2f}")
axs.xaxis.set_minor_locator(MultipleLocator(0.01))
axs.yaxis.set_major_locator(MultipleLocator(1000))
axs.yaxis.set_major_formatter("{x:.0f}")
axs.yaxis.set_minor_locator(MultipleLocator(200))
plot.saveToPDF(f"PixelResponseSlope_Generated_18")
