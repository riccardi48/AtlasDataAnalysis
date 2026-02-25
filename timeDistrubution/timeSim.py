import sys
from concentrationFunc import electron_diffusion_length_p_type,depletion_width_scaled
sys.path.append("..")
from scipy.special import erf as _erf
import numpy as np
from plotFiles.plotClass import plotGenerator
from landau import landau
from scipy.stats import moyal
from scipy.stats.sampling import NumericalInversePolynomial
from scipy.stats import norm,lognorm,goodness_of_fit,rv_continuous
import numpy as np
from scipy.optimize import root_scalar,curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def CCEFunc(x,t,N,w,L_n,tao):
    return (
        (N/2)
        * (  
            2*np.cosh(f(x,L_n,w))
            - np.exp(f(x,L_n,w)) * g(x,L_n,w,tao,t,1)
            - np.exp(-f(x,L_n,w)) * g(x,L_n,w,tao,t,-1)
            )
        )
def erf(x):
    return _erf(x)

def f(x,L_n,w):
    return (x-w)/L_n

def g(x,L_n,w,tao,t,const):
    return erf(f(x,L_n*2,w)*np.sqrt(tao/t) + np.sqrt(t/tao) * const)

class LandauDist:
    def __init__(self,mu,sig):
        self.mu = mu
        self.sig = sig
    def pdf(self, x):
       return landau.pdf(x,self.mu,self.sig)
    def cdf(self, x):
       return landau.cdf(x,self.mu,self.sig)

def findThresholdCross(x,N,w,L_n,tao,threshold):
    func = lambda t : CCEFunc(x,(t)/(10**9),N,w,L_n,tao) - threshold
    if func(1*10**9) < 0:
        return -100
    return root_scalar(func, bracket= [10**(-9), 1*10**15]).root#+np.random.random()*25

def pixelToDepth(p):
    return p * (50 * 10 **(-6)) / np.tan(np.deg2rad(86.5)) 

def depthToPixel(d):
    return d / (50 * 10**(-6))  * np.tan(np.deg2rad(86.5))

def logGaussian(x,mu,sig,scaler):
    return lognorm.pdf(x,sig,scale=mu) * scaler

def logGaussianCDFFunc(x, mu, sig):
    return lognorm.cdf(x,sig,scale=mu)

def fitLogGaussian(x,y):
    popt,pcov = curve_fit(logGaussian,x,y)
    return popt, pcov

def gaussianBinned(x, mu, sigma, scaler, edges):
    return (logGaussianCDFFunc(edges[1:], mu, sigma) - logGaussianCDFFunc(edges[:-1], mu, sigma)) * scaler

def fitHistLogGaussian(data):
    data = np.log(data)
    height,edges = np.histogram(data[np.invert(np.isnan(data))],bins=200)
    func = lambda x, mu, sigma, scaler: gaussianBinned(
        x, mu, sigma, scaler, np.exp(edges)
    )
    binCentres = (edges[:-1] + edges[1:]) / 2
    popt, pcov = curve_fit(
        func,
        np.exp(binCentres),
        height,
    )
    return popt,pcov



plotGen = plotGenerator("/home/atlas/rballard/AtlasDataAnalysis/output/")
plot = plotGen.newPlot(f"Sim/")

N_A = 2e14
z = 15*10**(-6)
N = 100
L_n = electron_diffusion_length_p_type(N_A)/100
D = 36*10**(-4)
tao = (L_n**2) / D
w = depletion_width_scaled(N_A, 48.6)
print(f"{w*10**6:.2f} µm")
mpv = 100
threshold = 38
print(f"{L_n* 1e6:.2f} µm" )


x = np.linspace(0.001*10**(-7),1*10**(-6),1000)
y = CCEFunc(w+z,x,N,w,L_n,tao)
plot.axs.plot(x*10**9,y)
plot.set_config(
    plot.axs,
    title=f"Charge Collection over time at depth {(w+z)*10**6:.2f} μm\n depletion width of {w*10**6:.2f} μm",
    xlabel="Time [ns]",
    ylabel="Charge Collected [%]",
    xlim=(0,None),
    ylim=(0, None),
)
plot.saveToPDF(f"CDF_Charge_Collection_{N_A:.2e}")

plot = plotGen.newPlot(f"Sim/")
x = np.linspace(0,100*10**(-6),1000)
y = CCEFunc(w+x,1,N,w,L_n,tao)
plot.axs.plot((x+w)*10**6,y)
plot.axs.plot(np.linspace(0,w,100)*10**6,np.full(100,N))
plot.set_config(
    plot.axs,
    title=f"",
    xlabel="Depth [μm]",
    ylabel="Charge Collected [%]",
    xlim=(0,None),
    ylim=(0, None),
)
plot.saveToPDF(f"CCE_{N_A:.2e}")


plot = plotGen.newPlot(f"Sim/")
dist = LandauDist(mpv,mpv/4)
urng = np.random.default_rng()
rng = NumericalInversePolynomial(dist, random_state=urng)
N = rng.rvs(100000)
height,x = np.histogram(N,bins=200,range=(0,1000))
plot.axs.stairs(height,x)
N = moyal.rvs(loc=mpv, scale=mpv/4, size=100000)
height,x = np.histogram(N,bins=200,range=(0,1000))
plot.axs.stairs(height,x)
plot.set_config(
    plot.axs,
    xlim=(0,None),
    ylim=(0, None),
    title=f"Landau Random Number Test",
    xlabel="Output",
    ylabel="Count",

)
plot.saveToPDF(f"Landau_Test")

plot = plotGen.newPlot(f"Sim/")
x = np.linspace(0,1000,1000)
y = logGaussian(x,mpv,1,1)
plot.axs.plot(x,y,color=plot.colorPalette[0],label="Log Normal PDF")
y = dist.pdf(x)
plot.axs.plot(x,y,color=plot.colorPalette[1],label="Landau PDF")
plot.set_config(
    plot.axs,
    xlim=(0,None),
    ylim=(0, None),
    title=f"Log Normal - Landau Comparison",
    xlabel="Output",
    ylabel="Count",
    legend=True,
)
plot.saveToPDF(f"Log_Normal_VS_Landau")



plot = plotGen.newPlot(f"Sim/",sizePerPlot=(10,7))
pixels = np.arange(0,30,2)
depths = pixelToDepth(pixels)
i = 0
for z in depths:
    if z < w:
        continue
    i+=1
    N = rng.rvs(10000)
    timeToThreshold = np.log(np.array([findThresholdCross(z,n,w,L_n,tao,threshold) for n in N]))
    height,x = np.histogram(timeToThreshold[np.invert(np.isnan(timeToThreshold))],bins=100)
    binCentres = (x[:-1] + x[1:]) / 2
    popt,pcov = fitLogGaussian(np.exp(binCentres),height)
    plot.axs.stairs(height,x,label=f"{z*10**6:.0f} μm",color=plot.colorPalette[i])
    E = logGaussian(np.exp(binCentres),*popt)
    O = height
    print(np.sum(((O - E)**2)/E)/(len(binCentres[height>0])-4))
    print(popt)
    x = np.linspace(np.min(binCentres),np.max(binCentres),1000)
    y = logGaussian(np.exp(x),*popt)
    plot.axs.plot(x,y,label=f"Gaussian Fitting",color=plot.colorPalette[i])
plot.set_config(
    plot.axs,
    ylim=(0, None),
    title=f"Time To Threshold With Landu Log Dist",
    xlabel="log10(Time To Threshold)",
    ylabel="Count",
    legend=True,
)
plot.saveToPDF(f"Time_To_Threshold_{N_A:.2e}_log",close=False)

plot = plotGen.newPlot(f"Sim/",sizePerPlot=(10,7))
binWidth = 1
_range = (1,100)
bins = int(np.ptp(_range)/binWidth)
pixels = np.arange(0,30,2)
depths = pixelToDepth(pixels)
i = 0
for z in depths:
    if z < w:
        continue
    i+=1
    N = rng.rvs(10000)
    timeToThreshold = np.array([findThresholdCross(z,n,w,L_n,tao,threshold) for n in N])
    height,x = np.histogram(timeToThreshold,bins=bins,range=_range)
    plot.axs.stairs(height,x,label=f"{z*10**6:.0f} μm",color=plot.colorPalette[i])
    binCentres = (x[:-1] + x[1:]) / 2
    popt,pcov = fitLogGaussian(binCentres,height)
    popt,pcov = fitHistLogGaussian(timeToThreshold)
    _x = np.linspace((np.min(x[x>0])),(np.max(x)),1000)
    E = logGaussian(binCentres,*popt)
    O = height
    print(np.sum(((O - E)**2)/E)/(len(binCentres[height>0])-4))
    print(popt)
    y = logGaussian(_x,*popt)#/((x[1]-x[0])/2)
    plot.axs.plot(_x,y,label=f"Log Gaussian Fitting",color=plot.colorPalette[i])
plot.set_config(
    plot.axs,
    xlim=_range,
    ylim=(0, None),
    title=f"Time To Threshold With Landu Dist",
    xlabel="Time To Threshold [ns]",
    ylabel="Count",
    legend=True,
)
plot.saveToPDF(f"Time_To_Threshold_{N_A:.2e}",close=False)

plot = plotGen.newPlot(f"Sim/")
binWidth = 25
_range = (0,1000)
bins = int(np.ptp(_range)/binWidth)
pixels = np.arange(0,26)
depths = pixelToDepth(pixels)
relativeRowList = []
relativeTSList = []
maxRow = 25
TSRange = 20
for i,z in enumerate(depths):
    if z < w:
        continue
    N = rng.rvs(10000)
    timeToThreshold = np.array([findThresholdCross(z,n,w,L_n,tao,threshold) for n in N])/25
    timeToThreshold = list(timeToThreshold[np.invert(np.isnan(timeToThreshold))])
    relativeTSList.extend(timeToThreshold)
    relativeRowList.extend(list(np.full(len(timeToThreshold),depthToPixel(z))))
array, yedges, xedges = np.histogram2d(relativeTSList,relativeRowList,range=((0,TSRange),(-0.5,maxRow+0.5)),bins=(TSRange,maxRow+1))
im = plot.axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],vmax = 10000, vmin = 0)
divider = make_axes_locatable(plot.axs)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation="vertical")
cbar.set_label("TS", rotation=270, labelpad=15)

plot.set_config(plot.axs,
    title="Row vs TS",
    xlabel="Relative Row [px]",
    ylabel="Relative Timestamp [TS]",
    xlim = (xedges[0],xedges[-1]),
    ylim = (yedges[0],yedges[-1]),
    xticks=[5,1],
    yticks=[5,1],
)
plot.saveToPDF(f"Template_Graph_{N_A:.2e}")

