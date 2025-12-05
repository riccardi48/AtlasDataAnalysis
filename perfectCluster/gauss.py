from funcs import gaussianBinned,gaussianCDFFunc,gaussianFunc
import matplotlib.pyplot as plt
import numpy as np

mu = 10.3
sigma = 0.7

x = np.linspace(0,30,300)
y = gaussianFunc(x,mu,sigma)
plt.figure()
plt.plot(x,y)
x = np.linspace(0,30,31)
y = gaussianBinned(x,mu,sigma,1)
plt.scatter(x,y)
plt.grid()
plt.savefig("/home/atlas/rballard/AtlasDataAnalysis/perfectCluster/test.pdf")
