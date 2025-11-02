import sys
sys.path.append("..")
from dataAnalysis import initDataFiles,configLoader
import numpy as np
from plotAnalysis import plotClass
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

config = configLoader.loadConfig()
#config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
dataFiles = initDataFiles(config)


for dataFile in dataFiles:
    path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/"
    dataFile.init_cluster_voltages()
    clusters,indexes = dataFile.get_clusters(excludeCrossTalk=True,returnIndexes=True,layer=4)
    voltages = []
    voltages_e = []
    relativeTSs = []
    ToTs = []
    for cluster in clusters:
        if cluster.getSize(True)<15:
            continue
        TS = cluster.getTSs(True)-np.min(cluster.getTSs(True))
        voltage = cluster.getHit_Voltages(True)
        voltage_e = cluster.getHit_VoltageErrors(True)
        ToT = cluster.getToTs(True)
        for i in range(len(TS)):
            voltages.append(voltage[i])
            voltages_e.append(voltage_e[i])
            relativeTSs.append(TS[i])
            ToTs.append(ToT[i])
    plot = plotClass(f"{config["pathToOutput"]}TimeWalk/")
    axs = plot.axs
    #axs.scatter(relativeTSs,voltages,color=plot.colorPalette[0],marker="x")
    #axs.errorbar(relativeTSs, voltages, yerr=voltages_e, fmt="none", elinewidth=1, capsize=3, color=plot.colorPalette[0], alpha=0.5)
    array, yedges, xedges = np.histogram2d(relativeTSs,voltages,bins=(21,80),range=((-0.5, 20.5),(0.16, 0.5)))
    #im = axs.hist2d(voltages,relativeTSs,bins=(80,21),range=((0, 2),(-0.5, 20.5)),cmap="plasma")
    norm = LogNorm(vmin=1, vmax=np.max(array, where=~np.isnan(array), initial=-1))
    norm = None
    row_sums = array.sum(axis=0)
    array = array / row_sums[np.newaxis,:]
    im = axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],norm=norm)
    cbar = plot.fig.colorbar(im, ax=axs, label="Count")
    levels = [0.05]
    color = plot.colorPalette[0]
    cs = axs.contour(array,origin='lower',levels=levels,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],colors=color)
    cbar.ax.axhline(levels, color=color)
    x = np.linspace(0,0.5,100)
    y = np.exp(-35*x)*2000
    axs.plot(x,y,color="w",label="Exponential Fit Example")

    plot.set_config(        
        axs,
        xlim=(0.16, 0.5),
        ylim=(-0.5, 20.5),
        title=f"relativeTS vs Voltage Layer 4 {dataFile.fileName}",
        ylabel=f"TS [25ns]",
        xlabel=f"Voltage [V]",
    )
    plot.saveToPDF(f"{dataFile.fileName}_TimeWalk_Layer4_norm")
    continue
    plot = plotClass(f"{config["pathToOutput"]}TimeWalk/")
    axs = plot.axs
    #axs.scatter(relativeTSs,voltages,color=plot.colorPalette[0],marker="x")
    #axs.errorbar(relativeTSs, voltages, yerr=voltages_e, fmt="none", elinewidth=1, capsize=3, color=plot.colorPalette[0], alpha=0.5)
    array, yedges, xedges = np.histogram2d(relativeTSs,ToTs,bins=(21,256),range=((-0.5, 20.5),(-0.5, 255.5)))
    #im = axs.hist2d(ToTs,relativeTSs,bins=(256,21),range=((-0.5, 255.5),(-0.5, 20.5)),cmap="plasma")
    norm = LogNorm(vmin=1, vmax=np.max(array, where=~np.isnan(array), initial=-1))
    norm = None
    im = axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],norm=norm)
    plot.set_config(        
        axs,
        xlim=(-0.5, 255.5),
        ylim=(-0.5, 20.5),
        title=f"relativeTS vs Voltage Layer 4 {dataFile.fileName}",
        ylabel=f"TS [25ns]",
        xlabel=f"ToT [25ns]",
    )  
    plt.colorbar(im, ax=plot.axs, label="Count")         
    plot.saveToPDF(f"{dataFile.fileName}_TimeWalk_Layer4_ToT")


