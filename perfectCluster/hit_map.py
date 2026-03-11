import sys
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from plotFiles.plotClass import plotGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt

config = configLoader.loadConfig()
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

plotGen = plotGenerator(config["pathToOutput"])

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    x_range = (0,372)
    y_range = (0,132)
    # the scatter plot:
    ax.scatter(x, y,marker="x")
    # now determine nice limits by hand:
    binwidth = 6
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=int(372/2),range=x_range)
    ax_histy.hist(y, bins=132,range=y_range, orientation='horizontal')
    ax_histx.set_xlim(x_range)
    ax_histy.set_ylim(y_range)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

for dataFile in dataFiles:
    clusters = dataFile.get_perfectClusters(layer=4,excludeCrossTalk=True)
    path = f"HitMap/{dataFile.fileName}/"
    plot = plotGen.newPlot(path)
    
    seeds_x = []
    seeds_y = []
    ends_x = []
    ends_y = []
    for cluster in tqdm(clusters):
        rows = cluster.getRows(excludeCrossTalk=True)
        columns = cluster.getColumns(excludeCrossTalk=True)
        seed = np.argmin(rows)
        end = np.argmax(rows)
        if cluster.flipped:
            seed,end = end,seed
        if not cluster.flipped:
            continue
        seeds_x.append(rows[seed])
        seeds_y.append(columns[seed])
        ends_x.append(rows[end])
        ends_y.append(columns[end])
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(10, 10),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
    scatter_hist(seeds_x, seeds_y, axs['scatter'], axs['histx'], axs['histy'])
    plot.saveToPDF("test")
    fig.savefig(f"/home/atlas/rballard/AtlasDataAnalysis/output/HitMap/{dataFile.fileName}/seeds.pdf")

    fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(10, 10),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
    scatter_hist(ends_x, ends_y, axs['scatter'], axs['histx'], axs['histy'])
    fig.savefig(f"/home/atlas/rballard/AtlasDataAnalysis/output/HitMap/{dataFile.fileName}/ends.pdf")

            
