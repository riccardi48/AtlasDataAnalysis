import sys
from funcs import fitTemplate,getTemplate,findSections,gaussian_loglike_pval,convertRowsForFit,scaleOnGaussian,scaleTemplate,isFlat,isOnePixel,isOnEdge,filterForTemplate
from findPerfectCluster import graphTSonRows
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm

def graphTSonRows(axs,cluster,section,estimate,spread,flipped,estimateTemplate,spreadTemplate,excludeCrossTalk=True):
    Timestamps = cluster.getTSs(excludeCrossTalk=excludeCrossTalk)
    Rows = cluster.getRows(excludeCrossTalk=excludeCrossTalk)
    relativeRows = Rows-np.min(Rows)
    relativeTS = Timestamps - np.min(Timestamps)
    mask = np.ones(len(relativeRows), dtype=bool)
    mask[section,] = False
    x = relativeRows
    if flipped:
        x = abs(x-np.max(x))
    axs.scatter(x[~mask], relativeTS[~mask], color=plot.colorPalette[2], marker="x",label="In Fitted Sections")
    axs.scatter(x[mask], relativeTS[mask], color=plot.colorPalette[8], marker="x",label="Not in Fitted Sections")
    if len(section) != 0:
        if flipped:
            anchor = np.argmax(relativeRows[section])
            rowsForFunc = relativeRows[section][anchor]-relativeRows
        else:
            anchor = np.argmin(relativeRows[section])
            rowsForFunc = relativeRows-relativeRows[section][anchor]
        sortIndexes = np.argsort(relativeRows)
        rowsForFunc = rowsForFunc[sortIndexes]
        relativeRows = relativeRows[sortIndexes]
        rowsToGraph = np.linspace(relativeRows[rowsForFunc==0][0],relativeRows[rowsForFunc==0][0]+len(estimate)+(len(estimate)*-int(flipped)*2),len(estimate))
        if flipped:
            rowsToGraph = abs(rowsToGraph-np.max(rowsToGraph))
        axs.plot(rowsToGraph,estimate,color=plot.colorPalette[0],linestyle="dashed",label=f"Fitted")
        axs.fill_between(rowsToGraph,estimate-spread,estimate+spread, alpha=0.2,color=plot.colorPalette[0])
        rowsToGraph = np.linspace(relativeRows[rowsForFunc==0][0],relativeRows[rowsForFunc==0][0]+len(estimateTemplate)+(len(estimateTemplate)*-int(flipped)*2),len(estimateTemplate))
        if flipped:
            rowsToGraph = abs(rowsToGraph-np.max(rowsToGraph))
        axs.plot(rowsToGraph,estimateTemplate,color=plot.colorPalette[3],linestyle="dashed",label=f"Original")
        axs.fill_between(rowsToGraph,estimateTemplate-spreadTemplate,estimateTemplate+spreadTemplate, alpha=0.2,color=plot.colorPalette[3])
    plot.set_config(axs,
        title="Relative TS in cluster",
        xlabel="Relative Row",
        ylabel="Relative TS",
        legend=True,
        ylim=(-0.5,np.max(relativeTS)+5),
        xlim=(np.min(relativeRows)-1,np.max(relativeRows)+1),
    )  
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1))
    


np.set_printoptions(precision=3,linewidth=125)

config = configLoader.loadConfig()
#config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
    estimate,spread = getTemplate(config)
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    params1List = []
    params2List = []
    firstPixel = []
    i = 0
    k = 0
    for cluster in tqdm(clusters, desc='Running over clusters'):
        if not isFlat(cluster):
            continue
        if isOnePixel(cluster):
            continue
        if isOnEdge(cluster):
            continue
        if cluster.getSize(excludeCrossTalk=True) <= 4:
            continue
        #if not isPerfectCluster(cluster,estimate,spread,minPval=0,excludeCrossTalk=True):
        #    continue
        sections = findSections(cluster,excludeCrossTalk=True)
        section = np.arange(len(cluster.getRows(excludeCrossTalk=True)))
        Timestamps = cluster.getTSs(excludeCrossTalk=True)[section]
        Rows = cluster.getRows(excludeCrossTalk=True)[section]
        relativeTS = Timestamps - np.min(Timestamps)
        if np.all(relativeTS<=6):
            continue
        if np.sum(relativeTS>6) <= 2:
            continue
        i += 1
        #print(Rows,Timestamps)
        #start = time.time()
        params,flipped = fitTemplate(cluster,section,estimate,spread,excludeCrossTalk=True)
        pVal = gaussian_loglike_pval(scaleOnGaussian(*filterForTemplate(*convertRowsForFit(Rows,Timestamps,flipped=flipped),*scaleTemplate(estimate,spread,params[0],params[1]))))
        if pVal < 0.2:
            k += 1
            continue
        #end = time.time()
        #print(f"{end - start} seconds to fit")
        if params[0] == 0.5 and params[1] == 0.5:
            k += 1
            continue
        x,y = convertRowsForFit(Rows,Timestamps,flipped=flipped)
        params1List.append(params[0])
        params2List.append(params[1])
        firstPixel.append(y[x==0])
        continue
        print(params)
        #print(flipped)
        path = base_path + f"Cluster_{cluster.getIndex()}/"
        #print(scaleTemplate(estimate,spread,*params)[0])
        #print(estimate)
        plot = plotClass(path)
        axs = plot.axs
        graphTSonRows(axs,cluster,section,*scaleTemplate(estimate,spread,*params),flipped,estimate,spread)

        plot.saveToPDF(f"Cluster_{cluster.getIndex()}_Row_vs_RelativeTS")
        input("")
    print("")
    print(f"{len(params1List)/i*100:.2f}% of {i} clusters had fits")
    print(f"{k/i*100:.2f}% of {i} clusters had failed fits")
    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    array, yedges, xedges = np.histogram2d(params1List,params2List,bins=(100,100),range=((0,2),(0,2)))
    axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    plot.set_config(axs,
        title="Fitted Params",
        xlabel="Flat Scaler",
        ylabel="Angle Scaler",
    )  
    plot.saveToPDF(f"Fitting_Params")
    plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    axs = plot.axs
    params1List = np.array(params1List)
    params2List = np.array(params2List)
    ratio = (params1List+params2List)/2
    angle = np.rad2deg(np.tan(ratio*np.arctan(np.deg2rad(dataFile.angle))))
    height,x = np.histogram(angle,bins=90,range=(0,90))
    axs.stairs(height, x, baseline=None, color=plot.colorPalette[0])
    plot.set_config(axs,
        title="Angle of clusters",
        xlabel="Angle",
        ylabel="Count",
        #xlim=(75,90),
    )  
    plot.saveToPDF(f"Fitting_Params_Angle")