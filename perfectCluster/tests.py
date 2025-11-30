from funcs import isFlat,gaussianBinned,gaussianFunc
import sys
sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from pixelCharges.plotCluster import clusterPlotter
from plotAnalysis import plotClass
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from dataAnalysis._fileReader import calcDataFileManager

def quad(x,a,b,c):
    return a*x**2+b*x+c

config = configLoader.loadConfig()
# config["filterDict"] = {"telescope":"kit","fileName":"angle1_4Gev_kit_1"}
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
#config["maxLine"] = 500000
dataFiles = initDataFiles(config)
for k, dataFile in enumerate(dataFiles):
    base_path = f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/Clusters/"
    plot1 = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    plot1List = []
    plot2 = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    plot2List = []
    plot3 = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    plot3List1 = []
    plot3List2 = []
    plot4 = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    plot5 = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
    lengthInDW = 820
    rowPitch = 50
    possibleRows = int(np.ceil(lengthInDW/50))+1
    print(possibleRows)
    clusters, indexes = dataFile.get_clusters(excludeCrossTalk=True, returnIndexes=True, layer=4)
    clusters = clusters
    for cluster in clusters:
        if not isFlat(cluster):
            continue
        relativeRows = abs(cluster.getRows(True) - np.max(cluster.getRows(True)))
        Timestamps = cluster.getTSs(True)
        relativeTS = Timestamps - np.min(Timestamps)
        if np.ptp(relativeRows) < possibleRows-4:
            continue
        path = base_path + f"Cluster_{cluster.getIndex()}/"
        #CP = clusterPlotter(cluster, path, "Line Check")
        sortIndexes = np.argsort(relativeRows)
        relativeRows = relativeRows[sortIndexes]
        relativeTS = relativeTS[sortIndexes]
        gaps = np.diff(relativeRows)
        if np.any(gaps>5):
            #print(gaps)
            gap = np.where(gaps>5)[0][0] + 1
            #print(gap)
            if gap > relativeRows.size/2:
                relativeTS = relativeTS[:gap]
                relativeRows = relativeRows[:gap]
            else:
                relativeTS = relativeTS[gap:]
                relativeRows = relativeRows[gap:]
        if np.all(relativeTS[-5:-1]<=2) and not np.all(relativeTS[1:5]<=2):
            relativeTS = np.flip(relativeTS)
            relativeRows = np.flip(relativeRows)
            #print("Flipped")
        if np.ptp(relativeRows) < possibleRows-2:
            continue
        #print(relativeRows[relativeRows<=possibleRows])
        #print(relativeTS[relativeRows<=possibleRows])
        #print(relativeRows[relativeRows>possibleRows])
        #print(relativeTS[relativeRows>possibleRows])

        lowTSs = relativeTS<=1
        firstPixel = lowTSs[0]
        #print(firstPixel)
        if len(np.where(lowTSs)[0]) > 0:
            lastLow = np.where(lowTSs)[0][-1]
        else:
            lastLow = -1
        #print(lastLow)
        TSdiffs = np.diff(relativeTS[lastLow:])/np.diff(relativeRows[lastLow:])
        #print(relativeTS[lastLow:])
        #print(TSdiffs)
        #CP.finishPlot("Relative TS", cluster.getTSs(True) - np.min(cluster.getTSs(True)))
        #CP = clusterPlotter(cluster, path, "ToT")
        #CP.finishPlot("ToT", cluster.getToTs(True))

        plot1List.append(relativeTS[0])

        plot2List.append(lastLow)

        plot3List1.extend(relativeRows)
        plot3List2.extend(relativeTS)
        #input()
    plot1.axs.hist(plot1List,bins=11,range=(-0.5,10.5))
    plot1.set_config(plot1.axs,
        title="First Pixel TS",
        xlabel="Relative TS",
        ylabel="Count",
        xlim = (0,10),
        )  
    plot1.saveToPDF("First_Pixel") 
    height,x = np.histogram(plot1List,bins=11,range=(-0.5,10.5))
    plot4.axs.plot(np.arange(11),np.cumsum(height)/np.sum(height))
    plot4.set_config(plot4.axs,
        title="First Pixel TS",
        xlabel="Relative TS",
        ylabel="Cumulative Count",
        xlim = (0,10),
        ylim = (0,None),
        )  
    plot4.saveToPDF("Cumulative_First_Pixel") 
    plot2.axs.hist(plot2List,bins=31,range=(-0.5,30.5))
    plot2.set_config(plot2.axs,
        title="Last Low Index",
        xlabel="Last Low Index",
        ylabel="Count",
        xlim = (0,30),
        )  
    plot2.saveToPDF("Last_Low")
    TSRange = 40
    RowRange = 25
    array, yedges, xedges = np.histogram2d(plot3List2,plot3List1,range=((-0.5,TSRange+0.5),(-0.5,RowRange+0.5)),bins=(TSRange+1,RowRange+1))
    plot3.axs.imshow(array,aspect='auto',origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
    estimate = []
    spread = []
    for i,x in enumerate(np.transpose(array)):
        bounds = ((-5,0,0),(30,30,np.sum(x)*2))
        if i > 20:
            popt,pcov = curve_fit(gaussianBinned,np.arange(x.size)[2:],x[2:],maxfev=5000,bounds=bounds)
        else:
            popt,pcov = curve_fit(gaussianBinned,np.arange(x.size),x,maxfev=5000,bounds=bounds)
        if popt[1] < 1:
            popt[1] = 1
        if popt[0] < 0.5:
            popt[0] = 0.5
        if i == 0:
            popt[1] = 5
        estimate.append(popt[0])
        spread.append(popt[1])
        if i in [0,8,13,20,25,29]:
            plot = plotClass(f"{config["pathToOutput"]}ClusterTracks/{dataFile.fileName}/TimeStamps/")
            axs = plot.axs
            _x = np.linspace(0,x.size,TSRange*10)
            axs.plot(_x,gaussianFunc(_x,*popt), color=plot.colorPalette[0])
            axs.stairs(x,yedges, baseline=None, color=plot.colorPalette[2],label = f"Slice of Row {i}")
            plot.set_config(axs,
                title=f"Relative TS distribution at row {i}",
                xlabel="Relative TS",
                ylabel="Count",
                xlim = (-0.5,TSRange+0.5),
                ylim = (0,None),
                legend=True,
                )  
            plot.axs.xaxis.set_major_locator(MultipleLocator(5))
            plot.axs.xaxis.set_major_formatter("{x:.0f}")
            plot.axs.xaxis.set_minor_locator(MultipleLocator(1))
            #plot.axs.yaxis.set_major_locator(MultipleLocator(5))
            #plot.axs.yaxis.set_major_formatter("{x:.0f}")
            #plot.axs.yaxis.set_minor_locator(MultipleLocator(1))
            plot.saveToPDF(f"Row_TS_{i}_slice")

    calcFileManager = calcDataFileManager(config["pathToCalcData"], "TSParams", config["maxLine"])
    estimate = np.array(estimate)
    spread = np.array(spread)
    calcFileName = calcFileManager.generateFileName(
        attribute=f"{dataFile.fileName}",
    )
    calcFileManager.saveFile(calcFileName=calcFileName,array=np.array([estimate,spread]))
    plot3.axs.scatter(np.arange(len(estimate)),estimate,marker="x",color=plot3.colorPalette[0],label="Gaussian Fitting on each Row\nError bars show std")
    plot3.axs.errorbar(
            np.arange(len(estimate)),
            estimate,
            yerr=spread,
            fmt="none",
            color=plot3.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )
    plot3.set_config(plot3.axs,
        title="Row vs TS",
        xlabel="Relative Row",
        ylabel="TS",
        xlim = (-0.5,RowRange+0.5),
        ylim = (-0.5,TSRange+0.5),
        legend=True,
        labelcolor="w",
        )
    x = np.linspace(0,RowRange,100)
    estimate = np.array(estimate)
    if dataFile.fileName == "angle6_4Gev_kit_2":
        z = np.polyfit(np.arange(estimate.size)[12:25],estimate[12:25],2)
        print(z)
    plot3.axs.plot(x,quad(x,*z),color=plot3.colorPalette[2],label="PolyFit")  
    plot3.axs.xaxis.set_major_locator(MultipleLocator(5))
    plot3.axs.xaxis.set_major_formatter("{x:.0f}")
    plot3.axs.xaxis.set_minor_locator(MultipleLocator(1))
    plot3.axs.yaxis.set_major_locator(MultipleLocator(5))
    plot3.axs.yaxis.set_major_formatter("{x:.0f}")
    plot3.axs.yaxis.set_minor_locator(MultipleLocator(1))
    plot3.saveToPDF("Row_TS") 
    plot1List = np.array(plot1List)
    plot2List = np.array(plot2List)
    height, x = np.histogram(plot2List[plot1List<=1],bins=TSRange+1,range=(-0.5,TSRange+0.5))
    plot5.axs.stairs(height/np.sum(height), x, baseline=None, color=plot5.colorPalette[0],label = "Low First Pixel")
    height, x = np.histogram(plot2List[plot1List>1],bins=TSRange+1,range=(-0.5,TSRange+0.5))
    plot5.axs.stairs(height/np.sum(height), x, baseline=None, color=plot5.colorPalette[1],label = "High First Pixel")
    #plot5.axs.hist([plot2List[plot1List<=1],plot2List[plot1List>1]],bins=31,range=(-0.5,30.5), stacked=True,color = [plot5.colorPalette[0],plot5.colorPalette[1]],label=["Low First Pixel","High First Pixel"])
    plot5.set_config(plot5.axs,
        title="First Pixel TS",
        xlabel="Relative TS",
        ylabel="Count",
        xlim = (0,TSRange),
        legend = True,
        )  
    plot5.saveToPDF("First_Pixel_Split")
