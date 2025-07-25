from dataAnalysis import *
from lowLevelFunctions import *
import matplotlib.pyplot as plt
from  matplotlib import patheffects
import os
import scipy

def filterDataFiles(allDataFiles: list[dataAnalysis], filterDict: dict = {}):
    dataFiles = []
    for dataFile in allDataFiles:
        boolList = []
        for f in filterDict.keys():
            attr = getattr(dataFile, "get_" + f)
            try:
                boolList.append((np.isin(attr(), filterDict[f])))
            except:
                boolList.append((attr() == filterDict[f]))
        if np.all(boolList):
            dataFiles.append(dataFile)
    dataFiles = np.array(dataFiles)[np.argsort([dataFile.get_angle() * 1000 + dataFile.get_voltage() for dataFile in dataFiles])]
    for dataFile in dataFiles:
        if dataFile.get_telescope() == "lancs":
            dataFile.dataHandler.getCrossTalk()
    return np.flip(dataFiles)


class depthAnalysis:
    def __init__(self, pathToCalcData, maxLine=None, maxClusterWidth=40, layers: list[int] = None, excludeCrossTalk=False):
        self.calcFileManager = calcDataFileManager(pathToCalcData, "Stats", maxLine)
        self.maxClusterWidth = maxClusterWidth
        self.layers = layers
        self.excludeCrossTalk = excludeCrossTalk

    def Hit_VoltageByPixel(self, dataFile: dataAnalysis,measuredAttribute = "Hit_Voltage"):
        fileCheck = True
        attribute = f"{measuredAttribute}ByPixel"
        file = f"{dataFile.get_fileName()}/depthAnalysis/hist/"

        for i in range(self.maxClusterWidth-1):
            calcFileName = self.calcFileManager.generateFileName(
                attribute=attribute, cut=self.excludeCrossTalk, name=f"_{i+2}", file=file, layers=self.layers
            )
            fileCheck = self.calcFileManager.fileExists(calcFileName=calcFileName)
            if not fileCheck:
                break
        if fileCheck:
            return
        dataFile.init_cluster_voltages()
        hitPositionArray,hitPositionErrorArray , counts, indexes = calcHit_VoltageByPixel(
            dataFile.get_clusters(layers=self.layers, excludeCrossTalk=self.excludeCrossTalk),
            dataFile.get_cluster_attr("RowWidths", layers=self.layers, excludeCrossTalk=self.excludeCrossTalk),
            maxClusterWidth=self.maxClusterWidth,
            excludeCrossTalk=self.excludeCrossTalk,
            returnIndexes=True,
            measuredAttribute = measuredAttribute
        )
        for i in range(self.maxClusterWidth-1):
            calcFileName = self.calcFileManager.generateFileName(
                attribute=attribute, cut=self.excludeCrossTalk, name=f"_{i+2}", file=file, layers=self.layers
            )
            array = np.append(hitPositionArray[i, : i + 2, : counts[i]], [indexes[i, : counts[i]]], axis=0)
            self.calcFileManager.saveFile(array, calcFileName=calcFileName, suppressText=True)
            calcFileName = self.calcFileManager.generateFileName(
                attribute=f"{attribute}Error", cut=self.excludeCrossTalk, name=f"_{i+2}", file=file, layers=self.layers
            )
            array = np.append(hitPositionErrorArray[i, : i + 2, : counts[i]], [indexes[i, : counts[i]]], axis=0)
            self.calcFileManager.saveFile(array, calcFileName=calcFileName, suppressText=True)

    def loadOneLength(self, dataFile: dataAnalysis, clusterWidth: int, returnIndexes: bool=False,error:bool = False,measuredAttribute = "Hit_Voltage")->np.ndarray:
        attribute = f"{measuredAttribute}ByPixel"
        if error:
            attribute = f"{attribute}Error"
        name = f"_{clusterWidth}"
        file = f"{dataFile.get_fileName()}/depthAnalysis/hist/"
        calcFileName = self.calcFileManager.generateFileName(attribute=attribute, cut=self.excludeCrossTalk, name=name, file=file, layers=self.layers)
        if self.calcFileManager.fileExists(calcFileName=calcFileName):
            toBeReturned = self.calcFileManager.loadFile(calcFileName=calcFileName, suppressText=True)
        else:
            self.Hit_VoltageByPixel(dataFile,measuredAttribute = measuredAttribute)
            toBeReturned = self.calcFileManager.loadFile(calcFileName=calcFileName, suppressText=True)
        if returnIndexes:
            return toBeReturned[:-1], toBeReturned[-1]
        return toBeReturned[:-1]

    def findPeak(self, dataFile: dataAnalysis, clusterWidth: int, fitting:str="histogram",measuredAttribute = "Hit_Voltage")->tuple[list,list]:
        if fitting == "histogram":
            attribute = f"{measuredAttribute}Peaks"
        elif fitting == "nnlf":
            attribute = f"{measuredAttribute}Peaks_nnlf"
        name = f"_{clusterWidth}"
        file = f"{dataFile.get_fileName()}/depthAnalysis/peaks/"
        calcFileName = self.calcFileManager.generateFileName(attribute=attribute, cut=self.excludeCrossTalk, name=name, file=file, layers=self.layers)
        if self.calcFileManager.fileExists(calcFileName=calcFileName):
            peaks = self.calcFileManager.loadFile(calcFileName=calcFileName,suppressText=True)
            calcFileName = self.calcFileManager.generateFileName(attribute=f"{attribute}_errors", cut=self.excludeCrossTalk, name=name, file=file, layers=self.layers)
            errors = self.calcFileManager.loadFile(calcFileName=calcFileName,suppressText=True)
            toBeReturned = (peaks,errors)
        else:
            hitPositionArray = self.loadOneLength(dataFile, clusterWidth,measuredAttribute = measuredAttribute)
            hitPositionErrorArray = self.loadOneLength(dataFile, clusterWidth,error=True,measuredAttribute = measuredAttribute)
            if measuredAttribute == "Hit_Voltage":
                _range = (0.162, 2)
            elif measuredAttribute == "ToT":
                _range = (10, 256)
            if fitting == "histogram":
                params = [0,3]
            elif fitting == "nnlf":
                params = [0,2]
            output = (self.findPeaks_widthRestricted(hitPositionArray,hitPositionErrorArray, fitting=fitting,_range=_range,params=params))
            peaks,errors = output[:,0],output[:,1]
            self.calcFileManager.saveFile(peaks,calcFileName=calcFileName,suppressText=False)
            calcFileName = self.calcFileManager.generateFileName(attribute=f"{attribute}_errors", cut=self.excludeCrossTalk, name=name, file=file, layers=self.layers)
            self.calcFileManager.saveFile(errors,calcFileName=calcFileName,suppressText=True)
            toBeReturned = (peaks,errors*5)
        return toBeReturned

    def findPeaks_standard(self,hitPositionArray,hitPositionErrorArray:list, fitting="histogram",_range=(0.162, 2),params=[0]):
        clusterWidth = len(hitPositionArray)
        if len(params) > 1:
            peaks = np.zeros((clusterWidth,len(params)))
        else:
            peaks = np.zeros(clusterWidth)
        for i in range(len(hitPositionArray)):
            values = hitPositionArray[i, :][hitPositionArray[i, :] != 0]
            errors = hitPositionErrorArray[i, :][hitPositionArray[i, :] != 0]
            errors = errors[np.invert(np.isnan(values))]
            values = values[np.invert(np.isnan(values))]
            if fitting == "histogram":
                peaks[i] = self.fitPeak(values,errors=errors, returnParams=params, _range=_range)
            elif fitting == "nnlf":
                peaks[i] = self.fitPeak_nnlf(values,errors=errors, returnParams=params, _range=_range)
        return peaks
    
    def findPeaks_widthRestricted(self,hitPositionArray:list,hitPositionErrorArray:list,fitting:str="histogram",_range:tuple[float]=(0.162, 2),params:list[int]=[0])->np.ndarray:
        clusterWidth = len(hitPositionArray)
        if clusterWidth < 10:
            peaks = self.findPeaks_standard(hitPositionArray,hitPositionErrorArray,params=params, fitting=fitting,_range=_range)
        else:
            if len(params) > 1:
                peaks = np.zeros((clusterWidth,len(params)))
            else:
                peaks = np.zeros(clusterWidth)
            widths = np.zeros(((clusterWidth - 4 if clusterWidth < 20 else 15) - 4))
            for i in range(4, (clusterWidth - 4) if clusterWidth < 20 else 15):
                i = -i
                values = hitPositionArray[i, :][hitPositionArray[i, :] != 0]
                errors = hitPositionErrorArray[i, :][hitPositionArray[i, :] != 0]
                errors = errors[np.invert(np.isnan(values))]
                values = values[np.invert(np.isnan(values))]
                if fitting == "histogram":
                    widths[i + 4] = self.fitPeak(values,errors=errors, returnParams=[1], _range=_range)
                elif fitting == "nnlf":
                    widths[i + 4] = self.fitPeak_nnlf(values,errors=errors, returnParams=[1], _range=_range)
            avgWidth = np.mean(widths)
            xi_bounds = (avgWidth / 2, avgWidth * 1.5)
            for i in range(clusterWidth):
                values = hitPositionArray[i, :][hitPositionArray[i, :] != 0]
                errors = hitPositionErrorArray[i, :][hitPositionArray[i, :] != 0]
                errors = errors[np.invert(np.isnan(values))]
                values = values[np.invert(np.isnan(values))]
                if fitting == "histogram":
                    peaks[i] = self.fitPeak(values,errors=errors, returnParams=params, xi_bounds=xi_bounds, _range=_range)
                elif fitting == "nnlf":
                    peaks[i] = self.fitPeak_nnlf(values,errors=errors, returnParams=params, xi_bounds=xi_bounds, _range=_range)
        return np.array(peaks)
    def fitPeak_nnlf(self, values,errors:list[float]=None, returnParams=[0], **kwargs):
        return self.fitHit_VoltageLandau_nnlf(values,errors=errors, returnParams=returnParams, **kwargs)

    def fitHit_VoltageLandau_nnlf(self, values,errors:list[float]=None, returnParams=[0], _range=(0.162, 2), x_mpv_bounds=(0.05, 0.4), xi_bounds=(0.01, 0.2)):
        bounds = [x_mpv_bounds, xi_bounds]
        values = values[(values >= _range[0]) & (values <= _range[1])]
        result = scipy.optimize.differential_evolution(
            neg_log_likelihood_truncated, bounds=bounds, args=(values,), polish=True, updating="deferred", workers=-1
        )
        returnErrors = [np.full(len(result.x),np.average(errors)),np.full(len(result.x),np.average(errors))]
        toBeReturned = np.append(result.x, returnErrors)
        return toBeReturned[returnParams]

    def fitPeak(self, values,errors:list[float]=None, returnParams=[0], **kwargs):
        return self.fitHit_VoltageLandau(values,errors=errors, returnParams=returnParams, **kwargs)

    def fitHit_VoltageLandau(self, values,errors:list[float]=None, returnParams=[0], _range=(0.162, 2), x_mpv_bounds=None, xi_bounds=(0.01, 50)):#x_mpv_bounds=(0.2, 1.4), xi_bounds=(0.01, 1)):
        if x_mpv_bounds is None:
            x_mpv_bounds = _range
        hist, binEdges, binCentres = self.histogramHit_Voltage(values, range=_range)
        histErrors = histogramErrors(values,errors,binEdges)
        Z = (1 - landau.cdf(_range[0], x_mpv_bounds[0], xi_bounds[0])) * (binEdges[1]-binEdges[0])
        bounds = [x_mpv_bounds, xi_bounds, (Z*len(values), np.inf)]
        bounds = tuple(zip(*bounds))
        initial_guess = [binCentres[3:][np.argmax(hist[3:])], np.mean(xi_bounds), len(values)* (binEdges[1]-binEdges[0])]
        #print(histErrors)
        #print(hist)
        popt, pcov = scipy.optimize.curve_fit(landauFunc, binCentres[hist>0], hist[hist>0], p0=initial_guess, bounds=bounds,absolute_sigma=True,maxfev=500*(hist.size+10))
        toBeReturned = np.append(popt, np.sqrt(np.diag(pcov)))
        #if toBeReturned[3] < (binCentres[1]-binCentres[0])/5:
        #    toBeReturned[3] = (binCentres[1]-binCentres[0])/5
        return toBeReturned[returnParams]

    def histogramHit_Voltage(self, values, range=(0.162, 1)):
        number = np.sum(values > range[0])
        if number > 5000:
            bins = 84
        elif number > 500:
            bins = 42
        else:
            bins = 21
        hist, binEdges = np.histogram(values, bins=bins, range=range)
        binCentres = (binEdges[:-1] + binEdges[1:]) / 2
        return hist, binEdges, binCentres

    def findClusterWidthDistribution(self, dataFile):
        x = np.zeros(self.maxClusterWidth - 1)
        y = np.zeros(self.maxClusterWidth - 1)
        for i in range(2, self.maxClusterWidth + 1):
            x[i - 2] = i
            y[i - 2] = len(self.loadOneLength(dataFile, i)[0])
        return x, y

    def findClusterAngleDistribution(self, dataFile: dataAnalysis, d, maxColumnWidth=1):
        rowWidths = dataFile.get_cluster_attr("RowWidths", layers=self.layers, excludeCrossTalk=self.excludeCrossTalk)
        columnWidths = dataFile.get_cluster_attr("ColumnWidths", layers=self.layers, excludeCrossTalk=self.excludeCrossTalk)
        rowWidths = rowWidths[(columnWidths <= maxColumnWidth)]
        x, heights = np.unique(rowWidths[rowWidths < self.maxClusterWidth], return_counts=True)
        bins = np.append(np.atan((x[0] - 0.5) / d), np.atan((x + 0.5) / d))
        heights = heights / np.rad2deg(np.diff(bins))
        return bins, heights

    def residual(self, dataFile, d):
        bins, values = self.findClusterAngleDistribution(dataFile, d)
        ignoreFirst = 10
        maxValueIndex = np.argmax(values[ignoreFirst:])
        shift = 0
        return (np.average(np.rad2deg(bins[maxValueIndex + ignoreFirst + shift: maxValueIndex + 2 + ignoreFirst + shift])) - dataFile.get_angle()) ** 2
        bins = np.rad2deg(bins)
        binCentres1 = (bins[:-1] + bins[1:]) / 2
        peaks1,properties1 = scipy.signal.find_peaks(values[ignoreFirst:],width = 2,height=0)
        bins, values = self.findClusterAngleDistribution(dataFile, d, maxColumnWidth=2)
        bins = np.rad2deg(bins)
        binCentres2 = (bins[:-1] + bins[1:]) / 2
        peaks2,properties2 = scipy.signal.find_peaks(values[ignoreFirst:],width = 2,height=0)
        #print(binCentres1[peaks1])
        #print(properties1)
        #print(binCentres2[peaks2])
        #print(properties2)
        if len(peaks1) == 0 or len(peaks2) == 0:
            return (binCentres1[maxValueIndex] - dataFile.get_angle()) ** 2 * 2
        return (binCentres1[peaks1[np.argmax(properties1["peak_heights"])]+ignoreFirst] - dataFile.get_angle())**2 + (binCentres2[peaks2[np.argmax(properties2["peak_heights"])]+ignoreFirst] - dataFile.get_angle())**2
        

    def find_d_value(self, dataFile):
        func = lambda d: self.residual(dataFile, d)
        self.residual(dataFile, 1)
        res = scipy.optimize.minimize(func, [1], bounds=[(0.5, 1.75)])
        return res.x[0]


class plotClass:
    def __init__(self, pathToOutput, sizePerPlot=(6.4, 4.8), shape=(1, 1), sharex=False, sharey=False, hspace=None):
        self.sizePerPlot = sizePerPlot
        self.shape = shape
        self.pathToOutput = pathToOutput
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(nrows=shape[1], ncols=shape[0], hspace=hspace)
        self.axs = gs.subplots(sharex=sharex, sharey=sharey)
        self.colorPalette = ["#CC3F0C", "#9A6D38", "#33673B", "#333745", "#8896AB", "#EDB0E4", "#CC8A8A", "#239A7E"]
        self.textColor = self.colorPalette[-1]

    def set_config(
        self,
        ax,
        ylim=None,
        xlim=None,
        title=None,
        legend=False,
        xlabel=None,
        ylabel=None,
        ncols=1,
        labelspacing=0.5,
        loc="best",
        handletextpad=0.8,
        columnspacing=2,
        legendTitle="",
    ):
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if title is not None:
            ax.set_title(title)
        if legend:
            ax.legend(
                frameon=False,
                ncols=ncols,
                labelspacing=labelspacing,
                loc=loc,
                handletextpad=handletextpad,
                columnspacing=columnspacing,
                title=legendTitle,
            )
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    def saveToOutput(self, name):
        self.fig.set_figwidth((self.sizePerPlot[0] * self.shape[0]))
        self.fig.set_figheight((self.sizePerPlot[1] * self.shape[1]))
        out_put_file_name = f"{self.pathToOutput}" + f"{name}" + f".pdf"
        os.makedirs("/".join(out_put_file_name.split("/")[:-1]), exist_ok=True)
        self.fig.savefig(f"{out_put_file_name}")
        plt.close()
        print(f"Saved Plot: {out_put_file_name}")
        print_mem_usage()


class clusterPlotter:
    def __init__(self, dataFile: dataAnalysis, buffer=3, excludeCrossTalk=True):
        self.dataFile = dataFile
        self.buffer = buffer
        self.excludeCrossTalk = excludeCrossTalk
        self.cmap = "plasma"
        self.crossTalkFinder = crossTalkFinder()
    def plotClusters(self, ax, clusters: list[clusterClass], z="Hit_Voltages"):
        numberOfPoints = np.sum(cluster.getSize(excludeCrossTalk=self.excludeCrossTalk) for cluster in clusters)
        x = np.zeros(numberOfPoints, dtype=int)
        y = np.zeros(numberOfPoints, dtype=int)
        Hit_Voltage = np.zeros(numberOfPoints, dtype=float)
        count = 0
        cmap = plt.colormaps[self.cmap]
        for cluster in clusters:
            x[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = cluster.getRows(excludeCrossTalk=self.excludeCrossTalk)
            y[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = cluster.getColumns(excludeCrossTalk=self.excludeCrossTalk)
            Hit_Voltage[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = getattr(cluster, "get" + z)(
                excludeCrossTalk=self.excludeCrossTalk
            )  # cluster.getHit_Voltages(excludeCrossTalk = self.excludeCrossTalk)
            count += cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)
        display, extent = self.constructDisplay(x, y)
        display = self.addToDisplay(display, x, y, Hit_Voltage)
        im = self.showDisplay(ax, display - np.nanmin(display), extent, vmin=0, vmax=np.nanmax(display) - np.nanmin(display) + 1)
        minTS = np.average(clusters[0].getTSs(excludeCrossTalk=self.excludeCrossTalk))
        for cluster in clusters:
            ang = np.random.uniform(0, 2)
            value = getattr(cluster, "get" + z)(excludeCrossTalk=self.excludeCrossTalk)
            value = np.reshape(value, np.size(value))[0]
            time = f"{TStoMS(np.average(cluster.getTSs(excludeCrossTalk=self.excludeCrossTalk)) - minTS):.2f} ms"
            #time = f"{np.average(cluster.getTSs(excludeCrossTalk=self.excludeCrossTalk)):.0f}"
            ax.annotate(
                time,
                (
                    np.average(cluster.getRows(excludeCrossTalk=self.excludeCrossTalk)),
                    np.average(cluster.getColumns(excludeCrossTalk=self.excludeCrossTalk)),
                ),
                xytext=(
                    np.average(cluster.getRows(excludeCrossTalk=self.excludeCrossTalk) + 3 * np.sin(ang)),
                    np.average(cluster.getColumns(excludeCrossTalk=self.excludeCrossTalk)) + 3 * np.cos(ang),
                ),
                xycoords=ax.transData,
                textcoords=ax.transData,
                color=cmap((value - np.nanmin(display)) / (np.nanmax(display) - np.nanmin(display) + 1)),
                fontweight="bold",
                horizontalalignment="left",
                verticalalignment="bottom",
                arrowprops=dict(facecolor="black", shrink=0.05, headwidth=2, headlength=2, width=1),
                path_effects=[patheffects.withStroke(linewidth=1, foreground="w")],
            )
            #self.addCrossTalk(ax,cluster)
        return im

    def constructDisplay(self, x, y):
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        display = np.zeros((y_range + self.buffer, x_range + self.buffer))
        display[display == 0] = None
        # For each hit the ToT is added, this makes the colour of the pixel show the ToT
        # Sets the correct axis values
        extent = np.array(
            [
                np.min(x) - self.buffer / 2,
                np.max(x) + self.buffer / 2,
                np.min(y) - self.buffer / 2,
                np.max(y) + self.buffer / 2,
            ]
        )
        return display, extent

    def addToDisplay(self, display, x, y, value):
        for i in range(len(x)):
            display[
                y[i] - np.min(y) + int((self.buffer - 1) / 2),
                x[i] - np.min(x) + int((self.buffer - 1) / 2),
            ] = value[i]
        return display

    def showDisplay(self, ax, display, extent, vmin=0, vmax=1):
        im = ax.imshow(display, cmap=self.cmap, extent=extent, aspect=3, origin="lower", vmin=vmin, vmax=vmax)
        return im

    def addCrossTalk(self,ax,cluster:clusterClass,color="r"):
        rows = cluster.getRows(excludeCrossTalk=False)
        rows = rows[cluster.crossTalk]
        columns = cluster.getColumns(excludeCrossTalk=False)
        columns = columns[cluster.crossTalk]
        ax.scatter(rows, columns, s=2,c=color)


class correlationPlotter:
    def __init__(self,pathToCalcData,layers=None,excludeCrossTalk=True,maxLine=None):
        self.calcFileManager = calcDataFileManager(pathToCalcData, "Correlation", maxLine)
        self.layers = layers
        self.excludeCrossTalk = excludeCrossTalk
    def RowRowCorrelation(self,dataFile:dataAnalysis,recalc:bool=False):
        attribute = f"RowRowCorrelation"
        file = f"{dataFile.get_fileName()}/"
        calcFileName = self.calcFileManager.generateFileName(
                attribute=attribute, cut=self.excludeCrossTalk, name="", file=file, layers=self.layers
            )
        if "RowRow" in self.__dict__ and not recalc:
            toBeReturned = self.RowRow
        elif self.calcFileManager.fileExists(calcFileName=calcFileName) and not recalc:
            toBeReturned = self.calcFileManager.loadFile(calcFileName=calcFileName)
        else:
            clusters = dataFile.get_clusters(layers=self.layers,recalc=recalc)
            if self.excludeCrossTalk:
                dataFile.get_crossTalk(recalc=recalc)
            self.RowRow = np.zeros((371, 371))
            print(f"Finding RowRow correlation")
            rows = dataFile.get_base_attr("Row")
            indexes = rows - np.min(rows)
            for cluster in clusters:
                #print(cluster.notCrossTalk)
                #print(cluster.getRows(excludeCrossTalk = self.excludeCrossTalk))
                #input()
                for pixel in cluster.getIndexes(excludeCrossTalk = self.excludeCrossTalk):
                    self.RowRow[indexes[pixel], indexes[cluster.getIndexes(excludeCrossTalk = self.excludeCrossTalk)]] += 1
            self.RowRow[np.where(self.RowRow == 0)] = None
            self.calcFileManager.saveFile(self.RowRow,calcFileName=calcFileName)
            toBeReturned = self.RowRow
        return toBeReturned
if __name__ == "__main__":
    pathToData = "/home/atlas/rballard/for_magda/data/Cut/202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
    pathToData = "/home/atlas/rballard/for_magda/data/Cut/202204071512_udp_beamonall_angle6_4Gev_kit_2_decode.dat"
    pathToOutput = "/home/atlas/rballard/Code_v2/output/"
    pathToCalcData = "/home/atlas/rballard/Code_v2/calculatedData/"
    dataFile = dataAnalysis(pathToData, pathToCalcData, maxLine=None)
    depth = depthAnalysis(pathToData, pathToOutput, pathToCalcData, maxLine=None, maxClusterWidth=40, layers=[4], excludeCrossTalk=True)
    depth.findPeak(dataFile, 10)
