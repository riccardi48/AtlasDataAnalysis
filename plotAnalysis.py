from dataAnalysis import dataAnalysis, calcDataFileManager, crossTalkFinder, clusterClass
from lowLevelFunctions import *
import matplotlib.pyplot as plt
from matplotlib import patheffects
import os
import scipy
from typing import Optional,Any,TypeAlias
import numpy.typing as npt

clusterArray: TypeAlias = npt.NDArray[np.object_]

class depthAnalysis:
    def __init__(
        self,
        pathToCalcData:str,
        maxLine:Optional[int]=None,
        maxClusterWidth:int=40,
        layers: Optional[list[int]] = None,
        excludeCrossTalk:bool=False,
    ):
        self.calcFileManager = calcDataFileManager(pathToCalcData, "Stats", maxLine)
        self.maxClusterWidth = maxClusterWidth
        self.layers = layers
        self.excludeCrossTalk = excludeCrossTalk

    def Hit_VoltageByPixel(self, dataFile: dataAnalysis, measuredAttribute:str="Hit_Voltage") -> None:
        fileCheck = True
        attribute = f"{measuredAttribute}ByPixel"
        file = f"{dataFile.get_fileName()}/depthAnalysis/hist/"

        for i in range(self.maxClusterWidth - 1):
            calcFileName = self.calcFileManager.generateFileName(
                attribute=attribute,
                cut=self.excludeCrossTalk,
                name=f"_{i+2}",
                file=file,
                layers=self.layers,
            )
            fileCheck = self.calcFileManager.fileExists(calcFileName=calcFileName)
            if not fileCheck:
                break
        if fileCheck:
            return
        dataFile.init_cluster_voltages()
        hitPositionArray, hitPositionErrorArray, counts, indexes = calcHit_VoltageByPixel(
            dataFile.get_clusters(layers=self.layers, excludeCrossTalk=self.excludeCrossTalk),
            dataFile.get_cluster_attr(
                "RowWidths", layers=self.layers, excludeCrossTalk=self.excludeCrossTalk
            ),
            maxClusterWidth=self.maxClusterWidth,
            excludeCrossTalk=self.excludeCrossTalk,
            returnIndexes=True,
            measuredAttribute=measuredAttribute,
        )
        for i in range(self.maxClusterWidth - 1):
            calcFileName = self.calcFileManager.generateFileName(
                attribute=attribute,
                cut=self.excludeCrossTalk,
                name=f"_{i+2}",
                file=file,
                layers=self.layers,
            )
            array = np.append(
                hitPositionArray[i, : i + 2, : counts[i]], [indexes[i, : counts[i]]], axis=0
            )
            self.calcFileManager.saveFile(array, calcFileName=calcFileName, suppressText=True)
            calcFileName = self.calcFileManager.generateFileName(
                attribute=f"{attribute}Error",
                cut=self.excludeCrossTalk,
                name=f"_{i+2}",
                file=file,
                layers=self.layers,
            )
            array = np.append(
                hitPositionErrorArray[i, : i + 2, : counts[i]], [indexes[i, : counts[i]]], axis=0
            )
            self.calcFileManager.saveFile(array, calcFileName=calcFileName, suppressText=True)

    def loadOneLength(
        self,
        dataFile: dataAnalysis,
        clusterWidth: int,
        returnIndexes: bool = False,
        error: bool = False,
        measuredAttribute:str="Hit_Voltage",
    ) -> npt.NDArray[np.float64]:
        attribute = f"{measuredAttribute}ByPixel"
        if error:
            attribute = f"{attribute}Error"
        name = f"_{clusterWidth}"
        file = f"{dataFile.get_fileName()}/depthAnalysis/hist/"
        calcFileName = self.calcFileManager.generateFileName(
            attribute=attribute, cut=self.excludeCrossTalk, name=name, file=file, layers=self.layers
        )
        if self.calcFileManager.fileExists(calcFileName=calcFileName):
            toBeReturned = self.calcFileManager.loadFile(
                calcFileName=calcFileName, suppressText=True
            )
        else:
            self.Hit_VoltageByPixel(dataFile, measuredAttribute=measuredAttribute)
            toBeReturned = self.calcFileManager.loadFile(
                calcFileName=calcFileName, suppressText=True
            )
        if returnIndexes:
            return toBeReturned[:-1], toBeReturned[-1]
        return toBeReturned[:-1]

    def findPeak(
        self,
        dataFile: dataAnalysis,
        clusterWidth: int,
        fitting: str = "histogram",
        measuredAttribute:str="Hit_Voltage",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if fitting == "histogram":
            attribute = f"{measuredAttribute}Peaks"
        elif fitting == "nnlf":
            attribute = f"{measuredAttribute}Peaks_nnlf"
        name = f"_{clusterWidth}"
        file = f"{dataFile.get_fileName()}/depthAnalysis/peaks/"
        calcFileName = self.calcFileManager.generateFileName(
            attribute=attribute, cut=self.excludeCrossTalk, name=name, file=file, layers=self.layers
        )
        if self.calcFileManager.fileExists(calcFileName=calcFileName):
            peaks = self.calcFileManager.loadFile(calcFileName=calcFileName, suppressText=True)
            calcFileName = self.calcFileManager.generateFileName(
                attribute=f"{attribute}_errors",
                cut=self.excludeCrossTalk,
                name=name,
                file=file,
                layers=self.layers,
            )
            errors = self.calcFileManager.loadFile(calcFileName=calcFileName, suppressText=True)
            toBeReturned = (peaks, errors)
        else:
            hitPositionArray = self.loadOneLength(
                dataFile, clusterWidth, measuredAttribute=measuredAttribute
            )
            hitPositionErrorArray = self.loadOneLength(
                dataFile, clusterWidth, error=True, measuredAttribute=measuredAttribute
            )
            if measuredAttribute == "Hit_Voltage":
                _range = (0.162, 2)
            elif measuredAttribute == "ToT":
                _range = (10, 256)
            if fitting == "histogram":
                params = [0, 3]
            elif fitting == "nnlf":
                params = [0, 2]
            output = self.findPeaks_widthRestricted(
                hitPositionArray,
                hitPositionErrorArray,
                fitting=fitting,
                _range=_range,
                params=params,
            )
            peaks, errors = output[:, 0], output[:, 1]
            self.calcFileManager.saveFile(peaks, calcFileName=calcFileName, suppressText=False)
            calcFileName = self.calcFileManager.generateFileName(
                attribute=f"{attribute}_errors",
                cut=self.excludeCrossTalk,
                name=name,
                file=file,
                layers=self.layers,
            )
            self.calcFileManager.saveFile(errors, calcFileName=calcFileName, suppressText=True)
            toBeReturned = (peaks, errors * 5)
        return toBeReturned

    def findPeaks_standard(
        self,
        hitPositionArray:npt.NDArray[np.float64],
        hitPositionErrorArray: npt.NDArray[np.float64],
        fitting:str="histogram",
        _range:tuple[float,float]=(0.162, 2),
        params:list[int]=[0],
    ) -> npt.NDArray[np.float64]:
        clusterWidth = len(hitPositionArray)
        if len(params) > 1:
            peaks = np.zeros((clusterWidth, len(params)))
        else:
            peaks = np.zeros(clusterWidth)
        for i in range(len(hitPositionArray)):
            values = hitPositionArray[i, :][hitPositionArray[i, :] != 0]
            errors = hitPositionErrorArray[i, :][hitPositionArray[i, :] != 0]
            errors = errors[np.invert(np.isnan(values))]
            values = values[np.invert(np.isnan(values))]
            if fitting == "histogram":
                peaks[i] = self.fitPeak(values, errors=errors, returnParams=params, _range=_range)
            elif fitting == "nnlf":
                peaks[i] = self.fitPeak_nnlf(
                    values, errors=errors, returnParams=params, _range=_range
                )
        return peaks

    def findPeaks_widthRestricted(
        self,
        hitPositionArray: npt.NDArray[np.float64],
        hitPositionErrorArray: npt.NDArray[np.float64],
        fitting: str = "histogram",
        _range: tuple[float,float] = (0.162, 2),
        params: list[int] = [0],
    ) -> npt.NDArray[np.float64]:
        clusterWidth = len(hitPositionArray)
        if clusterWidth < 10:
            peaks = self.findPeaks_standard(
                hitPositionArray,
                hitPositionErrorArray,
                params=params,
                fitting=fitting,
                _range=_range,
            )
        else:
            if len(params) > 1:
                peaks = np.zeros((clusterWidth, len(params)))
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
                    widths[i + 4] = self.fitPeak(
                        values, errors=errors, returnParams=[1], _range=_range
                    )
                elif fitting == "nnlf":
                    widths[i + 4] = self.fitPeak_nnlf(
                        values, errors=errors, returnParams=[1], _range=_range
                    )
            avgWidth = np.mean(widths)
            xi_bounds = (avgWidth / 2, avgWidth * 1.5)
            for i in range(clusterWidth):
                values = hitPositionArray[i, :][hitPositionArray[i, :] != 0]
                errors = hitPositionErrorArray[i, :][hitPositionArray[i, :] != 0]
                errors = errors[np.invert(np.isnan(values))]
                values = values[np.invert(np.isnan(values))]
                if fitting == "histogram":
                    peaks[i] = self.fitPeak(
                        values,
                        errors=errors,
                        returnParams=params,
                        xi_bounds=xi_bounds,
                        _range=_range,
                    )
                elif fitting == "nnlf":
                    peaks[i] = self.fitPeak_nnlf(
                        values,
                        errors=errors,
                        returnParams=params,
                        xi_bounds=xi_bounds,
                        _range=_range,
                    )
        return np.array(peaks)

    def fitPeak_nnlf(self, values:npt.NDArray[np.float64], errors: Optional[npt.NDArray[np.float64]] = None, returnParams:list[int]=[0], **kwargs) -> npt.NDArray[np.float64]:
        return self.fitHit_VoltageLandau_nnlf(
            values, errors=errors, returnParams=returnParams, **kwargs
        )

    def fitHit_VoltageLandau_nnlf(
        self,
        values:npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]] = None,
        returnParams:list[int]=[0],
        _range: tuple[float,float] = (0.162, 2),
        x_mpv_bounds: tuple[float,float]=(0.05, 0.4),
        xi_bounds: tuple[float,float]=(0.01, 0.2),
    ) -> npt.NDArray[np.float64]:
        bounds = [x_mpv_bounds, xi_bounds]
        values = values[(values >= _range[0]) & (values <= _range[1])]
        result = scipy.optimize.differential_evolution(
            neg_log_likelihood_truncated,
            bounds=bounds,
            args=(values,),
            polish=True,
            updating="deferred",
            workers=-1,
        )
        returnErrors = [
            np.full(len(result.x), np.average(errors)),
            np.full(len(result.x), np.average(errors)),
        ]
        toBeReturned = np.append(result.x, returnErrors)
        return toBeReturned[returnParams]

    def fitPeak(self, values:npt.NDArray[np.float64], errors: Optional[npt.NDArray[np.float64]] = None, returnParams:list[int]=[0], **kwargs) -> npt.NDArray[np.float64]:
        return self.fitHit_VoltageLandau(values, errors=errors, returnParams=returnParams, **kwargs)

    def fitHit_VoltageLandau(
        self,
        values:npt.NDArray[np.float64],
        errors: Optional[npt.NDArray[np.float64]] = None,
        returnParams:list[int]=[0],
        _range: tuple[float,float] = (0.162, 2),
        x_mpv_bounds: Optional[tuple[float,float]]=None,
        xi_bounds:tuple[float,float]=(0.01, 50),
    ) -> npt.NDArray[np.float64]:  
        # x_mpv_bounds=(0.2, 1.4), xi_bounds=(0.01, 1)):
        if x_mpv_bounds is None:
            x_mpv_bounds = _range
        hist, binEdges, binCentres = self.histogramHit_Voltage(values, range=_range)
        histErrors = histogramErrors(values, errors, binEdges)
        Z = (1 - landau.cdf(_range[0], x_mpv_bounds[0], xi_bounds[0])) * (binEdges[1] - binEdges[0])
        bounds = [x_mpv_bounds, xi_bounds, (Z * len(values), np.inf)]
        bounds = tuple(zip(*bounds))
        initial_guess = [
            binCentres[3:][np.argmax(hist[3:])],
            np.mean(xi_bounds),
            len(values) * (binEdges[1] - binEdges[0]),
        ]
        # print(histErrors)
        # print(hist)
        popt, pcov = scipy.optimize.curve_fit(
            landauFunc,
            binCentres[hist > 0],
            hist[hist > 0],
            p0=initial_guess,
            bounds=bounds,
            absolute_sigma=True,
            maxfev=500 * (hist.size + 10),
        )
        toBeReturned = np.append(popt, np.sqrt(np.diag(pcov)))
        # if toBeReturned[3] < (binCentres[1]-binCentres[0])/5:
        #    toBeReturned[3] = (binCentres[1]-binCentres[0])/5
        return toBeReturned[returnParams]

    def histogramHit_Voltage(self, values:npt.NDArray[np.float64], range: tuple[float,float]=(0.162, 1)) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    def findClusterWidthDistribution(self, dataFile: dataAnalysis) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        x = np.zeros(self.maxClusterWidth - 1)
        y = np.zeros(self.maxClusterWidth - 1)
        for i in range(2, self.maxClusterWidth + 1):
            x[i - 2] = i
            y[i - 2] = len(self.loadOneLength(dataFile, i)[0])
        return x, y

    def findClusterAngleDistribution(self, dataFile: dataAnalysis, d : float, maxColumnWidth : int=1) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        rowWidths = dataFile.get_cluster_attr(
            "RowWidths", layers=self.layers, excludeCrossTalk=self.excludeCrossTalk
        )
        columnWidths = dataFile.get_cluster_attr(
            "ColumnWidths", layers=self.layers, excludeCrossTalk=self.excludeCrossTalk
        )
        rowWidths = rowWidths[(columnWidths <= maxColumnWidth)]
        x, heights = np.unique(rowWidths[rowWidths < self.maxClusterWidth], return_counts=True)
        bins = np.append(np.atan((x[0] - 0.5) / d), np.atan((x + 0.5) / d))
        heights = heights / np.rad2deg(np.diff(bins))
        return bins, heights

    def residual(self, dataFile: dataAnalysis, d : float) -> float:
        bins, values = self.findClusterAngleDistribution(dataFile, d)
        ignoreFirst = 10
        maxValueIndex = np.argmax(values[ignoreFirst:])
        shift = 0
        return (
            np.average(
                np.rad2deg(
                    bins[
                        maxValueIndex
                        + ignoreFirst
                        + shift : maxValueIndex
                        + 2
                        + ignoreFirst
                        + shift
                    ]
                )
            )
            - dataFile.get_angle()
        ) ** 2
        bins = np.rad2deg(bins)
        binCentres1 = (bins[:-1] + bins[1:]) / 2
        peaks1, properties1 = scipy.signal.find_peaks(values[ignoreFirst:], width=2, height=0)
        bins, values = self.findClusterAngleDistribution(dataFile, d, maxColumnWidth=2)
        bins = np.rad2deg(bins)
        binCentres2 = (bins[:-1] + bins[1:]) / 2
        peaks2, properties2 = scipy.signal.find_peaks(values[ignoreFirst:], width=2, height=0)
        # print(binCentres1[peaks1])
        # print(properties1)
        # print(binCentres2[peaks2])
        # print(properties2)
        if len(peaks1) == 0 or len(peaks2) == 0:
            return (binCentres1[maxValueIndex] - dataFile.get_angle()) ** 2 * 2
        return (
            binCentres1[peaks1[np.argmax(properties1["peak_heights"])] + ignoreFirst]
            - dataFile.get_angle()
        ) ** 2 + (
            binCentres2[peaks2[np.argmax(properties2["peak_heights"])] + ignoreFirst]
            - dataFile.get_angle()
        ) ** 2

    def find_d_value(self, dataFile: dataAnalysis) -> float:
        func = lambda d: self.residual(dataFile, d)
        self.residual(dataFile, 1)
        res = scipy.optimize.minimize(func, [1], bounds=[(0.5, 1.75)])
        return res.x[0]


class plotClass:
    def __init__(
        self,
        pathToOutput:str,
        sizePerPlot:tuple[float,float]=(6.4, 4.8),
        shape:tuple[int,int]=(1, 1),
        sharex:bool=False,
        sharey:bool=False,
        hspace:Optional[float]=None,
    ):
        self.sizePerPlot = sizePerPlot
        self.shape = shape
        self.pathToOutput = pathToOutput
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(nrows=shape[1], ncols=shape[0], hspace=hspace)
        self.axs = gs.subplots(sharex=sharex, sharey=sharey)
        self.colorPalette = [
            "#CC3F0C",
            "#9A6D38",
            "#33673B",
            "#333745",
            "#8896AB",
            "#EDB0E4",
            "#CC8A8A",
            "#239A7E",
        ]
        self.textColor = self.colorPalette[-1]

    def set_config(
        self,
        ax: Any,
        ylim:Optional[tuple[float,float]]=None,
        xlim:Optional[tuple[float,float]]=None,
        title:Optional[str]=None,
        legend:bool=False,
        xlabel:Optional[str]=None,
        ylabel:Optional[str]=None,
        ncols:int=1,
        labelspacing:float=0.5,
        loc:str="best",
        handletextpad:float=0.8,
        columnspacing:float=2,
        legendTitle:str="",
    ) -> None:
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

    def finalizePlot(self) -> None:
        self.fig.set_figwidth((self.sizePerPlot[0] * self.shape[0]))
        self.fig.set_figheight((self.sizePerPlot[1] * self.shape[1]))

    def saveToPDF(self, name:str) -> None:
        self.finalizePlot()
        out_put_file_name = f"{self.pathToOutput}" + f"{name}" + f".pdf"
        os.makedirs("/".join(out_put_file_name.split("/")[:-1]), exist_ok=True)
        self.fig.savefig(f"{out_put_file_name}")
        plt.close()
        print(f"Saved Plot: {out_put_file_name}")
        print_mem_usage()


class clusterPlotter:
    def __init__(self, dataFile: dataAnalysis, buffer:int=3, excludeCrossTalk:bool=True):
        self.dataFile = dataFile
        self.buffer = buffer
        self.excludeCrossTalk = excludeCrossTalk
        self.cmap = "plasma"
        self.crossTalkFinder = crossTalkFinder()

    def plotClusters(self, ax:Any, clusters: clusterArray, z:str="Hit_Voltages") -> Any:
        numberOfPoints = np.sum(
            cluster.getSize(excludeCrossTalk=self.excludeCrossTalk) for cluster in clusters
        )
        x = np.zeros(numberOfPoints, dtype=int)
        y = np.zeros(numberOfPoints, dtype=int)
        Hit_Voltage = np.zeros(numberOfPoints, dtype=float)
        count = 0
        cmap = plt.colormaps[self.cmap]
        for cluster in clusters:
            x[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = (
                cluster.getRows(excludeCrossTalk=self.excludeCrossTalk)
            )
            y[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = (
                cluster.getColumns(excludeCrossTalk=self.excludeCrossTalk)
            )
            Hit_Voltage[
                count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)
            ] = getattr(cluster, "get" + z)(
                excludeCrossTalk=self.excludeCrossTalk
            )  # cluster.getHit_Voltages(excludeCrossTalk = self.excludeCrossTalk)
            count += cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)
        display, extent = self.constructDisplay(x, y)
        display = self.addToDisplay(display, x, y, Hit_Voltage)
        im = self.showDisplay(
            ax,
            display - np.nanmin(display),
            extent,
            vmin=0,
            vmax=np.nanmax(display) - np.nanmin(display) + 1,
        )
        minTS = np.average(clusters[0].getTSs(excludeCrossTalk=self.excludeCrossTalk))
        for cluster in clusters:
            ang = np.random.uniform(0, 2)
            value = getattr(cluster, "get" + z)(excludeCrossTalk=self.excludeCrossTalk)
            value = np.reshape(value, np.size(value))[0]
            time = f"{TStoMS(np.average(cluster.getTSs(excludeCrossTalk=self.excludeCrossTalk)) - minTS):.2f} ms"
            # time = f"{np.average(cluster.getTSs(excludeCrossTalk=self.excludeCrossTalk)):.0f}"
            ax.annotate(
                time,
                (
                    np.average(cluster.getRows(excludeCrossTalk=self.excludeCrossTalk)),
                    np.average(cluster.getColumns(excludeCrossTalk=self.excludeCrossTalk)),
                ),
                xytext=(
                    np.average(
                        cluster.getRows(excludeCrossTalk=self.excludeCrossTalk) + 3 * np.sin(ang)
                    ),
                    np.average(cluster.getColumns(excludeCrossTalk=self.excludeCrossTalk))
                    + 3 * np.cos(ang),
                ),
                xycoords=ax.transData,
                textcoords=ax.transData,
                color=cmap(
                    (value - np.nanmin(display)) / (np.nanmax(display) - np.nanmin(display) + 1)
                ),
                fontweight="bold",
                horizontalalignment="left",
                verticalalignment="bottom",
                arrowprops=dict(facecolor="black", shrink=0.05, headwidth=2, headlength=2, width=1),
                path_effects=[patheffects.withStroke(linewidth=1, foreground="w")],
            )
            # self.addCrossTalk(ax,cluster)
        return im

    def constructDisplay(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    def addToDisplay(self, display: npt.NDArray[np.float64], x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        for i in range(len(x)):
            display[
                y[i] - np.min(y) + int((self.buffer - 1) / 2),
                x[i] - np.min(x) + int((self.buffer - 1) / 2),
            ] = value[i]
        return display

    def showDisplay(self, ax: Any, display: npt.NDArray[np.float64], extent:npt.NDArray[np.float64], vmin:float=0, vmax:float=1) -> Any:
        im = ax.imshow(
            display, cmap=self.cmap, extent=extent, aspect=3, origin="lower", vmin=vmin, vmax=vmax
        )
        return im

    def addCrossTalk(self, ax: Any, cluster: clusterClass, color:Any="r"):
        rows = cluster.getRows(excludeCrossTalk=False)
        rows = rows[cluster.crossTalk]
        columns = cluster.getColumns(excludeCrossTalk=False)
        columns = columns[cluster.crossTalk]
        ax.scatter(rows, columns, s=2, c=color)


class correlationPlotter:
    def __init__(self, pathToCalcData:str, layers:Optional[list[int]]=None, excludeCrossTalk:bool=True, maxLine:Optional[int]=None):
        self.calcFileManager = calcDataFileManager(pathToCalcData, "Correlation", maxLine)
        self.layers = layers
        self.excludeCrossTalk = excludeCrossTalk

    def RowRowCorrelation(self, dataFile: dataAnalysis, recalc: bool = False) -> npt.NDArray[np.float64]:
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
            clusters = dataFile.get_clusters(layers=self.layers, recalc=recalc)
            if self.excludeCrossTalk:
                dataFile.get_crossTalk(recalc=recalc)
            self.RowRow = np.zeros((371, 371))
            print(f"Finding RowRow correlation")
            rows = dataFile.get_base_attr("Row")
            indexes = rows - np.min(rows)
            for cluster in clusters:
                # print(cluster.notCrossTalk)
                # print(cluster.getRows(excludeCrossTalk = self.excludeCrossTalk))
                # input()
                for pixel in cluster.getIndexes(excludeCrossTalk=self.excludeCrossTalk):
                    self.RowRow[
                        indexes[pixel],
                        indexes[cluster.getIndexes(excludeCrossTalk=self.excludeCrossTalk)],
                    ] += 1
            self.RowRow[np.where(self.RowRow == 0)] = None
            self.calcFileManager.saveFile(self.RowRow, calcFileName=calcFileName)
            toBeReturned = self.RowRow
        return toBeReturned


if __name__ == "__main__":
    pathToData = "/home/atlas/rballard/for_magda/data/Cut/202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
    pathToData = "/home/atlas/rballard/for_magda/data/Cut/202204071512_udp_beamonall_angle6_4Gev_kit_2_decode.dat"
    pathToOutput = "/home/atlas/rballard/Code_v2/output/"
    pathToCalcData = "/home/atlas/rballard/Code_v2/calculatedData/"
    dataFile = dataAnalysis(pathToData, pathToCalcData, maxLine=None)
    depth = depthAnalysis(
        pathToData,
        pathToOutput,
        pathToCalcData,
        maxLine=None,
        maxClusterWidth=40,
        layers=[4],
        excludeCrossTalk=True,
    )
    depth.findPeak(dataFile, 10)
