import sys
sys.path.append("..")
from dataAnalysis import dataAnalysis, clusterClass, clusterArray
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects
from matplotlib.ticker import MultipleLocator
import os
from typing import Optional, Any, TypeAlias
import numpy as np
import numpy.typing as npt



from matplotlib import rc
plt.rcParams['font.family'] = 'serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs
#rc('text', usetex=True)
rc("axes", titlesize=14,labelsize=10)
rc("xtick", labelsize=10)
rc("ytick",labelsize=10)
rc("legend",fontsize=10)


class plotGenerator:
    def __init__(
        self,
        pathToOutput: str,
    ):
        self.pathToOutput = pathToOutput
    def newPlot(self,pathToFile,**kwargs):
        return plotClass(f"{self.pathToOutput}{pathToFile}",**kwargs)

class plotClass:
    def __init__(
        self,
        pathToOutput: str,
        sizePerPlot: tuple[float, float] = (6.4, 4.8),
        shape: tuple[int, int] = (1, 1),
        sharex: bool = False,
        sharey: bool = False,
        hspace: Optional[float] = None,
        rect=None
    ):
        self.sizePerPlot = sizePerPlot
        self.shape = shape
        self.pathToOutput = pathToOutput
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(nrows=shape[1], ncols=shape[0], hspace=hspace)
        if self.shape[1] != 1 or self.shape[0] != 1:
            print("**")
            gs.tight_layout(self.fig,rect=rect)
            self.axs = gs.subplots(sharex=sharex, sharey=sharey)
        else:
            self.axs = gs.subplots(sharex=sharex, sharey=sharey)
            gs.tight_layout(self.fig,rect=rect)
        self.colorPalette = [
            "#CC3F0C",
            "#8896AB",
            "#2B337E",
            "#333745",
            "#EDB0E4",
            "#CC8A8A",
            "#239A7E",
            "#9A6D38",
            "#7B4173",
            "#525252",
            "#000000",
            "#AAAAAA",
        ]
        self.textColor = self.colorPalette[-1]
    def set_config(
        self,
        ax: Any,
        ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
        xlim: Optional[tuple[Optional[float], Optional[float]]] = None,
        title: Optional[str] = None,
        legend: bool = False,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ncols: int = 1,
        labelspacing: float = 0.5,
        loc: str = "best",
        handletextpad: float = 0.8,
        columnspacing: float = 2,
        legendTitle: str = "",
        labelcolor: str = None,
        xticks: list[int] = None,
        yticks: list[int] = None,
        xticksSig: int = None,
        yticksSig: int = None,
        grid = False,
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
                labelcolor=labelcolor,
            )
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.xaxis.set_major_locator(MultipleLocator(xticks[0]))
            ax.xaxis.set_major_formatter("{x:."+f"{0 if xticksSig is None else xticksSig}"+"f}")
            ax.xaxis.set_minor_locator(MultipleLocator(xticks[1]))
        if yticks is not None:
            ax.yaxis.set_major_locator(MultipleLocator(yticks[0]))
            ax.yaxis.set_major_formatter("{x:."+f"{0 if yticksSig is None else yticksSig}"+"f}")
            ax.yaxis.set_minor_locator(MultipleLocator(yticks[1]))
        if grid:
            ax.grid(which='major', color="#BBBBBB", linewidth=0.8,zorder=0)
            ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5,zorder=0)


    def finalizePlot(self) -> None:
        self.fig.set_figwidth((self.sizePerPlot[0] * self.shape[0]))
        self.fig.set_figheight((self.sizePerPlot[1] * self.shape[1]))

    def saveToPDF(self, name: str = "",close=True) -> None:
        self.finalizePlot()
        out_put_file_name = f"{self.pathToOutput}" + f"{name}" + f".pdf"
        os.makedirs("/".join(out_put_file_name.split("/")[:-1]), exist_ok=True)
        self.fig.savefig(f"{out_put_file_name}")
        if close:
            plt.close()
        print(f"\033[94mSaved Plot:\033[0m {name}.pdf\n\033[96m{out_put_file_name}\033[0m")

    def saveToPNG(self, name: str = "",close=True) -> None:
        self.finalizePlot()
        out_put_file_name = f"{self.pathToOutput}" + f"{name}" + f".png"
        os.makedirs("/".join(out_put_file_name.split("/")[:-1]), exist_ok=True)
        self.fig.savefig(f"{out_put_file_name}")
        if close:
            plt.close()
        print(f"\033[94mSaved Plot:\033[0m {name}.pdf\n\033[96m{out_put_file_name}\033[0m")


class clusterPlotter:
    def __init__(self, dataFile: dataAnalysis, buffer: int = 3, excludeCrossTalk: bool = True):
        self.dataFile = dataFile
        self.buffer = buffer
        self.excludeCrossTalk = excludeCrossTalk
        self.cmap = "plasma"
        #self.crossTalkFinder = crossTalkFinder()

    def plotClusters(self, ax: Any, clusters: clusterArray, z: str = "Hit_Voltages") -> Any:
        numberOfPoints = sum(
            cluster.getSize(excludeCrossTalk=self.excludeCrossTalk) for cluster in clusters
        )
        x = np.zeros(numberOfPoints, dtype=int)
        y = np.zeros(numberOfPoints, dtype=int)
        Hit_Voltage = np.zeros(numberOfPoints, dtype=float)
        count = 0
        cmap = plt.get_cmap(self.cmap)
        for cluster in clusters:
            x[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = (
                cluster.getRows(excludeCrossTalk=self.excludeCrossTalk)
            )
            y[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = (
                cluster.getColumns(excludeCrossTalk=self.excludeCrossTalk)
            )
            values = getattr(cluster, "get" + z)(excludeCrossTalk=self.excludeCrossTalk)
            if type(values) == type(np.array([])) and z == "TSs":
                values = values - np.min(values)
            Hit_Voltage[count : count + cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)] = (
                values
            )
            count += cluster.getSize(excludeCrossTalk=self.excludeCrossTalk)
        display, extent = self.constructDisplay(x, y)
        display = self.addToDisplay(display, x, y, Hit_Voltage)
        im = self.showDisplay(
            ax,
            display - np.nanmin(display),
            extent,
            vmin=0,
            #vmax=2,
            vmax=float(np.nanmax(display) - np.nanmin(display) + 1),
        )
        minTS = np.average(clusters[0].getEXT_TSs(excludeCrossTalk=self.excludeCrossTalk))
        for cluster in clusters:
            ang = np.random.uniform(-1, 2)
            value = getattr(cluster, "get" + z)(excludeCrossTalk=self.excludeCrossTalk)
            value = np.reshape(value, np.size(value))[0]
            time = f"{TStoMS(np.average(cluster.getEXT_TSs(excludeCrossTalk=self.excludeCrossTalk)) - minTS):.2f} ms"
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

    def constructDisplay(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        x_range = int(np.max(x) - np.min(x))
        y_range = int(np.max(y) - np.min(y))
        display = np.zeros((y_range + self.buffer, x_range + self.buffer))
        display[display == 0] = np.nan
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

    def addToDisplay(
        self,
        display: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        value: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        for i in range(len(x)):
            display[
                y[i] - np.min(y) + int((self.buffer - 1) / 2),
                x[i] - np.min(x) + int((self.buffer - 1) / 2),
            ] = value[i]
        return display

    def showDisplay(
        self,
        ax: Any,
        display: npt.NDArray[np.float64],
        extent: npt.NDArray[np.float64],
        vmin: float = 0,
        vmax: float = 1,
    ) -> Any:
        im = ax.imshow(
            display, cmap=self.cmap, extent=extent, aspect=3, origin="lower", vmin=vmin, vmax=vmax
        )
        return im

    def addCrossTalk(self, ax: Any, cluster: clusterClass, color: Any = "r"):
        rows = cluster.getRows(excludeCrossTalk=False)
        rows = rows[cluster.crossTalk]
        columns = cluster.getColumns(excludeCrossTalk=False)
        columns = columns[cluster.crossTalk]
        ax.scatter(rows, columns, s=2, c=color)
