from typing import Optional, Any,Union
from dataAnalysis._dependencies import (
    plt,
    np,
    MultipleLocator,
)
import os
def getMultiplier(_range):
    log = np.log10(_range)
    ticks = np.floor(log-0.5)
    majorTicks = 10**ticks
    minorTicks = majorTicks/5
    while _range//majorTicks < 4:
        majorTicks = majorTicks/2
        minorTicks = minorTicks/2
    while _range//majorTicks > 15:
        majorTicks = majorTicks*2
        minorTicks = minorTicks*2
    return majorTicks,minorTicks
class plotClass:
    def __init__(
        self,
        pathToOutput: str,
        sizePerPlot: tuple[float, float] = (6.4, 4.8),
        shape: tuple[int, int] = (1, 1),
        sharex: bool = False,
        sharey: bool = False,
        hspace: Optional[float] = None,
    ):
        self.sizePerPlot = sizePerPlot
        self.shape = shape
        self.pathToOutput = pathToOutput
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(nrows=shape[1], ncols=shape[0], hspace=hspace)
        self.axs = gs.subplots(sharex=sharex, sharey=sharey)
        self.colorPalette = [
            "#333745",
            "#CC3F0C",
            "#33673B",
            "#9A6D38",
            "#EDB0E4",
            "#CC8A8A",
            "#239A7E",
            "#8896AB",
        ]

    def set_config(
        self,
        ax: Any,
        ylim: Optional[tuple[Optional[float], Optional[float]]] = None,
        xlim: Optional[tuple[Optional[float], Optional[float]]] = None,
        prettyTicks:bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: bool = False,
        legend_ncols: int = 1,
        legend_labelspacing: float = 0.5,
        legend_loc: str = "best",
        legend_handletextpad: float = 0.8,
        legend_columnspacing: float = 2,
        legend_Title: str = "",
        legend_frameon=False,
    ) -> None:
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if title is not None:
            ax.set_title(title)
        if legend:
            ax.legend(
                frameon=legend_frameon,
                ncols=legend_ncols,
                labelspacing=legend_labelspacing,
                loc=legend_loc,
                handletextpad=legend_handletextpad,
                columnspacing=legend_columnspacing,
                title=legend_Title,
            )
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if prettyTicks:
            xlim = ax.get_xlim()
            xRange = np.ptp([xlim[0],xlim[1]])
            xMajorTicks,xMinorTicks = getMultiplier(xRange)
            ax.xaxis.set_major_locator(MultipleLocator(xMajorTicks))
            ax.xaxis.set_minor_locator(MultipleLocator(xMinorTicks))
            if xMinorTicks > 1:
                ax.xaxis.set_major_formatter("{x:.0f}")
            else:
                ax.xaxis.set_major_formatter("{x}")
            ylim = ax.get_ylim()
            yRange = np.ptp([ylim[0],ylim[1]])
            yMajorTicks,yMinorTicks = getMultiplier(yRange)
            ax.yaxis.set_major_locator(MultipleLocator(yMajorTicks))
            ax.yaxis.set_minor_locator(MultipleLocator(yMinorTicks))
            if yMinorTicks > 1:
                ax.yaxis.set_major_formatter("{x:.0f}")
            else:
                ax.yaxis.set_major_formatter("{x}")

    def finalizePlot(self) -> None:
        self.fig.set_figwidth((self.sizePerPlot[0] * self.shape[0]))
        self.fig.set_figheight((self.sizePerPlot[1] * self.shape[1]))

    def saveToPDF(self, name: str) -> None:
        self.finalizePlot()
        out_put_file_name = f"{self.pathToOutput}" + f"{name}" + f".pdf"
        os.makedirs("/".join(out_put_file_name.split("/")[:-1]), exist_ok=True)
        self.fig.savefig(f"{out_put_file_name}")
        plt.close()
        print(f"Saved Plot: {out_put_file_name}")

    def addToPDF(self, pdf):
        self.finalizePlot()
        pdf.savefig(self.fig)