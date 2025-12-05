from dataAnalysis._types import dataAnalysis
from typing import Optional, Any, Union
from ._genericModule import plotModule
from ._plot import plotClass
from dataAnalysis._dependencies import (
    np,
)

class crossTalkModule(plotModule):
    def CuttingComparison(self, config: Optional[dict] = None):
        config = self.configCheck(config)
        plot = plotClass(self.pathToOutput, shape=(2, 2), sizePerPlot=(5, 4))
        axs = plot.axs
        _CuttingComparisonHistogram(
            axs[0, 0],
            plot,
            self.dataFile.get_base_attr("ToT", layers=config["layers"])[0],
            self.dataFile.get_base_attr("ToT", layers=config["layers"], excludeCrossTalk=True)[0],
            bins=128,
            _range=(0, 256),
            xlabel="ToT [TS]",
        )
        _CuttingComparisonHistogram(
            axs[1, 0],
            plot,
            self.dataFile.get_base_attr("Column", layers=config["layers"])[0],
            self.dataFile.get_base_attr("Column", layers=config["layers"], excludeCrossTalk=True)[0],
            bins=131,
            _range=(1, 132),
            xlabel="Column [px]",
        )
        _CuttingComparisonHistogram(
            axs[1, 1],
            plot,
            self.dataFile.get_base_attr("Row", layers=config["layers"])[0],
            self.dataFile.get_base_attr("Row", layers=config["layers"], excludeCrossTalk=True)[0],
            bins=371,
            _range=(1, 372),
            xlabel="Row [px]",
        )
        _CuttingComparisonHistogram(
            axs[0, 1],
            plot,
            self.dataFile.get_cluster_attr("RowWidths", layers=config["layers"])[0],
            self.dataFile.get_cluster_attr("RowWidths", layers=config["layers"], excludeCrossTalk=True)[0],
            bins=372,
            _range=(0, 372),
            xlabel="Row Width [px]",
        )
        plot.fig.suptitle(
            f"{self.dataFile.fileName} removed cross talk comparison layer {config["layers"]}", fontsize="x-large"
        )
        if self.saveToSharedPdf:
            plot.addToPDF(self.pdf)
        else:
            plot.saveToPDF(f"CutComparison{"_"+"".join(str(x) for x in config["layers"]) if config["layers"] is not None else ""}")

def _CuttingComparisonHistogram(ax, plot, values, values_cut, bins=128, _range=(0, 256),xlabel=""):
    height, x = np.histogram(values_cut, bins=bins, range=_range)
    ax.stairs(height, x, color=plot.colorPalette[1], baseline=None, label="CrossTalk Cut")
    height, x = np.histogram(values, bins=bins, range=_range)
    ax.stairs(height, x, color=plot.colorPalette[0], baseline=None, label="Raw")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plot.set_config(
        ax,
        ylim=(0, None),
        xlim=_range,
        prettyTicks=True,
        xlabel=xlabel,
        ylabel="Frequency",
        legend=True,
    )
