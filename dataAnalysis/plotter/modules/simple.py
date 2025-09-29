from dataAnalysis._types import dataAnalysis
from typing import Optional, Any, Union
from ._genericModule import plotModule
from ._plot import plotClass
from dataAnalysis._dependencies import (
    np,
)
class simpleModule(plotModule):
    def rowHistogram(self, config: Optional[dict] = None,excludeCrossTalk:bool=False):
        config = self.configCheck(config)
        return self._genericAttributeHistogram(config, attribute="Row", excludeCrossTalk=excludeCrossTalk, bins=371, _range=(1, 372))
    
    def columnHistogram(self, config: Optional[dict] = None,excludeCrossTalk:bool=False):
        config = self.configCheck(config)
        return self._genericAttributeHistogram(config, attribute="Column", excludeCrossTalk=excludeCrossTalk, bins=49, _range=(1, 50))

    def ToTHistogram(self, config: Optional[dict] = None,excludeCrossTalk:bool=False):
        config = self.configCheck(config)
        return self._genericAttributeHistogram(config, attribute="ToT", excludeCrossTalk=excludeCrossTalk, bins=128, _range=(0, 256))

    def _genericAttributeHistogram(self, config: Optional[dict] = None, attribute: str = "ToT",excludeCrossTalk:bool=False, bins:int=128, _range:tuple[int,int]=(0, 256)):
        config = self.configCheck(config)
        plot = plotClass(self.pathToOutput, shape=(1, 1), sizePerPlot=(7, 7))
        axs = plot.axs
        _attributeHistogram(axs, plot, self.dataFile.get_base_attr(attribute, layers=config["layers"], excludeCrossTalk=excludeCrossTalk)[0], bins=bins, _range=_range,xlabel="")
        plot.fig.suptitle(
            f"{self.dataFile.fileName} removed cross talk comparison", fontsize="x-large"
        )
        if self.saveToSharedPdf:
            plot.addToPDF(self.pdf)
        else:
            plot.saveToPDF(f"CutComparison{"_"+"".join(str(x) for x in config["layers"]) if config["layers"] is not None else ""}")

def _attributeHistogram(ax, plot, values, bins=128, _range=(0, 256),xlabel=""):
    height, x = np.histogram(values, bins=bins, range=_range)
    ax.stairs(height, x, color=plot.colorPalette[1], baseline=None)
    plot.set_config(
        ax,
        xlabel=xlabel,
        ylabel="Counts",
        legend=False,
    )