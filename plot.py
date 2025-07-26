from plotConfigs import AngleDistribution,WidthDistribution,AngleDistribution_2,ColumnWidthDistribution,RowWidthDistribution,VoltageDepthScatter,HitDistributionInClusterAllOnOne,CuttingComparison,RowRowCorrelation
from dataAnalysis import initDataFiles
from plotAnalysis import depthAnalysis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    import configLoader
    config = configLoader.loadConfig()
    dataFiles = initDataFiles(config)
    for dataFile in dataFiles:
        with PdfPages(f"{config["pathToOutput"]}/Combined/{dataFile.get_fileName()}.pdf") as pdf:
            depth = depthAnalysis(config["pathToCalcData"], maxLine=config["maxLine"], maxClusterWidth=config["maxClusterWidth"], layers=config["layers"], excludeCrossTalk=config["excludeCrossTalk"])
            pdf.savefig(AngleDistribution(dataFile, depth, config["pathToOutput"], saveToPDF=False))
            pdf.savefig(AngleDistribution_2(dataFile, depth, config["pathToOutput"], saveToPDF=False))
            pdf.savefig(WidthDistribution(dataFile, depth, config["pathToOutput"], saveToPDF=False))
            for layer in config["layers"]:
                pdf.savefig(ColumnWidthDistribution(dataFile, config["pathToOutput"], layer=layer, saveToPDF=False))
                pdf.savefig(RowWidthDistribution(dataFile, config["pathToOutput"], layer=layer, saveToPDF=False))
                pdf.savefig(CuttingComparison(dataFile, config["pathToOutput"], layer=layer, saveToPDF=False))
            pdf.savefig(HitDistributionInClusterAllOnOne(dataFile, depth, config["pathToOutput"], vmin=2, vmax=config["maxClusterWidth"], saveToPDF=False))
            pdf.savefig(HitDistributionInClusterAllOnOne(dataFile, depth, config["pathToOutput"], vmin=2, vmax=12, saveToPDF=False))
            pdf.savefig(HitDistributionInClusterAllOnOne(dataFile, depth, config["pathToOutput"], vmin=13, vmax=config["maxClusterWidth"], saveToPDF=False))
            pdf.savefig(RowRowCorrelation(dataFile, config["pathToOutput"], config["pathToCalcData"], layers=config["layers"], excludeCrossTalk=False, saveToPDF=False))
            pdf.savefig(RowRowCorrelation(dataFile, config["pathToOutput"], config["pathToCalcData"], layers=config["layers"], excludeCrossTalk=True, saveToPDF=False))
            pdf.savefig(VoltageDepthScatter(dataFile,depth,config["pathToOutput"],hideLowWidths=False,measuredAttribute="Hit_Voltage", saveToPDF=False))
            pdf.savefig(VoltageDepthScatter(dataFile,depth,config["pathToOutput"],hideLowWidths=False,measuredAttribute="ToT", saveToPDF=False))
            pdf.savefig(VoltageDepthScatter(dataFile,depth,config["pathToOutput"],measuredAttribute="Hit_Voltage", saveToPDF=False))
            pdf.savefig(VoltageDepthScatter(dataFile,depth,config["pathToOutput"],measuredAttribute="ToT", saveToPDF=False))
            pdf.savefig()