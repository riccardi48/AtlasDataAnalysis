import sys

sys.path.append("..")
import numpy as np
from dataAnalysis import initDataFiles, configLoader
from plotFiles.plotClass import plotGenerator
from dataAnalysis.handlers._perfectCluster import (
    convertRowsForFit,
    pValOfSection,
    gaussian_loglike_pval,
    filterForTemplate,
    findBestSections,
    findConnectedSections,
)
from dataAnalysis.handlers._genericClusterFuncs import logGaussian, logGaussianCDFFunc
from dataAnalysis.handlers._clusterClass import clusterClass
from itertools import combinations
from matplotlib.ticker import MultipleLocator
from scipy.optimize import root_scalar
from tqdm import tqdm


def findLogPercentile(mu, sig, percentile):
    func = lambda x: logGaussianCDFFunc(np.array(x) - 0.5, mu, sig) - percentile
    return root_scalar(func, bracket=[0, 1000]).root - 1


def findMPV(estimate, spread):
    x = np.linspace(0.5, 25, 1000)
    y = logGaussian(x, estimate, spread, 1)
    return x[np.argmax(y)] - 0.5


def graphTSonRows(
    relativeRows, relativeTS, estimate, spread, path, excludeCrossTalk=True, label="", name=""
):
    plot = plotGen.newPlot(path)
    axs = plot.axs
    y, _estimate, _spread = filterForTemplate(relativeRows, relativeTS, estimate, spread)
    index = _estimate != 0
    mpvs = np.array([findMPV(mu, sig) for mu, sig in zip(_estimate[index], _spread[index])])
    median = np.array(
        [findLogPercentile(mu, sig, 0.5) for mu, sig in zip(_estimate[index], _spread[index])]
    )
    upper = np.array(
        [
            findLogPercentile(mu, sig, 0.5 + 0.34)
            for mu, sig in zip(_estimate[index], _spread[index])
        ]
    )
    lower = np.array(
        [
            findLogPercentile(mu, sig, 0.5 - 0.34)
            for mu, sig in zip(_estimate[index], _spread[index])
        ]
    )
    axs.scatter(relativeRows, relativeTS, color=plot.colorPalette[2], marker="x", label=f"{label}")
    # axs.scatter(x[mask], relativeTS[mask], color=plot.colorPalette[8], marker="x",label="Not in Fitted Sections")
    if len(relativeTS) != 0:
        # axs.plot(relativeRows[relativeRows<len(estimate)][index],mpvs,color=plot.colorPalette[0],linestyle="dashed",label=f"{label}")
        # axs.fill_between(relativeRows[relativeRows<len(estimate)][index],_estimate[index]-_spread[index],_estimate[index]+_spread[index], alpha=0.2,color=plot.colorPalette[0])
        plot.axs.scatter(
            relativeRows[relativeRows < len(estimate)][index],
            median - 0.5,
            marker="x",
            color=plot.colorPalette[0],
            label=f"Median with 1 std",
        )
        plot.axs.errorbar(
            relativeRows[relativeRows < len(estimate)][index],
            median - 0.5,
            yerr=[median - lower, upper - median],
            fmt="none",
            color=plot.colorPalette[0],
            elinewidth=1,
            capsize=3,
        )

    plot.set_config(
        axs,
        title="Relative TS in cluster",
        xlabel="Relative Row",
        ylabel="Relative TS",
        legend=True,
        ylim=(-0.5, np.max(relativeTS) + 5),
        xlim=(np.min(relativeRows) - 1, np.max(relativeRows) + 1),
    )
    axs.xaxis.set_major_locator(MultipleLocator(5))
    axs.xaxis.set_major_formatter("{x:.0f}")
    axs.xaxis.set_minor_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(5))
    axs.yaxis.set_major_formatter("{x:.0f}")
    axs.yaxis.set_minor_locator(MultipleLocator(1))
    plot.saveToPDF(f"{name}")


class clusterTestClass:
    def __init__(self, cluster, estimate, spread, excludeEnds=True):
        self.cluster = cluster
        self.estimate = estimate
        self.spread = spread
        self.excludeEnds = excludeEnds
        Timestamps = cluster.getTSs(True)
        Rows = cluster.getRows(True)
        self.relativeRows, self.relativeTimeStamps = convertRowsForFit(
            Rows, Timestamps, cluster.flipped
        )
        self.sortIndex = np.argsort(Rows)
        self.sortIndex = self.sortIndex[np.isin(np.arange(len(Rows)),cluster.section)[self.sortIndex]]
        self.relativeRows = self.relativeRows[cluster.section]
        self.relativeTimeStamps = self.relativeTimeStamps[cluster.section]
        self.data = {}
        self.fullSection = np.arange(len(self.relativeTimeStamps))
        self.cutSection = self.fullSection
        if self.excludeEnds:
            self.cutSection = self.fullSection[1:-1]

    def testOnce(self, n, makePlot=False):
        for i, perm in tqdm(
            enumerate(combinations(self.cutSection, len(self.cutSection) - n)), desc=f"n={n}"
        ):
            if self.excludeEnds:
                perm = [0] + list(perm) + [self.fullSection[-1]]
            perm = np.array(perm)
            """
            tempCluster = clusterClass(
                cluster.getIndex(),
                cluster.getIndexes(True)[self.sortIndex[perm]],
                cluster.getLayer(),
                cluster.getColumns(True)[self.sortIndex[perm]],
                cluster.getRows(True)[self.sortIndex[perm]],
                cluster.getToTs(True)[self.sortIndex[perm]],
                cluster.getEXT_TSs(True)[self.sortIndex[perm]],
            )
            tempCluster.setCrossTalk(np.zeros(tempCluster.indexes.size, dtype=bool))
            sections = findConnectedSections(tempCluster.getRows(True), tempCluster.getColumns(True))
            _pVal, _flipped, _perm = findBestSections(tempCluster, sections,estimate,spread)
            #print(_pVal, _flipped, _perm)
            """
            pVal, flipped = self.pValCluster(perm, name = f"TS_Rows_{n}_{i}")
            self.addData(n, float(pVal))
            if makePlot:
                graphTSonRows(
                    self.relativeRows[perm],
                    self.relativeTimeStamps[perm],
                    estimate,
                    spread,
                    f"Cluster_{cluster.getIndex()}/",
                    excludeCrossTalk=True,
                    label=f"p-Value = {pVal:.4f}",
                    name=f"TS_Rows_{n}_{i}",
                )

    def pValCluster(self,section, makePlot=False, name=""):
        pVal, flipped = pValOfSection(
                self.cluster, self.sortIndex[section], self.estimate, self.spread, True
            )
        if makePlot:
            graphTSonRows(
                self.relativeRows[section],
                self.relativeTimeStamps[section],
                estimate,
                spread,
                f"Cluster_{cluster.getIndex()}/",
                excludeCrossTalk=True,
                label=f"p-Value = {pVal:.4f}",
                name=f"{name}",
            )
        return pVal, flipped
    def addData(self, n, value):
        if n not in self.data:
            self.data[n] = [value]
        else:
            self.data[n].append(value)


def plotTestData(data, plotGen, name=""):
    plot = plotGen.newPlot(f"Cluster_{cluster.getIndex()}/")
    x = data.keys()
    y_min = []
    y_max = []
    y_mean = []
    y_med = []
    y_std = []
    for values in data.values():
        y_min.append(np.min(values))
        y_max.append(np.max(values))
        y_mean.append(np.mean(values))
        y_med.append(np.median(values))
        y_std.append(np.std(values))
    y_std = np.array(y_std)
    y_mean = np.array(y_mean)
    plot.axs.plot(x, y_med, color=plot.colorPalette[0], linestyle="dashed", label=f"Median")
    plot.axs.fill_between(
        x, y_min, y_max, color=plot.colorPalette[0], alpha=0.2, linestyle="dashed", label=f"Range"
    )
    plot.axs.plot(x, y_mean, color=plot.colorPalette[2], linestyle="dashed", label=f"Mean")
    plot.axs.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        color=plot.colorPalette[2],
        alpha=0.4,
        linestyle="dashed",
        label=f"Std",
    )

    plot.set_config(
        plot.axs,
        title="Removal of pixels effect on data",
        xlabel="Number of pixels removed",
        ylabel="p value",
        legend=True,
        ylim=(0, 1),
    )
    plot.axs.xaxis.set_major_locator(MultipleLocator(1))
    plot.axs.xaxis.set_major_formatter("{x:.0f}")
    plot.axs.xaxis.set_minor_locator(MultipleLocator(1))
    plot.axs.yaxis.set_major_locator(MultipleLocator(0.1))
    plot.axs.yaxis.set_major_formatter("{x:.1f}")
    plot.axs.yaxis.set_minor_locator(MultipleLocator(0.02))
    plot.saveToPDF(f"PixelRemovalTest{name}")


config = configLoader.loadConfig()
config["filterDict"] = {"telescope": "kit", "angle": 86.5, "voltage": 48.6}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    base_path = f"{config["pathToOutput"]}ClusterFitting/{dataFile.fileName}/"
    plotGen = plotGenerator(base_path)
    clusters = dataFile.get_perfectClusters(excludeCrossTalk=True, layer=4)
    estimate, spread = dataFile.get_timeStampTemplate(excludeCrossTalk=True, layer=4)
    for cluster in clusters[200:]:
        Timestamps = cluster.getTSs(True)
        Rows = cluster.getRows(True)
        relativeRows, relativeTimeStamps = convertRowsForFit(Rows, Timestamps, cluster.flipped)
        print(relativeRows, relativeTimeStamps)
        print(cluster.section)
        pVal, flipped = pValOfSection(cluster, cluster.section, estimate, spread, True)
        graphTSonRows(
            relativeRows[cluster.section],
            relativeTimeStamps[cluster.section],
            estimate,
            spread,
            f"Cluster_{cluster.getIndex()}/",
            excludeCrossTalk=True,
            label=f"p-Value = {pVal:.4f}",
            name=f"TS_Rows",
        )
        test = clusterTestClass(cluster, estimate, spread, True)
        test.testOnce(1, makePlot=True)
        test = clusterTestClass(cluster, estimate, spread, True)
        for i in range(7):
            test.testOnce(i, makePlot=False)
        plotTestData(test.data, plotGen, name="_ExcludeEdges")
        test = clusterTestClass(cluster, estimate, spread, False)
        for i in range(7):
            test.testOnce(i, makePlot=False)
        plotTestData(test.data, plotGen, name="")

        #input()
