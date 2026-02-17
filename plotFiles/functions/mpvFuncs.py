import sys
from .efficiencyFuncs import calcEfficiency,getPercentFromDict
from .genericFuncs import convertToRelative
sys.path.append("/home/atlas/rballard/AtlasDataAnalysis")
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit,brentq
from landau import landau
from tqdm import tqdm
from dataAnalysis._fileReader import calcDataFileManager

def getPList(clusters,maxWidth=30):
    efficiencyDict = calcEfficiency(clusters,maxWidth=maxWidth)
    pList,errors = getPercentFromDict(efficiencyDict)
    return pList,errors


def generateHistLists(clusters, maxWidth = 30):
    histLists = [[] for _ in range(maxWidth)]
    histErrorsLists = [[] for _ in range(maxWidth)]
    l = 0
    for cluster in clusters:
        rows = cluster.getRows(True)[cluster.section]
        voltage = cluster.getHit_Voltages(True)[cluster.section]
        voltageErrors = cluster.getHit_VoltageErrors(True)[cluster.section]
        # relativeRows = abs(cluster.getRows(True)[cluster.section] - np.max(cluster.getRows(True)[cluster.section]))
        relativeRows, voltage = convertToRelative(rows, voltage, flipped=cluster.flipped)
        _, voltageErrors = convertToRelative(rows, voltageErrors, flipped=cluster.flipped)
        # if np.ptp(relativeRows) < possibleRows-2:
        #    continue

        for i in range(relativeRows.size):
            if relativeRows[i] >= maxWidth:
                continue
            if (
                not np.isnan(voltage[i])
                and not np.isinf(voltage[i])
                and not voltage[i] == 0
                and not voltageErrors[i] == 0
                and not np.isnan(voltageErrors[i])
                and not np.isinf(voltageErrors[i])
            ):
                histLists[relativeRows[i]].append(voltage[i])
                histErrorsLists[relativeRows[i]].append(voltageErrors[i])
            else:
                # print(i, voltage[i], voltageErrors[i])
                l += 1
    print(f"Skipped {l} hits due to invalid voltage")
    return histLists,histErrorsLists

class mpvData:
    def __init__(self,dataFile):
        self._range=(0.160, 2)
        self.maxWidth = 30
        self.minPval = 0.5
        self.layer = 4
        self.dataFile = dataFile
        self.fittings = {}
        self.constrainedFittings = {}
    def initFittings(self,config):
        calcFileManager = calcDataFileManager(config["pathToCalcData"], self.dataFile.fileName, config["maxLine"])
        calcFileName = calcFileManager.generateFileName(
            attribute=f"landauFittings",
        )
        fileCheck = calcFileManager.fileExists(calcFileName=calcFileName)
        if not fileCheck:
            self.dataFile.get_crossTalk()
            self.dataFile.init_cluster_voltages()
            self.calcFittings()
            calcFileManager.saveFile(calcFileName=calcFileName,array=np.array([self.fittings,self.constrainedFittings,self.pList,self.histLists,self.histErrorsLists],dtype=object))
        else:
            self.fittings,self.constrainedFittings,self.pList,self.histLists,self.histErrorsLists = calcFileManager.loadFile(calcFileName=calcFileName)

    def calcFittings(self):
        clusters = self.dataFile.get_perfectClusters(minPval=self.minPval, layer=self.layer, maxRow=25)
        self.histLists,self.histErrorsLists = generateHistLists(clusters, maxWidth = self.maxWidth)
        self.pList,_ = getPList(clusters,maxWidth=self.maxWidth)
        for i in tqdm(range(self.maxWidth), desc="Calculating Landau fittings"):
            values = np.array(self.histLists[i])
            valuesErrors = np.array(self.histErrorsLists[i])
            p = self.pList[i]
            if p <= 0.05 or len(values) < 100:
                continue
            x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = getBestFitting(
                values, valuesErrors, self._range[0], 1
            )
            self.fittings[i] = [x_mpv, xi, scale, x_mpv_e, xi_e, scale_e]
            if p <= 0.99:
                x_mpv, xi, scale, x_mpv_e, xi_e, scale_e = getBestFitting(
                    values, valuesErrors, self._range[0], p
                )
                self.constrainedFittings[i] = [x_mpv, xi, scale, x_mpv_e, xi_e, scale_e]
            else:
                self.constrainedFittings[i] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    def calcExpectedDepth(self):
        MPVs = np.zeros(self.maxWidth)
        for i in range(self.maxWidth-1):
            MPVs[i] = self.fittings[i][0] if i in self.fittings else np.nan
            if i in self.constrainedFittings:
                if not np.isnan(self.constrainedFittings[i][0]):
                    MPVs[i] = self.constrainedFittings[i][0]
        print(MPVs)
        totalVoltage = np.sum(MPVs) - MPVs[0]/2
        averageMaxVoltage = np.average(MPVs[2:8])
        N = (49.5/50 * np.tan(np.deg2rad(86.5)))
        print(averageMaxVoltage*N)
        print(totalVoltage)
        print(totalVoltage/averageMaxVoltage)
        print((totalVoltage/averageMaxVoltage)*50/np.tan(np.deg2rad(86.5)))





def getBestFitting(values, valuesErrors, x0, p):
    x_mpvList = []
    xiList = []
    scaleList = []
    x_mpv_eList = []
    xi_eList = []
    scale_eList = []
    for points_per_bin in [100, 200]:
        for _range in ((0.160, 2.32), (0.160, 2), (0.160, 1.732)):
            for min_bin_width in [
                (_range[1] - _range[0]) / 80,
                (_range[1] - _range[0]) / 120,
            ]:
                hist, binEdges, binCentres = histogramHit_Voltage_Errors(
                    values,
                    valuesErrors,
                    _range=_range,
                    points_per_bin=points_per_bin,
                    min_bin_width=min_bin_width,
                )
                if p == 1:
                    func = lambda x, x_mpv, x_xi, scaler: landauBinned(
                        x, x_mpv, x_xi, scaler, binEdges
                    )
                    p0 = [0.4, 0.1, np.sum(hist) * 2]
                    popt, pcov = curve_fit(
                        func,
                        binCentres,
                        hist,
                        maxfev=1200,
                        p0=p0,
                    )
                    x_mpv, xi, scale = popt
                    x_mpv_e, xi_e, scale_e = np.sqrt(np.diag(pcov))
                else:
                    if 1 - p <= landauCDFFunc(x0, x0, 1):
                        bounds = ([x0*1.001, 0], [0.6, np.inf])
                        p0 = [x0 * 2, np.sum(hist) * 2]
                    else:
                        bounds = ([0, 0], [x0*0.999, np.inf])
                        p0 = [x0 / 2, np.sum(hist) * 2]
                    lower = 1e-12
                    upper = 100000
                    find_sigma = lambda mpv: brentq(
                        lambda s: landauCDFFunc(x0, mpv, s) - (1 - p),
                        lower,  # lower bound for sigma
                        upper,  # upper bound
                    )
                    func = lambda x, x_mpv, scaler: landauBinned(
                        x, x_mpv, find_sigma(x_mpv), scaler, binEdges
                    )
                    popt, pcov = curve_fit(
                        func,
                        binCentres,
                        hist,
                        maxfev=1200,
                        bounds=bounds,
                        p0=p0,
                    )
                    x_mpv, scale = popt
                    x_mpv_e, scale_e = np.sqrt(np.diag(pcov))
                    xi = find_sigma(x_mpv)
                    xi_e = (
                        abs(find_sigma(x_mpv + x_mpv_e) - xi)
                        + abs(find_sigma(x_mpv - x_mpv_e) - xi)
                    ) / 2
                x_mpvList.append(x_mpv)
                xiList.append(xi)
                scaleList.append(scale)
                x_mpv_eList.append(x_mpv_e)
                xi_eList.append(xi_e)
                scale_eList.append(scale_e)

    e_stat = np.sqrt(np.mean(np.array(x_mpv_eList) ** 2))
    e_bin = np.std(np.array(x_mpvList))
    x_mpv = np.mean(x_mpvList)
    x_mpv_e = np.sqrt(e_stat**2 + e_bin**2)
    e_stat = np.sqrt(np.mean(np.array(xi_eList) ** 2))
    e_bin = np.std(np.array(xiList))
    xi = np.mean(xiList)
    xi_e = np.sqrt(e_stat**2 + e_bin**2)
    e_stat = np.sqrt(np.mean(np.array(scale_eList) ** 2))
    e_bin = np.std(np.array(scaleList))
    scale = np.mean(scaleList)
    scale_e = np.sqrt(e_stat**2 + e_bin**2)
    return x_mpv, xi, scale, x_mpv_e, xi_e, scale_e

def histogramHit_Voltage_Errors(
    values,
    valuesErrors,
    _range=(0.162, 2),
    points_per_bin=100,
    min_bin_width=None,
    max_bin_width=None,
):
    valuesErrors = valuesErrors[(values > _range[0]) & (values < _range[1])]
    values = values[(values > _range[0]) & (values < _range[1])]
    sortIndex = np.argsort(values)
    valuesErrors = valuesErrors[sortIndex]
    values = values[sortIndex]
    if min_bin_width is None:
        min_bin_width = (values.max() - values.min()) / 80
    if max_bin_width is None:
        max_bin_width = (values.max() - values.min()) / 10

    n = len(values)
    n_bins = n // points_per_bin

    if n_bins < 1:
        points_per_bin = 10
        n_bins = n // points_per_bin
        if n_bins < 1:
            points_per_bin = 1
            n_bins = n // points_per_bin
            if n_bins < 1:
                raise ValueError("Too few data points for the desired points per bin.")

    edges = np.interp(np.linspace(0, n, n_bins + 1), np.arange(n), values)

    new_edges = [edges[0]]
    for i in range(1, len(edges)):
        proposed_edge = edges[i]
        last_edge = new_edges[-1]
        width = proposed_edge - last_edge

        if width < min_bin_width:
            continue
        elif width > max_bin_width:
            num_subbins = int(np.ceil(width / max_bin_width))
            sub_edges = np.linspace(last_edge, proposed_edge, num_subbins + 1)[1:]
            new_edges.extend(sub_edges)
        else:
            new_edges.append(proposed_edge)
    binEdges = np.array(new_edges)
    #hist, binEdges = np.histogram(values, bins=binEdges)
    hist = np.sum(
            gaussianBinned(
                np.tile([values], (binEdges.size - 1, 1)),
                np.tile([valuesErrors], (binEdges.size - 1, 1)),
                1,
                np.tile(binEdges[:, np.newaxis], (1, values.size)),
            ),
            axis=1,
        )
    binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    return hist, binEdges, binCentres



def gaussianCDFFunc(x, mu, sig):
    return norm.cdf((x - mu) / sig)


def gaussianBinned(mu, sigma, scaler, edges):
    return (gaussianCDFFunc(edges[1:], mu, sigma) - gaussianCDFFunc(edges[:-1], mu, sigma)) * scaler

def gaussianPDFFunc(x, mu, sig, scaler):
    return norm.pdf((x - mu) / sig) * scaler

def landauBinned(x, x_mpv, xi, scaler, edges):
    return (landauCDFFunc(edges[1:], x_mpv, xi) - landauCDFFunc(edges[:-1], x_mpv, xi)) * scaler

def landauCDFFunc(
    x,
    x_mpv,
    xi,
    # threshold: float = 0.16,
):
    x0 = convert_x_mpv_to_x0(x_mpv, xi)
    # y = stats.landau.cdf((x - x0) / xi)
    y = landau.cdf(x, x_mpv, xi)
    # y = np.reshape(y, np.size(y))
    return y

def landauFunc(
    x,
    x_mpv,
    xi,
    scaler,
    # threshold: float = 0.16,
):
    # x0 = convert_x_mpv_to_x0(x_mpv, xi)
    # y = stats.landau.pdf((x - x0) / xi) * scaler
    y = landau.pdf(x, x_mpv, xi) * scaler
    # y = np.reshape(y, np.size(y))
    return y

def convert_x_mpv_to_x0(x_mpv, xi):
    return x_mpv + 0.22278298 * xi

def chargeCollectionEfficiencyFunc(
    depth,
    V_0,
    t_epi,
    edl,
    base = 0,
    GeV = None,
):

    if GeV is not None:
        if GeV == 4:
            base = 0.00
        elif GeV == 6:
            base = 0.00
    depth = np.reshape(depth, np.size(depth))
    voltage = np.zeros(depth.shape)
    voltage[depth < t_epi] = V_0
    voltage[depth >= t_epi] = np.exp(-(depth[depth >= t_epi] - t_epi) / edl) * (V_0 - base) + base
    return voltage

def fitVoltageDepth(
    x,
    y,
    yerr,
    GeV = 6,
):
    unzippedBounds = [(0, np.inf), (0, 100), (0, np.inf)]
    lower_bounds, upper_bounds = zip(*unzippedBounds)
    bounds = (list(lower_bounds), list(upper_bounds))
    initial_guess = [0.5, 30, 10]
    cut = x > 0  # (x<70) & (y > 0.17)
    func = lambda depth, V_0, t_epi, edl: chargeCollectionEfficiencyFunc(
        depth, V_0, t_epi, edl, GeV=GeV
    )
    popt, pcov = curve_fit(
        func,
        x[cut],
        y[cut],
        p0=initial_guess,
        maxfev=10000000,
        bounds=bounds,
        sigma=yerr[cut] / y[cut],
        absolute_sigma=False,
    )
    return popt, pcov

def depletionWidthFunc(
    V, a, b, c
):
    return np.sqrt(a * (V + b)) + c
