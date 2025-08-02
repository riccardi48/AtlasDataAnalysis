from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataAnalysis import *
import numpy as np
import scipy
import numpy.typing as npt
from landau import landau
from typing import Optional, Union, Any

def checkDirection(values, x, width):
    if width <= 6 or len(values) <= 3:
        return True
    gaps = np.where(np.diff(x) > 1)[0]
    if gaps.size > 0:
        if np.sum(
            (x[gaps] - np.min(x) > width / 2) | (x[gaps + 1] - np.min(x) > width / 2)
        ) > np.sum((x[gaps] - np.min(x) > width / 2) | (x[gaps + 1] - np.min(x) > width / 2)):
            rightToLeft = False
        elif np.sum(
            (x[gaps] - np.min(x) > width / 2) | (x[gaps + 1] - np.min(x) > width / 2)
        ) < np.sum((x[gaps] - np.min(x) > width / 2) | (x[gaps + 1] - np.min(x) > width / 2)):
            rightToLeft = True

    if "rightToLeft" not in locals():
        try:
            gradient = (
                np.average(np.gradient(values[1:-1], x[1:-1]), weights=np.diff(x[:-1]))
                / np.average(x[1:-1])
                * len(x[1:-1])
            )
        except:
            print(width, values, x)
        if gradient > 0.001:
            rightToLeft = True
        elif gradient < -0.001:
            rightToLeft = False
        else:
            # Default is true as beam is going right to left
            rightToLeft = True
    return rightToLeft


def calcHit_VoltageByPixel(
    clusters: clusterArray,
    clusterWidths: npt.NDArray[np.int_],
    maxClusterWidth: int = 40,
    excludeCrossTalk: bool = True,
    returnIndexes: bool = False,
    measuredAttribute: str = "Hit_Voltage",
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    Optional[npt.NDArray[np.int_]],
]:
    cluster: clusterClass
    uniqueClusterWidths, unique_counts = np.unique(clusterWidths, return_counts=True)
    hitPositionArray = np.empty(
        (maxClusterWidth, maxClusterWidth, np.max(unique_counts)), dtype=float
    )
    hitPositionErrorArray = np.empty(
        (maxClusterWidth, maxClusterWidth, np.max(unique_counts)), dtype=float
    )
    counts = np.zeros(maxClusterWidth, dtype=int)
    if returnIndexes:
        indexes = np.empty((maxClusterWidth, np.max(unique_counts)), dtype=int)
    for i, cluster in enumerate(clusters):
        width = cluster.getRowWidth(excludeCrossTalk)
        if (
            width > 0
            and width <= maxClusterWidth
            and len(np.unique(cluster.getColumns(excludeCrossTalk))) == 1
            and cluster.getSize(excludeCrossTalk) > width / 3
        ):
            Hit_Voltages = np.zeros(width, dtype=float)
            Hit_VoltageErrors = np.zeros(width, dtype=float)
            sort_array = np.argsort(cluster.getRows(excludeCrossTalk))
            x = cluster.getRows(excludeCrossTalk)[sort_array]

            if measuredAttribute == "Hit_Voltage":
                values = cluster.getHit_Voltages(excludeCrossTalk)[sort_array]
                errors = cluster.getHit_VoltageErrors(excludeCrossTalk)[sort_array]
            elif measuredAttribute == "ToT":
                values = cluster.getToTs(excludeCrossTalk)[sort_array]
                errors = cluster.getToTErrors(excludeCrossTalk)[sort_array]
            index = (x - np.min(x)).astype(int)
            if checkDirection(values, x, width):
                values = np.flip(values)
                errors = np.flip(errors)
                index = np.flip(index)
            # input("---------")
            Hit_Voltages[index] = values
            Hit_VoltageErrors[index] = errors
            if counts[width - 2] < np.max(unique_counts):
                hitPositionArray[width - 2, :width, counts[width - 2]] = Hit_Voltages
                hitPositionErrorArray[width - 2, :width, counts[width - 2]] = Hit_VoltageErrors
                if returnIndexes:
                    indexes[width - 2, counts[width - 2]] = cluster.index
                counts[width - 2] += 1
    if returnIndexes:
        return (
            hitPositionArray[:, :, : np.max(counts) + 2],
            hitPositionErrorArray[:, :, : np.max(counts) + 2],
            counts,
            indexes[:, : np.max(counts) + 2],
        )
    return (
        hitPositionArray[:, :, : np.max(counts) + 2],
        hitPositionErrorArray[:, :, : np.max(counts) + 2],
        counts,
        None,
    )


class _ShapeInfo:
    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf), inclusive=(True, True)):
        self.name = name
        self.integrality = integrality
        self.endpoints = domain
        self.inclusive = inclusive

        domain = list(domain)
        if np.isfinite(domain[0]) and not inclusive[0]:
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and not inclusive[1]:
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain


def landauFunc(
    x: npt.NDArray[np.float64],
    x_mpv: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    scaler: npt.NDArray[np.float64],
    threshold: float = 0.16,
) -> npt.NDArray[np.float64]:
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    y[x < threshold] = 0
    return y


def gaussianFunc(
    x: Union[npt.NDArray[np.float64] | float],
    mu: Union[npt.NDArray[np.float64] | float],
    sig: Union[npt.NDArray[np.float64] | float],
    scaler: Union[npt.NDArray[np.float64] | float],
) -> Union[npt.NDArray[np.float64] | float]:
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    ) * scaler


def gaussianCDFFunc(
    x: Union[npt.NDArray[np.float64] | float],
    mu: Union[npt.NDArray[np.float64] | float],
    sig: Union[npt.NDArray[np.float64] | float],
) -> npt.NDArray[np.float64]:
    y = scipy.stats.norm.cdf(x, mu, sig)
    return np.asarray(y, dtype=np.float64)

def neg_log_likelihood_truncated(
    params: npt.NDArray[np.float64], data: npt.ArrayLike, threshold: float = 0.162
) -> float:
    x_mpv, xi = params
    if xi <= 0:
        return np.inf

    # Normalization constant for Landau above threshold
    Z = 1 - landau.cdf(threshold, x_mpv, xi)
    if Z <= 0 or np.isnan(Z) or np.isinf(Z):
        return np.inf

    # Evaluate truncated PDF for data > threshold
    pdf_vals = landau.pdf(data, x_mpv, xi) / Z
    if np.any(pdf_vals <= 0) or np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)):
        return np.inf

    return -np.sum(np.log(pdf_vals))


def errorFunc_Voltage(
    x: npt.NDArray[np.float64], n: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    error = (np.exp((x - 0.16) / (-0.3)) / 10) ** np.sqrt(n) + 0.1 * np.sqrt(n)
    error[0] = error[0] * 2
    return error


def calcDepth(
    d: float, clusterWidth: int, angle: float, depthCorrection: bool = True, upTwo: bool = False
) -> npt.NDArray[np.float64]:
    # 50 for 5 microns row width
    shift = 0
    if upTwo:
        shift = 2
    x = np.linspace(d * 50, 0, clusterWidth)
    if depthCorrection and clusterWidth + shift > (d * np.tan(np.deg2rad(angle))):
        x = x * ((clusterWidth + shift) / (d * np.tan(np.deg2rad(angle))))
    return x


def adjustPeakVoltage(
    y: npt.NDArray[np.float64], y_err: npt.NDArray[np.float64], d: float, clusterWidth: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    y[1:-1] = y[1:-1] * (1 / np.sqrt((((d * 0.7) ** 2) / ((clusterWidth - 1) ** 2)) + 1))
    y_err[1:-1] = y_err[1:-1] * (1 / np.sqrt((((d * 0.7) ** 2) / ((clusterWidth - 1) ** 2)) + 1))
    return y, y_err


def histogramErrors(
    values: npt.NDArray[np.float64],
    binEdges: npt.NDArray[np.float64],
    errors: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    if errors is None:
        return np.zeros(len(binEdges[1:]), dtype=float)
    histErrors = np.zeros(len(binEdges[1:]))
    errorOut = (
        gaussianCDFFunc(binEdges[0], values, errors) - gaussianCDFFunc(0, values, errors)
    ) ** 2
    histErrors[0] += np.sqrt(np.sum(errorOut))
    for i in range(len(binEdges[1:])):
        inRange = (values > binEdges[i]) & (values < binEdges[i + 1])
        errorOut = (
            gaussianCDFFunc(binEdges[i + 1], values[np.invert(inRange)], errors[np.invert(inRange)])
            - gaussianCDFFunc(binEdges[i], values[np.invert(inRange)], errors[np.invert(inRange)])
        ) ** 2
        errorIn = (
            gaussianCDFFunc(binEdges[i], values[inRange], errors[inRange])
            + (1 - gaussianCDFFunc(binEdges[i + 1], values[inRange], errors[inRange]))
        ) ** 2
        histErrors[i] += np.sqrt(np.sum(np.append(errorOut, errorIn)))
    histErrors[histErrors < 4] = 4
    """
    if np.max(binEdges) < 100:
        histErrors[binEdges[1:] < 0.3] = (
            histErrors[binEdges[1:] < 0.3]
            * 3
            * np.exp(-(binEdges[1:][binEdges[1:] < 0.3] - 0.1) / 0.3)
        )
    else:
        histErrors[binEdges[1:] < 10] = (
            histErrors[binEdges[1:] < 10] * 3 * np.exp(-binEdges[1:][binEdges[1:] < 10] / 15)
        )
    """
    histErrors = histErrors / (binEdges[1:] - binEdges[:-1])
    return histErrors


def chargeCollectionEfficiencyFunc(
    depth: npt.NDArray[np.float64],
    V_0: npt.NDArray[np.float64],
    t_epi: npt.NDArray[np.float64],
    edl: npt.NDArray[np.float64],
    base: float = 0,
    GeV: Optional[int] = None,
) -> npt.NDArray[np.float64]:

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
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    yerr: npt.NDArray[np.float64],
    GeV: int = 6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    unzippedBounds = [(0, np.inf), (0, 100), (1, np.inf)]
    lower_bounds, upper_bounds = zip(*unzippedBounds)
    bounds = (list(lower_bounds), list(upper_bounds))
    initial_guess = [0.25, 10, 50]
    cut = x > 0  # (x<70) & (y > 0.17)
    func = lambda depth, V_0, t_epi, edl: chargeCollectionEfficiencyFunc(
        depth, V_0, t_epi, edl, GeV=GeV
    )
    popt, pcov = scipy.optimize.curve_fit(
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
    V: npt.NDArray[np.float64], a: float, b: float, c: float = 0
) -> npt.NDArray[np.float64]:
    return np.sqrt(a * (V + b)) + c

def linearLine(a,b,x):
    m = 1/(a-b)
    y = m*(x-a)
    return y
def calcDepthsFromTSs(cluster,excludeCrossTalk=True,residual=1/np.sqrt(12)):
    rows = cluster.getRows(excludeCrossTalk)
    sortArray = np.argsort(rows)
    rows = rows[sortArray]
    TSs = cluster.getTSs(excludeCrossTalk)[sortArray]
    if np.ptp(TSs) > 500:
        TSs = (TSs+512)%1024
    relativeTSs = TSs - np.min(TSs)
    highTSs = np.where(relativeTSs>=2)[0]
    #print(highTSs)
    if highTSs.size > 0 and np.average(highTSs) < cluster.getRowWidth(excludeCrossTalk)/2:
        rightToLeft = True
    else:
        rightToLeft = False
    relativeRows = rows-np.min(rows)
    if rightToLeft:
        rows = np.flip(rows)
        relativeTSs = np.flip(relativeTSs)
        highTSs = np.where(relativeTSs>=2)[0]
    if 0 in highTSs:
        minStart = residual
        maxStart = 0.5
        startIndex = 1
    else:
        minStart = -residual
        maxStart = +residual
        startIndex = 0
    try:
        if highTSs.size > 0 and np.all(np.isin(np.arange(highTSs[startIndex],highTSs[-1]+1),highTSs)):
            minStop = relativeRows[highTSs[startIndex]-1]-residual
            maxStop = relativeRows[highTSs[startIndex]-1]+1**(-rightToLeft)-residual
        else:
            minStop = relativeRows[-1]-residual
            maxStop = relativeRows[-1]+1-residual
    except:
        print(highTSs)
        print(startIndex)
        print(rightToLeft)
        print(relativeTSs)
        print(relativeRows)
    minDepth = np.zeros(relativeRows.shape)
    maxDepth = np.zeros(relativeRows.shape)
    minDepth[rows<maxStop] = linearLine(minStart,minStop,relativeRows[rows<maxStop])
    minDepth[rows>=maxStop]  = linearLine(maxStart,minStop,relativeRows[rows>=maxStop])
    maxDepth[rows<maxStop] = linearLine(maxStart,maxStop,relativeRows[rows<maxStop])
    maxDepth[rows>=maxStop]  = linearLine(minStart,maxStop,relativeRows[rows>=maxStop])
    depth = np.average([minDepth,maxDepth],axis=0)
    error = np.ptp([minDepth,maxDepth],axis=0)/2
    if rightToLeft:
        depth = np.flip(depth)
        error = np.flip(error)
    depth = depth[np.argsort(sortArray)]
    error = error[np.argsort(sortArray)]
    cluster.depth = depth
    cluster.depthError = error
