import numpy as np
import scipy
from landau import landau
import psutil
import os
import numpy.typing as npt
from typing import Optional, Any, TypeAlias, Union
from clusterClass import clusterClass

clusterArray: TypeAlias = npt.NDArray[np.object_]


def usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / float(2**20)


def print_mem_usage() -> None:
    print(f"\033[93mCurrent Mem Usage:{usage():.2f}Mb\033[0m")


def isFiltered(dataAnalysis: object, filter_dict: dict = {}) -> bool:
    # Takes in an array of data_class and filters and sorts them
    # filter_dict has keys that are attributes of data_class with values that you want to filter for
    # Returns filtered list sorted by angle and then by voltage
    for f in filter_dict.keys():
        attr = getattr(dataAnalysis, "get_" + f)
        if isinstance(f, list):
            if not np.isin(attr, filter_dict[f]):
                return True
        else:
            if not attr == filter_dict[f]:
                return True
    return False


def readFileName(path: str) -> tuple[str, int, float, str]:
    angle_dict = {
        "angle1": 45,
        "angle2": 40.5,
        "angle3": 28,
        "angle4": 20.5,
        "angle5": 11,
        "angle6": 86.5,
    }
    voltage_dict = {
        "V48": 48,
        "V30": 30,
        "V20": 20,
        "V15": 15,
        "V10": 10,
        "V8": 8,
        "V6": 6,
        "V4": 4,
        "V2": 2,
        "V0": 0,
    }
    file_name = "_".join(path.split("/")[-1].split(".")[0].split("_")[3:]).removesuffix("_decode")
    angle: float = 0
    for k in angle_dict.keys():
        if k in file_name:
            angle = angle_dict[k]
            break
    voltage = -1
    for k in voltage_dict.keys():
        if k in file_name:
            voltage = voltage_dict[k]
            break

    if "kit" in path:
        telescope = "kit"
    elif "lancs" in path:
        telescope = "lancs"
    if voltage == -1:
        voltage = 48
    return file_name, voltage, angle, telescope


def calcToT(TS: npt.NDArray[np.int_], TS2: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    return (TS2 * 2 - TS) % 256


def calcHit_Voltage(
    rows: npt.NDArray[np.int_],
    columns: npt.NDArray[np.int_],
    ToTs: npt.NDArray[np.int_],
    Layers: npt.NDArray[np.int_],
) -> npt.NDArray[np.float_]:
    hit_voltage = np.empty(len(rows), dtype=float)
    for k in range(1, 5):
        calibration_array = np.load(f"/home/atlas/rballard/Code/tot_calibration/data_{k}.npy")
        calibration_array_indexes = calibration_array[columns[Layers == k], rows[Layers == k]]
        u_0 = calibration_array_indexes[:, 0]
        a = calibration_array_indexes[:, 1]
        b = calibration_array_indexes[:, 2]
        c = calibration_array_indexes[:, 3]
        hit_voltage[Layers == k] = np.real(
            lambert_W_ToT_to_u(ToTs[Layers == k], u_0, a, b, c)
        )
    return hit_voltage


def calcHit_VoltageError(
    rows: npt.NDArray[np.int_],
    columns: npt.NDArray[np.int_],
    ToTs: npt.NDArray[np.int_],
    Layers: npt.NDArray[np.int_],
) -> npt.NDArray[np.float_]:
    hit_voltage = np.empty(len(rows), dtype=float)
    hit_voltageError = np.empty(len(rows), dtype=float)
    for k in range(1, 5):
        calibration_array = np.load(f"/home/atlas/rballard/Code/tot_calibration/data_{k}.npy")
        calibration_array_indexes = calibration_array[columns[Layers == k], rows[Layers == k]]
        u_0 = calibration_array_indexes[:, 0]
        a = calibration_array_indexes[:, 1]
        b = calibration_array_indexes[:, 2]
        c = calibration_array_indexes[:, 3]
        hit_voltage[Layers == k] = np.real(
            lambert_W_ToT_to_u(ToTs[Layers == k], u_0, a, b, c)
        )
        upper = lambert_W_ToT_to_u(ToTs[Layers == k] + 2, u_0, a, b, c)
        lower = lambert_W_ToT_to_u(ToTs[Layers == k] - 2, u_0, a, b, c)
        upperError = upper - hit_voltage[Layers == k]
        lowerError = hit_voltage[Layers == k] - lower
        hit_voltageError[Layers == k] = np.max([upperError, lowerError], axis=0)
    return hit_voltageError


def lambert_W_ToT_to_u(
    ToT: Union[npt.NDArray[np.int_]|npt.NDArray[np.float64]| float| int],
    u_0: Union[npt.NDArray[np.float64]| float],
    a: Union[npt.NDArray[np.float64]| float],
    b: Union[npt.NDArray[np.float64]| float],
    c: Union[npt.NDArray[np.float64]| float],
) -> Union[npt.NDArray[np.float64]| float]:
    u = u_0 + (a / b) * scipy.special.lambertw((b / a) * u_0 * np.exp((ToT - c - (b * u_0)) / a))
    return np.real(u)


def lambert_W_u_to_ToT(
    u: npt.NDArray[np.float64],
    u_0: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    u = np.reshape(u, np.size(u))
    ToT = np.empty(np.shape(u))
    ToT[u > u_0] = (a * np.log((u[u > u_0] - u_0) / u_0)) + (b * u[u > u_0]) + c
    ToT[u <= u_0] = 0
    ToT[ToT < 0] = 0
    return ToT


def calcClusters(
    Layers: npt.NDArray[np.int_],
    TriggerID: npt.NDArray[np.int_],
    TS: npt.NDArray[np.int_],
    time_variance: np.uint16 = np.uint16(80),
    trigger_variance: np.int32 = np.int32(1),
) -> clusterArray:
    clusters = np.empty(len(Layers), dtype=object)
    count = 0
    count_2 = 0
    max_cluster_size = 300
    # Does one layer at a time. Ensures each cluster is on one layer
    for j in range(1, 5):
        diff_TS = np.min(
            [np.diff(TS[Layers == j]), np.diff((TS[Layers == j] + 512) % 1024)], axis=0
        )
        diff_TriggerID = np.diff(TriggerID[Layers == j])
        pixels = np.zeros(max_cluster_size, dtype=int)
        onLayer = np.where(Layers == j)[0]
        for i, index in enumerate(onLayer):
            if index != onLayer[-1]:
                if (
                    # Doesn't account for looping around 1024
                    (abs(diff_TS[i]) <= time_variance)
                    and (abs(diff_TriggerID[i]) <= trigger_variance)
                    and (count_2 < max_cluster_size - 1)
                ):
                    pixels[count_2] = index
                    count_2 += 1
                else:
                    pixels[count_2] = index
                    count_2 += 1
                    clusters[count] = np.array(pixels[:count_2], dtype=int)
                    count += 1
                    pixels[:] = 0
                    count_2 = 0
            else:
                pixels[count_2] = index
                count_2 += 1
                clusters[count] = np.array(pixels[:count_2], dtype=int)
                count += 1
                pixels[:] = 0
                count_2 = 0
    clusters = clusters[:count]
    print(f"{len(clusters)} clusters found")
    return clusters

def checkDirection(values,x,width):
    if width <=6 or len(values) <= 3:
        return True
    gaps = np.where(np.diff(x) > 1)[0]
    #print(f"gaps: {gaps}")
    if gaps.size > 0:
        if np.sum((x[gaps]-np.min(x) > width/2)|(x[gaps+1]-np.min(x) > width/2)) > np.sum((x[gaps]-np.min(x) > width/2)|(x[gaps+1]-np.min(x) > width/2)):
            rightToLeft = False
        elif np.sum((x[gaps]-np.min(x) > width/2)|(x[gaps+1]-np.min(x) > width/2)) < np.sum((x[gaps]-np.min(x) > width/2)|(x[gaps+1]-np.min(x) > width/2)):
            rightToLeft = True
    
    if 'rightToLeft' not in locals():
        try:
            gradient = np.average(np.gradient(values[1:-1],x[1:-1]),weights=np.diff(x[:-1]))/np.average(x[1:-1])*len(x[1:-1])
        except:
            print(width,values,x)
        #print(f"gradient: {gradient}")
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
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
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
            and cluster.getSize(excludeCrossTalk) > width/3
        ):
            Hit_Voltages = np.zeros(width, dtype=float)
            Hit_VoltageErrors = np.zeros(width, dtype=float)
            sort_array = np.argsort(cluster.getRows(excludeCrossTalk))
            # Hit_Voltage = Hit_Voltages[cluster]
            x = cluster.getRows(excludeCrossTalk)[sort_array]
            
            if measuredAttribute == "Hit_Voltage":
                values = cluster.getHit_Voltages(excludeCrossTalk)[sort_array]
                errors = cluster.getHit_VoltageErrors(excludeCrossTalk)[sort_array]
            elif measuredAttribute == "ToT":
                values = cluster.getToTs(excludeCrossTalk)[sort_array]
                errors = cluster.getToTErrors(excludeCrossTalk)[sort_array]
            index = (x - np.min(x)).astype(int)
            #print(f"rightToLeft: {checkDirection(values,x,width)}")
            #print(f"x :{x}")
            #print(f"values: {values}")
            if checkDirection(values,x,width):
                values = np.flip(values)
                errors = np.flip(errors)
                index = np.flip(index)
            #input("---------")
            Hit_Voltages[index] = values
            Hit_VoltageErrors[index] = errors
            # Hit_Voltage = Hit_Voltage[np.argsort(x)]
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
) -> npt.NDArray[np.float64]:
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    y[x < 0.16] = 0
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


def TStoMS(TS: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    return TS * 25 / 1000000


def MStoTS(Time: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    return np.round(Time * 1000000 / 25).astype(np.int_)


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
        # print(len(values),len(errors))
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
        # print(gaussianCDFFunc(binEdges[i+1],values,errors),gaussianCDFFunc(binEdges[i+1],values,errors))
        # histErrors[i] = np.sum([scipy.integrate.quad(gaussianFunc,binEdges[i],binEdges[i+1],args=(value,error)) for value,error in zip(values,errors)])
    histErrors[histErrors < 4] = 4
    if np.max(binEdges) < 100:
        histErrors[binEdges[1:] < 0.3] = histErrors[binEdges[1:] < 0.3]*1*np.exp(-(binEdges[1:][binEdges[1:] < 0.3]-0.16)/0.5)
    else:
        histErrors[binEdges[1:] < 10] = histErrors[binEdges[1:] < 10]*3*np.exp(-binEdges[1:][binEdges[1:] < 10]/15)
    histErrors = histErrors  / (binEdges[1:] - binEdges[:-1])
    return histErrors# * ((binEdges[1:] - binEdges[:-1])/np.min(np.diff(binEdges)))**2


def chargeCollectionEfficiencyFunc(
    depth: npt.NDArray[np.float64],
    V_0: npt.NDArray[np.float64],
    t_epi: npt.NDArray[np.float64],
    edl: npt.NDArray[np.float64],
    base: float = 0,
    GeV:Optional[int]=None
) -> npt.NDArray[np.float64]:
    
    if GeV is not None:
        if GeV == 4:
            base = 0.12
        elif GeV == 6:
            base = 0.09
    voltage = np.zeros(depth.shape)
    voltage[depth < t_epi] = V_0
    voltage[depth >= t_epi] = np.exp(-(depth[depth >= t_epi] - t_epi) / edl) * (V_0-base) + base
    #voltage[voltage < base] = base
    return voltage


def fitVoltageDepth(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], yerr: npt.NDArray[np.float64] , GeV:int=6
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    unzippedBounds = [(0, np.inf), (0, 100), (1, np.inf)]
    lower_bounds, upper_bounds = zip(*unzippedBounds)
    bounds = (list(lower_bounds), list(upper_bounds))
    initial_guess = [0.25, 10, 50]
    cut = x > 0  # (x<70) & (y > 0.17)
    func = lambda depth,V_0,t_epi,edl:chargeCollectionEfficiencyFunc(depth,V_0,t_epi,edl,GeV=GeV)
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


def depletionWidthFunc(V: npt.NDArray[np.float64], a: float, b: float,c:float = 0) -> npt.NDArray[np.float64]:
    return np.sqrt(a * (V + b)) + c


def trueTimeStamps(clusters: clusterArray, ext_TS: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    new_ext_TS = np.zeros(ext_TS.size)
    for cluster in clusters:
        firstTS = np.min(cluster.getTSs(excludeCrossTalk=True))
        firstTS1024 = firstTS % 1024
        new_ext_TS[cluster.getIndexes()] = firstTS + ((cluster.getTSs() % 1024) - firstTS1024)
    return new_ext_TS.astype(np.int_)

