from typing import Union
from dataAnalysis._types import clusterClass,dataAnalysis,clusterArray
from dataAnalysis._dependencies import (
    npt,                # numpy.typing
    np,                 # numpy
    lambertw,           # scipy
)

def _lambert_W_ToT_to_u(
    ToT: Union[npt.NDArray[np.int_] | npt.NDArray[np.float64] | float | int],
    u_0: Union[npt.NDArray[np.float64] | float],
    a: Union[npt.NDArray[np.float64] | float],
    b: Union[npt.NDArray[np.float64] | float],
    c: Union[npt.NDArray[np.float64] | float],
) -> Union[npt.NDArray[np.float64] | float]:
    u = u_0 + (a / b) * lambertw((b / a) * u_0 * np.exp((ToT - c - (b * u_0)) / a))
    return np.real(u)


def _lambert_W_u_to_ToT(
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

def calcHit_Voltage(
    rows: npt.NDArray[np.int_],
    columns: npt.NDArray[np.int_],
    ToTs: npt.NDArray[np.int_],
    Layers: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    hit_voltage = np.empty(len(rows), dtype=float)
    for k in range(1, 5):
        calibration_array = np.load(f"/home/atlas/rballard/Code/tot_calibration/data_{5-k}.npy")
        calibration_array_indexes = calibration_array[columns[Layers == k], rows[Layers == k]]
        u_0 = calibration_array_indexes[:, 0]
        a = calibration_array_indexes[:, 1]
        b = calibration_array_indexes[:, 2]
        c = calibration_array_indexes[:, 3]
        hit_voltage[Layers == k] = np.real(_lambert_W_ToT_to_u(ToTs[Layers == k], u_0, a, b, c))
    return hit_voltage

def calcHit_VoltageError(
    rows: npt.NDArray[np.int_],
    columns: npt.NDArray[np.int_],
    ToTs: npt.NDArray[np.int_],
    Layers: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    hit_voltage = np.empty(len(rows), dtype=float)
    hit_voltageError = np.empty(len(rows), dtype=float)
    for k in range(1, 5):
        calibration_array = np.load(f"/home/atlas/rballard/Code/tot_calibration/data_{5-k}.npy")
        calibration_array_indexes = calibration_array[columns[Layers == k], rows[Layers == k]]
        u_0 = calibration_array_indexes[:, 0]
        a = calibration_array_indexes[:, 1]
        b = calibration_array_indexes[:, 2]
        c = calibration_array_indexes[:, 3]
        hit_voltage[Layers == k] = np.real(_lambert_W_ToT_to_u(ToTs[Layers == k], u_0, a, b, c))
        upper = _lambert_W_ToT_to_u(ToTs[Layers == k] + 2, u_0, a, b, c)
        lower = _lambert_W_ToT_to_u(ToTs[Layers == k] - 2, u_0, a, b, c)
        upperError = upper - hit_voltage[Layers == k]
        lowerError = hit_voltage[Layers == k] - lower
        hit_voltageError[Layers == k] = np.max([upperError, lowerError], axis=0)
    return hit_voltageError