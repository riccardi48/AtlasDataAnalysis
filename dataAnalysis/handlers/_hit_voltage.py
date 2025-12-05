from typing import Union
from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    npt,  # numpy.typing
    np,  # numpy
    lambertw,  # scipy
    tqdm,  # tqdm
)


def _lambert_W_ToT_to_u(
    ToT: Union[npt.NDArray[np.int_] | npt.NDArray[np.float64] | float | int],
    u_0: Union[npt.NDArray[np.float64] | float],
    a: Union[npt.NDArray[np.float64] | float],
    b: Union[npt.NDArray[np.float64] | float],
    c: Union[npt.NDArray[np.float64] | float],
) -> Union[npt.NDArray[np.float64] | float]:
    u = u_0 + (a / b) * np.absolute(lambertw((b / a) * u_0 * np.exp(np.float64(((ToT - c - (b * u_0)) / a)))))
    u[(a==np.nan)|(b == np.nan)|(c == np.nan)] = np.nan
    return u


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


def jacobian_inverse(u, a, b, c, u0=0.161):
    D = a / (u - u0) + b
    J = np.array([-np.log((u - u0) / u0) / D, -u / D, -1.0 / D])  # ∂u/∂a  # ∂u/∂b  # ∂u/∂c
    return J


def calcHit_Voltage(
    rows: npt.NDArray[np.int_],
    columns: npt.NDArray[np.int_],
    ToTs: npt.NDArray[np.int_],
    Layers: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    hit_voltage = np.empty(len(rows), dtype=float)
    for k in range(1, 5):
        calibration_array = tryLoadCalibrationData(
            k
        )  # (132, 372, 8) shape. Corresponding to (column,row,parameter). Parameters in order: u_0,a,b,c,u_0_e,a_e,b_e,c_e
        calibration_array_indexes = calibration_array[
            columns[Layers == k], rows[Layers == k]
        ]  # (num of hits on layer,8) shape. Gives parameter for each hit.
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
    hit_voltage = np.full(len(rows),-1, dtype=float)
    hit_voltageError = np.full(len(rows),-1, dtype=float)
    for k in np.unique(Layers):
        calibration_array = tryLoadCalibrationData(k)
        covArray = np.load(
            f"/home/atlas/rballard/for_magda/data/Calibration_data/calibration/data_covariance_{k}.npy"
        )
        calibration_array_indexes = calibration_array[columns[Layers == k], rows[Layers == k]]
        covArray_indexes = covArray[columns[Layers == k], rows[Layers == k]]
        u_0 = calibration_array_indexes[:, 0]
        a = calibration_array_indexes[:, 1]
        b = calibration_array_indexes[:, 2]
        c = calibration_array_indexes[:, 3]
        hit_voltage[Layers == k] = _lambert_W_ToT_to_u(ToTs[Layers == k], u_0, a, b, c)
        jacobians = jacobian_inverse(hit_voltage[Layers == k], a, b, c, u0=u_0)
        jacobians = np.flip(np.rot90(jacobians, k=3), axis=1)
        hit_voltageError[Layers == k] = [
            np.sqrt(J @ pcov @ J.T) for J, pcov in zip(jacobians, covArray_indexes)
        ]
    hit_voltageError[hit_voltage==-1] = np.nan
    hit_voltage[hit_voltage==-1] = np.nan
    hit_voltageError[hit_voltageError > hit_voltage] = hit_voltage[hit_voltageError > hit_voltage]

    return hit_voltage, hit_voltageError


def tryLoadCalibrationData(k):
    import os

    file = f"/home/atlas/rballard/for_magda/data/Calibration_data/calibration/data_{k}.npy"
    if os.path.isfile(file):
        return np.load(file)
    else:
        calibrate(k)
        return np.load(file)


def calibrate(k):
    import re
    from scipy.optimize import curve_fit
    from dataAnalysis._memCheck import usage

    dataFile = (
        f"/home/atlas/rballard/for_magda/data/Calibration_data/ToT_Calibration_L{5-k}_20220509.dat"
    )
    array = np.zeros((132, 372, 4), dtype=float)
    covArray = np.zeros((132, 372, 3, 3), dtype=float)
    with open(dataFile, "r") as inp:
        inp_string = inp.read()
    sections = np.array(inp_string.split("#"))[2::2]
    sections = np.array([i.split("Error")[-1].replace("\n\n", "")[1:] for i in sections])
    for i in tqdm(range(np.size(sections)), desc=f"Calibrating Layer {k}"):
        section = np.array(re.split("\n|\t", sections[i]))
        section = np.reshape(section, (section.size // 3, 3)).astype(float)
        section[:, 2][section[:, 2] == 0] = 0.001
        u = section[:, 0]
        ToT = section[:, 1]
        error = section[:, 2]
        index = np.invert(((ToT < 5) & (u > 0.3)) | (ToT < -5))
        u = u[index]
        ToT = ToT[index]
        error = error[index]
        bounds = ([0.5, 1, 1], [20, 200, 200])
        u_0 = 0.161
        func = lambda x, a, b, c: _lambert_W_u_to_ToT(x, u_0, a, b, c)
        popt, pcov = curve_fit(
            func,
            u,
            ToT,
            p0=[6, 45, 60],
            sigma=error,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=1000,
        )
        covArray[i % 132, i // 132] = pcov
        (a, b, c) = popt
        array[i % 132, i // 132] = [u_0, a, b, c]
    array[array==0] = np.nan
    covArray[covArray==0] = np.nan
    np.save(f"/home/atlas/rballard/for_magda/data/Calibration_data/calibration/data_{k}.npy", array)
    np.save(
        f"/home/atlas/rballard/for_magda/data/Calibration_data/calibration/data_covariance_{k}.npy",
        covArray,
    )
