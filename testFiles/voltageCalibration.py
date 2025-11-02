import sys

sys.path.append("..")
from typing import Union
from dataAnalysis._dependencies import (
    npt,  # numpy.typing
    np,  # numpy
    lambertw,  # scipy
)
from plotAnalysis import plotClass
from dataAnalysis._memCheck import usage

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

def calibrate(k):
    import re
    from scipy.optimize import curve_fit
    dataFile = f"/home/atlas/rballard/for_magda/data/Calibration_data/ToT_Calibration_L{5-k}_20220509.dat"
    header = [
                "inj. Voltage (in V)",
                "Number of Signals detected (of 0)",
                "Error",
            ]
    array = np.zeros((132, 372, 4), dtype=float)
    with open(dataFile, "r") as inp:
        inp_string = inp.read()
    sections = np.array(inp_string.split("#"))[2::2]
    sections = np.array([i.split("Error")[-1].replace("\n\n", "")[1:] for i in sections])
    for i in range(np.size(sections)):
        if i == 230*132+103:
            section = np.array(re.split("\n|\t", sections[i]))
            section = np.reshape(section, (section.size // 3, 3)).astype(float)
            section[:, 2][section[:, 2] == 0] = 0.001
            u = section[:, 0]
            ToT = section[:, 1]
            error = section[:, 2]
            index = np.invert(((ToT < 5) & (u > 0.3)) | (ToT < -5))
            u = u[index]
            ToT = ToT[index] / 2
            error = error[index]
            bounds = ([0.05, 0.0001, 0.0001], [200, 2000, 2000])
            u_0 = 0.161
            u_0_e = 0.001
            func = lambda x,a,b,c: _lambert_W_u_to_ToT(x,u_0,a,b,c)
            popt, pcov = curve_fit(
                func, u, ToT, p0=[6, 45, 60], sigma=error, absolute_sigma=True, bounds=bounds, maxfev=100000
            )
            (a, b, c) = popt
            array[i % 132, i // 132] = [u_0, a, b, c]
            plot = plotClass("/home/atlas/rballard/AtlasDataAnalysis/output/VoltageCalibration/")
            axs = plot.axs
            axs.scatter(u, ToT, marker="x", s=3, color=plot.colorPalette[0])
            axs.errorbar(u, ToT, yerr=error, fmt="none", elinewidth=1, capsize=3, color=plot.colorPalette[0], alpha=0.5)
            x = np.linspace(0, 3, 1000)
            y = _lambert_W_u_to_ToT(x, u_0, a, b, c)
            axs.plot(x, y, color=plot.colorPalette[1], label=f"Fit: $u_0$={u_0:.3f} V, a={a:.2f}, b={b:.2f}, c={c:.2f}")
            plot.set_config(
                axs,
                title=f"Voltage Calibration Layer {k} Pixel ({i % 132},{i // 132})",
                xlabel="Injected Voltage [V]",
                ylabel="ToT [TS]",
                xlim=(0, 1),
                ylim=(0, 100),
                legend=True,
            )
            plot.saveToPDF(f"VoltageCalibration_Layer{k}_Pixel_{i % 132}_{i // 132}")
            input()
        else:
            continue
    #np.save(f"/home/atlas/rballard/for_magda/data/Calibration_data/calibration/data_{5-k}.npy", array)

calibrate(4)