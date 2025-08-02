from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    npt,  # numpy.typing
    np,  # numpy
)


def _fileNameFromPath(path: str) -> str:
    return "_".join(path.split("/")[-1].split(".")[0].split("_")[3:]).removesuffix("_decode")


def readFileName(path: str) -> tuple[str, int, float, str]:
    file_name: str = _fileNameFromPath(path)
    angle_dict = {
        "angle1": 45,
        "angle2": 40.5,
        "angle3": 28,
        "angle4": 20.5,
        "angle5": 11,
        "angle6": 86.5,
    }
    angle: float = 0
    for k in angle_dict.keys():
        if k in file_name:
            angle = angle_dict[k]
            break
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
    voltage: int = 48
    for k in voltage_dict.keys():
        if k in file_name:
            voltage = voltage_dict[k]
            break

    telescope: str = "unknown"
    if "kit" in path:
        telescope = "kit"
    elif "lancs" in path:
        telescope = "lancs"
    return file_name, voltage, angle, telescope


def calcToT(TS: npt.NDArray[np.int_], TS2: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    return (TS2 * 2 - TS) % 256


def trueTimeStamps(clusters: clusterArray, ext_TS: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    new_ext_TS = np.zeros(ext_TS.size)
    for cluster in clusters:
        firstTS = np.min(cluster.getTSs(excludeCrossTalk=True))
        firstTS1024 = firstTS % 1024
        new_ext_TS[cluster.getIndexes()] = firstTS + ((cluster.getTSs() % 1024) - firstTS1024)
    return new_ext_TS.astype(np.int_)


def TStoMS(TS: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    return TS * 25 / 1000000


def MStoTS(Time: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    return np.round(Time * 1000000 / 25).astype(np.int_)
