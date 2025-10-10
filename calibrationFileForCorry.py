import numpy as np
from dataAnalysis.handlers._hit_voltage import calibrate
def saveForCorry(k,array):
    with open(f"/home/atlas/rballard/for_magda/data/Calibration_data/calibrationDat/tot_calibration_layer_{k}.dat","w") as out:
        for j in range(372):
            for i in range(132):
                u_0, a, b, c,u_0_e, a_e, b_e, c_e = array[i,j]
                str = f"# Pixel ({i}|{j})\n  x0      {u_0} +- {0}\n  offset  {c} +- {0}\n  lnscale {a} +- {0}\n  linear  {b} +- {0}\n\n"
                out.write(str)

def tryLoadCalibrationData(k):
    import os
    file = f"/home/atlas/rballard/for_magda/data/Calibration_data/calibration/data_{k}.npy"
    if os.path.isfile(file):
        return np.load(file)
    else:
        calibrate(k)
        return np.load(file)

if __name__ == "__main__":
    for k in range(1,5):
        _array = tryLoadCalibrationData(k)
        saveForCorry(k,_array)
