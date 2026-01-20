import numpy as np
import matplotlib.pyplot as plt
def convertToRelative(Rows,values,flipped):
    x = Rows - np.min(Rows)
    sortIndexes = np.argsort(x)
    x = x[sortIndexes]
    values = values[sortIndexes]
    x = x - np.min(x)
    if flipped:
        x = -x+x[-1]
        x = np.flip(x)
        values = np.flip(values)
    return x,values

def getColor(dataFile):
    cmap = plt.get_cmap("plasma")
    color=cmap(dataFile.voltage/48.6)
    if dataFile.fileName == "angle6_4Gev_kit_2":
        color = "r"
    return color