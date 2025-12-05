import sys

sys.path.append("..")

from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np

from dataAnalysis.handlers._crossTalkFinder import crossTalkFinder

def checkHit(layer1,TS1,TriggerID1,layer2,TS2,TriggerID2,timeVariance = 100, triggerVariance = 1):
    if layer1 != layer2:
        return False
    elif abs(TriggerID1 - TriggerID2) > triggerVariance:
        return False
    elif abs(TS1 - TS2) > timeVariance and abs((TS1 + 512) - (TS2 + 512)) > timeVariance:
        return False
    return True

def calcToT(TS, TS2):
    return (TS2 * 2 - TS) % 256

def checkCorrelationType(ToT1,ToT2):
    if ToT1<30 and ToT2<30:
        return 1
    if ToT1<30 and ToT2>=30 and ToT2<255:
        return 2
    if ToT1>=30 and ToT2<30 and ToT1<255:
        return 3
    if ToT1>=30 and ToT1<255 and ToT2>=30 and ToT2<255:
        return 4
    if ToT1>=255:
        return 5
    if ToT2>=255:
        return 6
    return 0

victimPixel = (45,60)

ctf = crossTalkFinder()
crosstalkRows = ctf.crossTalkDict[victimPixel[0]]

crosstalkPixels = [(int(crosstalkRows[i,0]),victimPixel[1]) for i in range(len(crosstalkRows))]
crosstalkPixels.append((200,victimPixel[1]))
crosstalkPixels.append((victimPixel[0]+1,victimPixel[1]))


config = configLoader.loadConfig()
config["filterDict"] = {"angle":0}
dataFiles = initDataFiles(config)

for dataFile in dataFiles:
    df = dataFile.get_dataFrame()
    df["Voltage"] = dataFile.get_base_attr("Hit_Voltage")[0]
    for crosstalkPixel in crosstalkPixels:
        victimPixelDF = df[(df['Row'] == victimPixel[0])&(df['Column'] == victimPixel[1])]
        victimPixelToT = []
        crosstalkPixelToT = []
        referencePixelDF = df[(df['Row'] == crosstalkPixel[0])&(df['Column'] == crosstalkPixel[1])]
        for i,victimRow in victimPixelDF.iterrows():
            for j,referenceRow in referencePixelDF.iterrows():
                if checkHit(victimRow["Layer"],victimRow["TS"],victimRow["TriggerID"],referenceRow["Layer"],referenceRow["TS"],referenceRow["TriggerID"]):
                    if victimRow["Voltage"]>=referenceRow["Voltage"]:
                        victimPixelToT.append(victimRow["Voltage"])
                        ToT = calcToT(referenceRow["TS"], referenceRow["TS2"])
                        crosstalkPixelToT.append((ToT+128)%256-128)
                    #else:
                    #    victimPixelToT.append(referenceRow["Voltage"])
                    #    crosstalkPixelToT.append(calcToT(victimRow["TS"], victimRow["TS2"]))
                    victimPixelDF.drop(i)
                    referencePixelDF.drop(j)
        victimPixelToT = np.array(victimPixelToT)
        crosstalkPixelToT = np.array(crosstalkPixelToT)
        plot = plotClass(config["pathToOutput"] + f"Correlations/singlePixel/{dataFile.fileName}/")
        axs = plot.axs
        axs.scatter(victimPixelToT,crosstalkPixelToT,color=plot.colorPalette[0],marker="x",)
        axs.scatter(victimPixelDF["Voltage"],[0 for _ in victimPixelDF["Voltage"]],color=plot.colorPalette[1],marker="x",)
        if len(victimPixelToT) > 0:
            victimPercent = len(victimPixelToT)/(len(victimPixelDF)+len(victimPixelToT))*100
        else:
            victimPercent=0
        if len(crosstalkPixelToT) > 0:
            crosstalkPercent = len(crosstalkPixelToT)/(len(referencePixelDF)+len(crosstalkPixelToT))*100
        else:
            crosstalkPercent = 0
        plot.set_config(        
            axs,
            ylim=(-5, 30),
            xlim=(0, 3),
            title=f"Pixel Correlation {dataFile.fileName} {victimPixel} {crosstalkPixel}\n{victimPercent:.2f}% of victim hits and {crosstalkPercent:.2f}% of reference hits are correlated",
            xlabel=f"{victimPixel} Voltage",
            ylabel=f"{crosstalkPixel} ToT",
            )       
        plot.saveToPDF(f"{victimPixel[0]}-{victimPixel[1]}_{crosstalkPixel[0]}-{crosstalkPixel[1]}_Voltage")
