import json


def setConfigDefaults():
    data = {
        "maxLine": None,
        "pathToOutput": "/home/atlas/rballard/AtlasDataAnalysis/output/",
        "pathToCalcData": "/home/atlas/rballard/AtlasDataAnalysis/calculatedData/",
        "pathToData": "/home/atlas/rballard/for_magda/data/Cut/",
        "fileFormate": "202204*udp*_decode.dat",
        "pathToDataOutput": "/home/atlas/rballard/for_magda/data/pythonOutput/",
        "filterDict": {
            "telescope": "kit",
            "angle":86.5,
        }
    }
    with open("config.json", "w") as f:
        json.dump(data, f)


def loadConfig():
    data = json.load("config.json")
    return data


if __name__ == "__main__":
    setConfigDefaults()
