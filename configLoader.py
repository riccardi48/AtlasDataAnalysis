import json


def setConfigDefaults():
    path = "defaultConfig.json"
    config = defaultConfig()
    with open(path, "w") as f:
        json.dump(config, f)

# filterDict = {
# "telescope" : ["kit","lancs"],
# "voltage" : [0,2,4,6,8,10,15,20,30,48],
# "angle" : [0,11,20.5,28,40.5.45.86.5]
# "fileName" : ["angle6_4Gev_kit_2",...]
# }

def defaultConfig():
    config = {
        "maxLine": None,
        "pathToOutput": "/home/atlas/rballard/AtlasDataAnalysis/output/",
        "pathToCalcData": "/home/atlas/rballard/AtlasDataAnalysis/calculatedData/",
        "pathToData": "/home/atlas/rballard/for_magda/data/Cut/",
        "fileFormate": "202204*udp*_decode.dat",
        "pathToDataOutput": "/home/atlas/rballard/for_magda/data/pythonOutput/",
        "filterDict": {
            "telescope": "kit",
            "angle": 86.5,
        },
        "maxClusterWidth": 30,
        "layers": [4],
        "excludeCrossTalk": True,
    }
    return config


def saveConfig(config: dict, path: str = "defaultConfig.json"):
    with open(path, "w") as f:
        json.dump(config, f)


def loadConfig(path: str = "") -> dict:
    if path == "":
        return defaultConfig()
    try:
        with open(path) as json_file:
            config = json.load(json_file)
    except:
        print(f"{path} does not exist. Loading default config")
        return defaultConfig()
    default = defaultConfig()
    for key in default.keys():
        if key not in config:
            config[key] = default[key]
    return config


if __name__ == "__main__":
    setConfigDefaults()
