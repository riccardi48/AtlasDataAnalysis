from os import getpid
from ._dependencies import Process
def usage() -> float:
    process = Process(getpid())
    return process.memory_info()[0] / float(2**20)

def printMemUsage() -> None:
    print(f"\033[93mCurrent Mem Usage:{usage():.2f}Mb\033[0m")
