import numpy.typing as npt
import numpy as np
import pandas as pd
from numba import njit
import numba
from landau import landau
from psutil import Process
from scipy.special import lambertw
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages