import numpy.typing as npt
import numpy as np
import pandas as pd
from numba import njit
import numba
from landau import landau
from psutil import Process
from scipy.special import lambertw
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import norm,chi2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm