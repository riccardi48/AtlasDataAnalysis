import sys

sys.path.append("..")

from scipy.stats import linregress
import numpy as np
from dataAnalysis import initDataFiles, configLoader


def R2MM(rows):
    return rows * 50  # 50 micrometer per row


def MM2R(micrometer):
    return micrometer / 50  # 50 micrometer per row


def C2MM(columns):
    return columns * 150  # 150 micrometer per column


def MM2C(micrometer):
    return micrometer / 150  # 150 micrometer per column


def isFlat(cluster):
    return np.unique(cluster.getColumns(True)).size == 1


def gaussianBinned(x, mu, sigma, scaler):
    width = x[1]-x[0]
    edges = np.append(x - width/2,x[-1] + width/2)
    return (gaussianCDFFunc(edges[1:],mu,sigma)-gaussianCDFFunc(edges[:-1],mu,sigma))*scaler

from scipy.stats import norm

def gaussianCDFFunc(x,mu,sig):
    return norm.cdf((x-mu)/sig)

def gaussianFunc(x,mu,sig,scaler):
    return norm.pdf(x,mu,sig)*scaler
