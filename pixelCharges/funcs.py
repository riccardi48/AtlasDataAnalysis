import numpy as np
import sys
sys.path.append("..")
from landau import landau
from scipy.stats import linregress

def angle_with_error_mc(orthCharge, error_orthCharge, charge, error_charge, n_samples=10000):
    """
    Monte Carlo error propagation for: angle = 90 - rad2deg(arcsin(orthCharge/charge))
    """
    # Generate random samples from normal distributions
    orthCharge_samples = np.random.normal(orthCharge, error_orthCharge, n_samples)
    charge_samples = np.random.normal(charge, error_charge, n_samples)
    
    # Calculate ratio
    ratio_samples = orthCharge_samples / charge_samples
    
    # Only keep valid samples (ratio in [-1, 1])
    valid_mask = np.abs(ratio_samples) <= 1
    valid_ratios = ratio_samples[valid_mask]
    
    if len(valid_ratios) == 0:
        return np.nan, np.nan
    
    # Calculate full angle formula for valid samples
    angle_samples = 90 - np.rad2deg(np.arcsin(valid_ratios))
    
    # Mean and standard deviation
    mean_angle = np.mean(angle_samples)
    error_angle = np.std(angle_samples)
    validity_fraction = len(valid_ratios) / n_samples
    
    return mean_angle, error_angle


def isTrack(cluster,voltage=48.6):
    if np.unique(cluster.getColumns(excludeCrossTalk=True)).size > 1:
        return False
    #landau_value = landau.cdf(cluster.getClusterCharge(excludeCrossTalk=True), 13, 1)
    #if landau_value < 0.01 or landau_value > 0.99:
    #   return False
    if cluster.getClusterCharge(excludeCrossTalk=True) < 10 * np.sqrt(voltage/48.6) or cluster.getClusterCharge(excludeCrossTalk=True) > 22:
        return False
    #relativeRows = cluster.getRows(excludeCrossTalk=True)-np.min(cluster.getRows(excludeCrossTalk=True))
    #Timestamps = cluster.getTSs(True)
    #TS = Timestamps - np.min(Timestamps)
    #result = linregress(relativeRows,TS,nan_policy="omit")
    #if result.slope > 0 and np.max(TS) > 2:
        #print(cluster.getIndex(),"*")
    #    return False
    #if np.max(TS)>10 and result.slope > -0.2:
        #print(cluster.getIndex(),"!")
    #    return False
    return True

