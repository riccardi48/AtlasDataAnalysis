from dataAnalysis._types import clusterClass,dataAnalysis,clusterArray
from dataAnalysis._dependencies import (
    npt,                # numpy.typing
    np,                 # numpy
    njit,               # numba
)
import warnings
warnings.filterwarnings('ignore', 'unsafe cast from int64 to int32')
def calcClusters(
    Layers: npt.NDArray[np.int_],
    TriggerID: npt.NDArray[np.int_],
    TS: npt.NDArray[np.int_],
    time_variance: np.int_ = 80,
    trigger_variance: np.int_ = 1,
) -> list[npt.NDArray[np.int_]]:
    """Optimized cluster calculation with vectorized operations and JIT compilation"""
    
    # Pre-allocate result list
    all_clusters = []
    
    # Process each layer
    for layer in range(1, 5):
        layer_mask = Layers == layer
        if not np.any(layer_mask):
            continue
            
        layer_indices = np.where(layer_mask)[0]
        if len(layer_indices) <= 1:
            # Single pixel clusters
            for idx in layer_indices:
                all_clusters.append(np.array([idx], dtype=np.int32))
            continue
        
        # Extract layer data
        layer_ts = TS[layer_mask]
        layer_trigger = TriggerID[layer_mask]
        
        # Use JIT-compiled function for the heavy lifting
        

        cluster_boundaries = _find_cluster_boundaries_jit(
            layer_ts, layer_trigger, time_variance, trigger_variance
        )
        
        # Build clusters from boundaries
        start_idx = 0
        for end_idx in cluster_boundaries:
            cluster_indices = layer_indices[start_idx:end_idx+1]
            all_clusters.append(cluster_indices.astype(np.int32))
            start_idx = end_idx + 1
    return np.array(all_clusters,dtype=object)


@njit(cache=True)
def _find_cluster_boundaries_jit(ts_array, trigger_array, time_var, trigger_var):
    """JIT-compiled function to find cluster boundaries"""
    n = len(ts_array)
    if n <= 1:
        return np.array([0], dtype=np.int32)
    
    boundaries = []
    
    for i in range(n - 1):
        # Calculate time difference with wraparound handling
        ts_diff1 = abs(ts_array[i + 1] - ts_array[i])
        ts_diff2 = abs((ts_array[i + 1] + 512) % 1024 - (ts_array[i] + 512) % 1024)
        min_ts_diff = min(ts_diff1, ts_diff2)
        
        # Calculate trigger difference
        trigger_diff = abs(trigger_array[i + 1] - trigger_array[i])
        
        # Check if this should end a cluster
        if min_ts_diff > time_var or trigger_diff > trigger_var:
            boundaries.append(i)
    
    # Always end with the last index
    boundaries.append(n - 1)
    
    return np.array(boundaries, dtype=np.int32)