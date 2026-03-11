from dataAnalysis._dependencies import (
    npt,  # numpy.typing
    np,  # numpy
    njit,  # numba
    tqdm,  # tqdm
    numba,  # numba
)
import warnings

List = numba.typed.List
types = numba.types

warnings.filterwarnings("ignore", "unsafe cast from int64 to int32")

################################################################################
# File split into my own code and AI written optimized code
# Original code below
# "calcClusters" is the function called externally. 
# Change the function name of the function to be called externally.
################################################################################

class clusterChecker:
    def __init__(self, layer, index, TS, TriggerID):
        self.layer = layer
        self.indexes = [index]
        self.minTS = TS
        self.maxTS = TS
        self.minTriggerID = TriggerID
        self.maxTriggerID = TriggerID

    def checkHit(self, layer, index, TS, TriggerID, timeVariance=40, triggerVariance=1):
        if self.layer != layer:
            return False
        if TriggerID <= self.maxTriggerID and TriggerID >= self.minTriggerID:
            pass
        elif (
            abs(TriggerID - self.maxTriggerID) > triggerVariance
            and abs(TriggerID - self.minTriggerID) > triggerVariance
        ):
            return False
        if TS <= self.maxTS and TS >= self.minTS:
            return True
        elif (abs(TS - self.maxTS) > timeVariance and abs(TS - self.minTS) > timeVariance) and (
            abs((TS + 512) - (self.maxTS + 512)) > timeVariance
            and abs((TS + 512) - (self.minTS + 512)) > timeVariance
        ):
            return False
        return True

    def addHit(self, index, TS, TriggerID):
        self.indexes.append(index)
        if TS > self.maxTS:
            self.maxTS = TS
        if TS < self.minTS:
            self.minTS = TS
        if TriggerID > self.maxTriggerID:
            self.maxTriggerID = TriggerID
        if TriggerID < self.minTriggerID:
            self.minTriggerID = TriggerID

    def checkActive(self, TriggerID,index, triggerVariance=100,indexVariance=50):
        if self.maxTriggerID + triggerVariance >= TriggerID or np.max(self.indexes) + indexVariance >= index:
            return True
        return False


def _calcClusters(
    Layers: npt.NDArray[np.int_],
    TriggerID: npt.NDArray[np.int_],
    TS: npt.NDArray[np.int_],
    time_variance: np.int_ = 80,
    trigger_variance: np.int_ = 1,
) -> list[npt.NDArray[np.int_]]:
    activeClusters = []
    finishedClusters = []
    for i in tqdm(range(len(Layers)), desc="Calculating Clusters"):
        layer = Layers[i]
        triggerID = TriggerID[i]
        ts = TS[i]
        index = i
        addedToCluster = False
        for cluster in activeClusters:
            if cluster.checkHit(
                layer,
                index,
                ts,
                triggerID,
                timeVariance=time_variance,
                triggerVariance=trigger_variance,
            ):
                cluster.addHit(index, ts, triggerID)
                addedToCluster = True
                break
        if not addedToCluster:
            activeClusters.append(clusterChecker(layer, index, ts, triggerID))
        finishedClusters += [
            np.array(cluster.indexes)
            for cluster in activeClusters
            if not cluster.checkActive(triggerID,index)
        ]
        activeClusters = [cluster for cluster in activeClusters if cluster.checkActive(triggerID,index)]
    finishedClusters += [cluster.indexes for cluster in activeClusters]
    return np.array(finishedClusters, dtype=object)

################################################################################
# AI optimized code below
################################################################################

@njit
def check_hit(
    cluster_layer,
    cluster_min_ts,
    cluster_max_ts,
    cluster_min_trigger_id,
    cluster_max_trigger_id,
    layer,
    ts,
    trigger_id,
    time_variance=40,
    trigger_variance=1,
):
    """Check if a hit belongs to a cluster."""
    if cluster_layer != layer:
        return False
    
    # Check trigger ID
    if trigger_id <= cluster_max_trigger_id and trigger_id >= cluster_min_trigger_id:
        pass
    elif (
        abs(trigger_id - cluster_max_trigger_id) > trigger_variance
        and abs(trigger_id - cluster_min_trigger_id) > trigger_variance
    ):
        return False
    
    # Check timestamp
    if ts <= cluster_max_ts and ts >= cluster_min_ts:
        return True
    elif (
        abs(ts - cluster_max_ts) > time_variance
        and abs(ts - cluster_min_ts) > time_variance
    ) and (
        abs((ts + 512) - (cluster_max_ts + 512)) > time_variance
        and abs((ts + 512) - (cluster_min_ts + 512)) > time_variance
    ):
        return False
    
    return True


@njit
def check_active(cluster_max_trigger_id, trigger_id, trigger_variance=10):
    """Check if a cluster is still active."""
    return cluster_max_trigger_id + trigger_variance >= trigger_id


@njit(nogil=True)
def process_hits_batch(
    layers,
    trigger_ids,
    timestamps,
    start_idx,
    end_idx,
    active_layers,
    active_min_ts,
    active_max_ts,
    active_min_trigger,
    active_max_trigger,
    active_indexes,
    finished_clusters,
    time_variance,
    trigger_variance,
):
    """Process a batch of hits and update cluster state."""
    for i in range(start_idx, end_idx):
        layer = layers[i]
        trigger_id = trigger_ids[i]
        ts = timestamps[i]
        
        added_to_cluster = False
        
        # Check active clusters
        for c in range(len(active_layers)):
            if check_hit(
                active_layers[c],
                active_min_ts[c],
                active_max_ts[c],
                active_min_trigger[c],
                active_max_trigger[c],
                layer,
                ts,
                trigger_id,
                time_variance,
                trigger_variance,
            ):
                # Add hit to cluster
                old_indexes = active_indexes[c]
                new_indexes = np.empty(len(old_indexes) + 1, dtype=np.int32)
                new_indexes[:-1] = old_indexes
                new_indexes[-1] = i
                active_indexes[c] = new_indexes
                
                # Update cluster bounds
                if ts > active_max_ts[c]:
                    active_max_ts[c] = ts
                if ts < active_min_ts[c]:
                    active_min_ts[c] = ts
                if trigger_id > active_max_trigger[c]:
                    active_max_trigger[c] = trigger_id
                if trigger_id < active_min_trigger[c]:
                    active_min_trigger[c] = trigger_id
                
                added_to_cluster = True
                break
        
        # Create new cluster if needed
        if not added_to_cluster:
            active_layers.append(layer)
            active_min_ts.append(ts)
            active_max_ts.append(ts)
            active_min_trigger.append(trigger_id)
            active_max_trigger.append(trigger_id)
            new_cluster = np.array([i], dtype=np.int32)
            active_indexes.append(new_cluster)
        
        # Move inactive clusters to finished
        c = 0
        while c < len(active_layers):
            if not check_active(active_max_trigger[c], trigger_id, 10):
                finished_clusters.append(active_indexes[c])
                
                last_idx = len(active_layers) - 1
                if c != last_idx:
                    active_layers[c] = active_layers[last_idx]
                    active_min_ts[c] = active_min_ts[last_idx]
                    active_max_ts[c] = active_max_ts[last_idx]
                    active_min_trigger[c] = active_min_trigger[last_idx]
                    active_max_trigger[c] = active_max_trigger[last_idx]
                    active_indexes[c] = active_indexes[last_idx]
                
                active_layers.pop()
                active_min_ts.pop()
                active_max_ts.pop()
                active_min_trigger.pop()
                active_max_trigger.pop()
                active_indexes.pop()
            else:
                c += 1


def calcClusters(
    Layers: npt.NDArray[np.int_],
    TriggerID: npt.NDArray[np.int_],
    TS: npt.NDArray[np.int_],
    time_variance: np.int_ = 80,
    trigger_variance: np.int_ = 1,
    batch_size: int = 10000,
) -> np.ndarray:
    """
    Calculate clusters with progress bar.
    
    Parameters:
    -----------
    Layers : array of layer indices
    TriggerID : array of trigger IDs
    TS : array of timestamps
    time_variance : time window for clustering (default: 80)
    trigger_variance : trigger ID window for clustering (default: 1)
    batch_size : size of batches for progress updates (default: 10000)
    
    Returns:
    --------
    numpy array of cluster index arrays (dtype=object)
    """
    # Convert to int32 for Numba
    layers = Layers.astype(np.int32)
    trigger_ids = TriggerID.astype(np.int32)
    timestamps = TS.astype(np.int32)
    
    n = len(layers)
    
    # Initialize typed lists
    active_layers = List.empty_list(types.int32)
    active_min_ts = List.empty_list(types.int32)
    active_max_ts = List.empty_list(types.int32)
    active_min_trigger = List.empty_list(types.int32)
    active_max_trigger = List.empty_list(types.int32)
    
    # For list of arrays
    active_indexes = List()
    active_indexes.append(np.array([0], dtype=np.int32))
    active_indexes.clear()
    
    finished_clusters = List()
    finished_clusters.append(np.array([0], dtype=np.int32))
    finished_clusters.clear()
    
    # Process in batches with progress bar
    with tqdm(total=n, desc="Calculating Clusters") as pbar:
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            
            # Process this batch (JIT compiled, releases GIL)
            process_hits_batch(
                layers,
                trigger_ids,
                timestamps,
                batch_start,
                batch_end,
                active_layers,
                active_min_ts,
                active_max_ts,
                active_min_trigger,
                active_max_trigger,
                active_indexes,
                finished_clusters,
                time_variance,
                trigger_variance,
            )
            
            # Update progress bar
            pbar.update(batch_end - batch_start)
    
    # Add remaining active clusters to finished
    for c in range(len(active_layers)):
        finished_clusters.append(active_indexes[c])
    
    # Convert to list of numpy arrays
    result = [np.array(cluster, dtype=np.int32) for cluster in finished_clusters]
    
    return np.array(result, dtype=object)