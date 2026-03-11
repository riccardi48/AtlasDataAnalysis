import sys

sys.path.append("..")

from dataAnalysis.handlers._genericClusterFuncs import isFlat
from dataAnalysis import initDataFiles, configLoader
from plotAnalysis import plotClass
import numpy as np
from bisect import bisect_left

MOD32 = 2**32
MOD10 = 2**10

def lis_pairs(pairs):
    """Return longest increasing subsequence of (i, j) pairs where both i and j increase."""
    # Sort by i, then find LIS on j values
    pairs = sorted(pairs)
    tails, indices, parents = [], [], {}
    for p in pairs:
        j = p[1]
        pos = bisect_left(tails, j)
        if pos == len(tails): tails.append(j)
        else: tails[pos] = j
        indices.append((pos, p))

    # Reconstruct
    result, target = [], len(tails) - 1
    for pos, p in reversed(indices):
        if pos == target:
            result.append(p); target -= 1
    return list(reversed(result))


def sync_sensors(ts_a32, ts_b32, tolerance=20):
    """
    Sync two sensors with different 32-bit clock offsets and potential rollover.
    Uses 10-bit values for matching, ordering constraint to reject false matches,
    and 32-bit timestamps to resolve the coarse offset.
    tolerance: allowed difference in 10-bit counts (default 4 = 100ns at 25ns/count)

    Returns (offset, matched, only_a, only_b)
      offset:  integer to add to b's 32-bit timestamps to align with a
      matched: list of (a_idx, b_idx) pairs
      only_a:  indices into ts_a32 with no match
      only_b:  indices into ts_b32 with no match
    """
    a32 = np.array(ts_a32, dtype=np.int64) % MOD32
    b32 = np.array(ts_b32, dtype=np.int64) % MOD32

    # All candidate matches within tolerance on 10-bit circular distance
    candidates = []
    for j, bv in enumerate(b32):
        diffs = np.minimum((a32 - bv) % MOD32, (bv - a32) % MOD32)
        for i in np.where(diffs <= tolerance)[0]:
            candidates.append((int(i), j))

    # Keep only order-preserving matches (LIS on index pairs)
    matched = lis_pairs(candidates)

    if not matched:
        raise ValueError("No matches found — check tolerance or data alignment")

    # Estimate offset: median of (a32 - b32) over matched pairs, unwrapped
    raw_offsets = [(a32[i] - b32[j]) % MOD32 for i, j in matched]
    raw_offsets = [o - MOD32 if o > MOD32 // 2 else o for o in raw_offsets]
    offset = int(np.median(raw_offsets))

    used_a = {i for i, _ in matched}
    used_b = {j for _, j in matched}
    only_a = [i for i in range(len(a32)) if i not in used_a]
    only_b = [j for j in range(len(b32)) if j not in used_b]
    return offset, matched, only_a, only_b


config = configLoader.loadConfig()
config["filterDict"] = {"angle": 86.5 , "voltage":48.6}
config["filterDict"] = {"fileName":["angle6_4Gev_kit_2","angle6_4GeV_lancs_3"]}
dataFiles = initDataFiles(config)

clusters1 = dataFiles[0].get_clusters(excludeCrossTalk=True,layers=config["layers"])

a10 = np.array([cluster.getIndexes(True)[0] for cluster in clusters1])
a32 = np.array([cluster.getEXT_TSs(True)[0] for cluster in clusters1])
a32 = dataFiles[0].get_base_attr("TriggerID")[a10]

clusters2 = dataFiles[1].get_clusters(excludeCrossTalk=True,layers=config["layers"])


b10 = np.array([cluster.getIndexes(True)[0] for cluster in clusters2])
b32 = np.array([cluster.getEXT_TSs(True)[0] for cluster in clusters2])
b32 = dataFiles[1].get_base_attr("TriggerID")[b10]

offset, matched, only_a, only_b = sync_sensors(a32[:10000], b32[:10000])

#print(f"True offset:     {OFFSET}")
print(f"Detected offset: {offset}")
#print(f"Matched pairs (a_idx, b_idx): {matched}")

for a_idx,b_idx in matched:
    print(clusters1[a_idx].getTSs(True))
    print(clusters1[a_idx].getRows(True))
    print(clusters1[a_idx].getColumns(True))
    print(clusters2[b_idx].getTSs(True))
    print(clusters2[b_idx].getRows(True))
    print(clusters2[b_idx].getColumns(True))
    print(clusters2[b_idx].getColumns(True)[0]-clusters1[a_idx].getColumns(True))
    input()