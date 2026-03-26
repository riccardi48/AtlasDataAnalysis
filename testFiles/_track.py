"""
sensor_sync.py
==============
Synchronise two electron-hit sensors with:
  - a bounded coarse offset search (you already know they are close)
  - linear clock-drift correction estimated from sliding-window cross-correlations
  - 1-to-1 coincidence matching after drift correction

All timestamps are assumed to be integers in units of 25 ns (one TS tick).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.stats import linregress


# ──────────────────────────────────────────────────────────────────────────────
# Public result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftModel:
    """
    Linear clock-drift model.

    The corrected time of a hit in sensor B is:
        t_B_corrected = t_B - offset(t_B)
        offset(t)     = offset_ns + drift_rate * t

    Attributes
    ----------
    offset_ns : float
        Constant offset term in nanoseconds.
    drift_rate : float
        Dimensionless drift rate (ns of drift per ns of elapsed time).
        E.g. 1e-5 means 10 µs of drift per second.
    chunk_times_ns : np.ndarray
        Mid-point times of ALL chunks that passed the z-score gate (ns).
        Includes both inliers and sigma-clipped outliers.
    chunk_offsets_ns : np.ndarray
        Measured offset at each chunk mid-point (ns).
    r_squared : float
        R² of the final robust linear fit (inliers only).
    inlier_mask : np.ndarray[bool]
        True for chunks that were kept in the final fit; False for chunks
        that were rejected as outliers by sigma-clipping.
    n_outliers : int
        Number of chunks rejected as outliers.
    """
    offset_ns:        float
    drift_rate:       float
    chunk_times_ns:   np.ndarray
    chunk_offsets_ns: np.ndarray
    r_squared:        float
    inlier_mask:      np.ndarray
    n_outliers:       int

    def correct(self, ts_b_ns: np.ndarray) -> np.ndarray:
        """Return drift-corrected timestamps for sensor B."""
        return ts_b_ns - (self.offset_ns + self.drift_rate * ts_b_ns)
    
    def correct_TS(self, ts_b: np.ndarray) -> np.ndarray:
        """Return drift-corrected timestamps for sensor B."""
        ts_b_ns = ts_b* _TS_NS
        return (ts_b_ns - (self.offset_ns + self.drift_rate * ts_b_ns))/_TS_NS

    def __repr__(self) -> str:
        outlier_str = f", outliers_rejected={self.n_outliers}" if self.n_outliers else ""
        return (
            f"DriftModel(offset={self.offset_ns:.1f} ns, "
            f"drift_rate={self.drift_rate:.3e} ns/ns, "
            f"R²={self.r_squared:.4f}"
            f"{outlier_str})"
        )


@dataclass
class SyncResult:
    """
    Full result of sensor synchronisation and coincidence matching.

    Attributes
    ----------
    drift_model : DriftModel
        Fitted linear drift model (see DriftModel).
    matched_a : np.ndarray[int]
        Indices into sensor_a of matched (shared) hits.
    matched_b : np.ndarray[int]
        Indices into sensor_b of matched (shared) hits.
        ``matched_a[i]`` pairs with ``matched_b[i]``.
    dt_ns : np.ndarray[float]
        Residual time difference for each matched pair *after* drift correction,
        in nanoseconds.  Useful for checking match quality.
    only_a : np.ndarray[int]
        Indices into sensor_a of hits with no match in sensor_b
        (environmental / background hits seen only by A).
    only_b : np.ndarray[int]
        Indices into sensor_b of hits with no match in sensor_a.
    coarse_offset_ns : float
        Initial coarse offset found by the global cross-correlation (ns).
    """
    drift_model:      DriftModel
    matched_a:        np.ndarray
    matched_b:        np.ndarray
    dt_ns:            np.ndarray
    only_a:           np.ndarray
    only_b:           np.ndarray
    coarse_offset_ns: float

    # ── convenience properties ─────────────────────────────────────────────

    @property
    def n_matched(self) -> int:
        return len(self.matched_a)

    @property
    def n_only_a(self) -> int:
        return len(self.only_a)

    @property
    def n_only_b(self) -> int:
        return len(self.only_b)

    def summary(self) -> str:
        total_a = self.n_matched + self.n_only_a
        total_b = self.n_matched + self.n_only_b
        pct_a = 100.0 * self.n_matched / total_a if total_a else 0
        pct_b = 100.0 * self.n_matched / total_b if total_b else 0
        return (
            f"─── Sync Result ───────────────────────────────\n"
            f"  Coarse offset   : {self.coarse_offset_ns:.1f} ns\n"
            f"  {self.drift_model}\n"
            f"  Matched pairs   : {self.n_matched:>8,}  "
            f"({pct_a:.2f}% of A,  {pct_b:.2f}% of B)\n"
            f"  Only in A       : {self.n_only_a:>8,}\n"
            f"  Only in B       : {self.n_only_b:>8,}\n"
            f"  Median |Δt|     : {np.median(self.dt_ns):.1f} ns\n"
            f"───────────────────────────────────────────────"
        )

    def __repr__(self) -> str:
        return self.summary()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_TS_NS = 25  # nanoseconds per timestamp tick


# ──────────────────────────────────────────────────────────────────────────────
# Robust linear regression (iterative sigma-clipping)
# ──────────────────────────────────────────────────────────────────────────────

def _robust_linregress(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 2.5,
    max_iter: int = 10,
) -> tuple[float, float, float, np.ndarray]:
    """
    Fit y = slope*x + intercept using iterative sigma-clipping.

    After each ordinary least-squares fit, residuals are computed and any
    point whose |residual| > sigma * MAD_std is excluded from the next
    iteration.  Uses the median absolute deviation (MAD) as the scatter
    estimate so that a cluster of outliers cannot inflate the threshold and
    hide themselves.

    Parameters
    ----------
    x, y     : 1-D arrays of the same length (at least 2 inliers required).
    sigma    : Clipping threshold in units of the robust scatter estimate.
               2.5 is conservative enough to keep genuine scatter but tight
               enough to reject a chunk whose peak landed on a coincidental
               noise spike.
    max_iter : Maximum clipping rounds before stopping.

    Returns
    -------
    slope, intercept, r_squared, inlier_mask
        inlier_mask is a boolean array, True = point was kept in the final fit.
    """
    mask = np.ones(len(x), dtype=bool)
    slope, intercept, r_sq = 0.0, float(np.mean(y)), 0.0

    for _ in range(max_iter):
        xi, yi = x[mask], y[mask]
        if len(xi) < 2:
            break

        res = linregress(xi, yi)
        slope, intercept = res.slope, res.intercept
        r_sq = res.rvalue ** 2

        # Residuals for ALL points against the current fit.
        residuals = y - (slope * x + intercept)

        # MAD of *inlier* residuals as a robust scatter estimate.
        inlier_res = residuals[mask]
        mad = np.median(np.abs(inlier_res - np.median(inlier_res)))
        robust_std = 1.4826 * mad if mad > 0 else np.std(inlier_res)
        robust_std = max(robust_std, 1e-12)  # guard against perfect-line edge case

        new_mask = np.abs(residuals) <= sigma * robust_std

        if np.array_equal(new_mask, mask):
            break  # converged — no change
        mask = new_mask

    return slope, intercept, r_sq, mask


def _to_ns_array(hits: list | np.ndarray) -> np.ndarray:
    """Convert a list/array of TS ticks to a sorted float64 array in ns."""
    arr = np.asarray(hits, dtype=np.float64) * _TS_NS
    arr.sort()
    return arr


def _cross_correlate_bounded(
    ts_a: np.ndarray,
    ts_b: np.ndarray,
    search_min_ns: float,
    search_max_ns: float,
    bin_ns: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a histogram of (t_B - t_A) differences restricted to
    [search_min_ns, search_max_ns].

    Uses a two-pointer sweep so cost is O((N_A + N_B) + K) where K is the
    number of pairs inside the window — much cheaper than a full O(N²) scan
    when the search window is small relative to the data length.

    Returns
    -------
    bin_centres : np.ndarray   (ns)
    counts      : np.ndarray   (int)
    """
    bins = np.arange(search_min_ns, search_max_ns + bin_ns, bin_ns)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)

    j_start = 0
    for i in range(len(ts_a)):
        t_lo = ts_a[i] + search_min_ns
        t_hi = ts_a[i] + search_max_ns

        # Advance left pointer
        while j_start < len(ts_b) and ts_b[j_start] < t_lo:
            j_start += 1

        j = j_start
        while j < len(ts_b) and ts_b[j] <= t_hi:
            delta = ts_b[j] - ts_a[i]
            idx = int((delta - search_min_ns) / bin_ns)
            if 0 <= idx < len(counts):
                counts[idx] += 1
            j += 1

    return bins[:-1] + bin_ns / 2.0, counts


def _peak_offset(bin_centres: np.ndarray, counts: np.ndarray) -> float:
    """
    Return the offset of the histogram peak with sub-bin accuracy using
    parabolic interpolation around the maximum bin.
    """
    idx = int(np.argmax(counts))
    # Boundary cases: no interpolation possible
    if idx == 0 or idx == len(counts) - 1:
        return float(bin_centres[idx])
    y0, y1, y2 = counts[idx - 1], counts[idx], counts[idx + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(bin_centres[idx])
    # Sub-bin shift in units of bin width
    bin_width = float(bin_centres[1] - bin_centres[0])
    shift = 0.5 * (y0 - y2) / denom  # in bins
    return float(bin_centres[idx]) + shift * bin_width


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def sync_sensors(
    sensor_a: list | np.ndarray,
    sensor_b: list | np.ndarray,
    # ── max offset constraint (set ONE or NEITHER) ────────────────────────────
    max_offset_ts: Optional[int]   = None,
    max_offset_ns: Optional[float] = None,
    bunch_spacing_ts: Optional[int] = None,
    max_offset_bunches: Optional[int] = None,
    # ── drift estimation ──────────────────────────────────────────────────────
    n_drift_chunks: int   = 20,
    min_hits_per_chunk: int = 30,
    # ── coincidence matching ──────────────────────────────────────────────────
    coincidence_window_ts: int   = 4,
    coincidence_window_ns: Optional[float] = None,
    # ── diagnostics ──────────────────────────────────────────────────────────
    verbose: bool = True,
    max_Time_for_init_search=3000000000,
) -> SyncResult:
    """
    Synchronise two hit-time sensors and find shared (coincident) hits.

    Parameters
    ----------
    sensor_a, sensor_b : list or np.ndarray
        Hit timestamps as integer TS ticks (1 tick = 25 ns).
        They do **not** need to be pre-sorted.

    max_offset_ts : int, optional
        Maximum expected offset between the two sensors in TS ticks.
        The coarse cross-correlation only searches ± this value.
    max_offset_ns : float, optional
        Same as max_offset_ts but in nanoseconds.
        (Exactly one of max_offset_ts / max_offset_ns / bunch-based args
         should be set; if none are given a default of ±10 000 ns is used.)
    bunch_spacing_ts : int, optional
        Spacing between bunches in TS ticks.  Required if using
        ``max_offset_bunches``.
    max_offset_bunches : int, optional
        Maximum expected offset in units of bunch spacing.
        Converted internally to ns via bunch_spacing_ts.

    n_drift_chunks : int
        Number of time slices used to estimate the clock drift.
        More chunks → finer drift resolution but each chunk has fewer hits.
        Default 20.
    min_hits_per_chunk : int
        Chunks with fewer hits in either sensor are skipped for drift
        estimation.  Default 30.

    coincidence_window_ts : int
        Half-width of the matching window in TS ticks (default 4 → ±100 ns).
        A pair is accepted if |t_A - t_B_corrected| ≤ window.
    coincidence_window_ns : float, optional
        Same as coincidence_window_ts but in nanoseconds.  Overrides
        coincidence_window_ts if given.

    verbose : bool
        Print progress and summary to stdout.  Default True.

    Returns
    -------
    SyncResult
        See ``SyncResult`` dataclass for full documentation of fields.

    Examples
    --------
    >>> result = sync_sensors(
    ...     sensor_a, sensor_b,
    ...     max_offset_bunches=3,
    ...     bunch_spacing_ts=40,        # 40 ticks × 25 ns = 1 µs bunches
    ...     coincidence_window_ts=4,    # ±100 ns matching window
    ... )
    >>> print(result.summary())
    >>> shared_hits_a = sensor_a_array[result.matched_a]
    >>> shared_hits_b = sensor_b_array[result.matched_b]
    """

    # ── 1.  Resolve maximum search offset ────────────────────────────────────
    if max_offset_bunches is not None:
        if bunch_spacing_ts is None:
            raise ValueError(
                "bunch_spacing_ts must be provided when using max_offset_bunches."
            )
        max_offset_ns_resolved = max_offset_bunches * bunch_spacing_ts * _TS_NS
    elif max_offset_ts is not None:
        max_offset_ns_resolved = max_offset_ts * _TS_NS
    elif max_offset_ns is not None:
        max_offset_ns_resolved = float(max_offset_ns)
    else:
        max_offset_ns_resolved = 10_000.0  # 10 µs default
        if verbose:
            print(f"[sync] No max offset specified — defaulting to ±{max_offset_ns_resolved:.0f} ns")

    if verbose:
        print(f"[sync] Max offset search window : ±{max_offset_ns_resolved:.0f} ns "
              f"({max_offset_ns_resolved / _TS_NS:.0f} TS ticks)")

    # ── 2.  Resolve coincidence window ────────────────────────────────────────
    if coincidence_window_ns is not None:
        cw_ns = float(coincidence_window_ns)
    else:
        cw_ns = coincidence_window_ts * _TS_NS

    if verbose:
        print(f"[sync] Coincidence window       : ±{cw_ns:.0f} ns")

    # ── 3.  Convert to sorted ns arrays ──────────────────────────────────────
    ts_a = _to_ns_array(sensor_a)
    ts_b = _to_ns_array(sensor_b)

    if verbose:
        print(f"[sync] Sensor A hits: {len(ts_a):,}   Sensor B hits: {len(ts_b):,}")

    # ── 4.  Coarse offset via global cross-correlation ────────────────────────
    if verbose:
        print("[sync] Step 1/3 — Coarse cross-correlation …")
    min_ts = np.min([np.min(ts_a),np.min(ts_b)])
    maxTime = max_Time_for_init_search*_TS_NS
    bin_centres, counts = _cross_correlate_bounded(
        ts_a[ts_a<min_ts + maxTime], ts_b[ts_b<min_ts + maxTime],
        search_min_ns=-max_offset_ns_resolved,
        search_max_ns= max_offset_ns_resolved,
        bin_ns=_TS_NS*100,
    )

    # The peak must stand above the noise floor with statistical significance.
    # Use a z-score: (peak - mean) / std > threshold.
    # This is robust to wide search windows where the absolute noise floor is
    # high but the peak is still a clear outlier.
    if counts.max() == 0:
        raise RuntimeError(
            "Cross-correlation found zero pairs inside the search window. "
            "The true offset is likely outside the search window. "
            "Consider increasing max_offset_ts / max_offset_ns / max_offset_bunches."
        )
    counts_mean = counts.mean()
    counts_std  = counts.std()
    peak_zscore = (counts.max() - counts_mean) / max(counts_std, 1.0)
    if peak_zscore < 3.0:
        raise RuntimeError(
            f"Cross-correlation found no significant peak inside the search "
            f"window (peak z-score={peak_zscore:.2f} < 3.0, "
            f"max count={counts.max()}, mean={counts_mean:.1f}, std={counts_std:.1f}). "
            "The true offset is likely outside the search window. "
            "Consider increasing max_offset_ts / max_offset_ns / max_offset_bunches."
        )

    coarse_offset = _peak_offset(bin_centres, counts)

    if verbose:
        print(f"[sync]   Coarse offset = {coarse_offset:.1f} ns  "
              f"(peak count = {counts.max():,})")

    # ── 5.  Sliding-window drift estimation ───────────────────────────────────
    if verbose:
        print(f"[sync] Step 2/3 — Drift estimation ({n_drift_chunks} chunks) …")

    t_min = min(ts_a[0],  ts_b[0])
    t_max = max(ts_a[-1], ts_b[-1])
    duration = t_max - t_min
    chunk_size = duration / n_drift_chunks

    # Per-chunk refinement window: narrow band around the coarse offset.
    # Using a tight window is critical — if it is as wide as max_offset the
    # noise floor dominates and random peaks corrupt the drift fit.
    # We allow ±(max_offset / n_drift_chunks * 3) clamped to a sensible range.
    refine_window = max(
        3 * _TS_NS,                              # at least 3 ticks
        min(
            max_offset_ns_resolved / n_drift_chunks * 4,
            max_offset_ns_resolved * 0.1,        # never more than 30 % of total range
        ),
    )
    # But always at least wide enough to detect the true drift across one chunk.
    # Estimated max drift across one chunk = |drift_estimate| * chunk_size.
    # At this stage we don't know drift, so use a generous 200 ns floor.
    refine_window = max(refine_window, 200.0)

    if verbose:
        print(f"[sync]   Per-chunk refinement window: ±{refine_window:.0f} ns")

    chunk_times   = []
    chunk_offsets = []

    for c in range(n_drift_chunks):
        t_lo = t_min + c * chunk_size
        t_hi = t_lo + chunk_size
        mid  = (t_lo + t_hi) / 2.0

        sub_a = ts_a[(ts_a >= t_lo) & (ts_a < t_hi)]
        # Select B hits in the expected time range (shifted by coarse offset)
        # with an extra ±refine_window guard band.
        sub_b = ts_b[
            (ts_b >= t_lo + coarse_offset - refine_window) &
            (ts_b <  t_hi + coarse_offset + refine_window)
        ]

        if len(sub_a) < min_hits_per_chunk or len(sub_b) < min_hits_per_chunk:
            if verbose:
                print(f"[sync]   Chunk {c+1:>2}/{n_drift_chunks}: skipped "
                      f"(A={len(sub_a)}, B={len(sub_b)} hits)")
            continue

        bc, cnt = _cross_correlate_bounded(
            sub_a, sub_b,
            search_min_ns=coarse_offset - refine_window,
            search_max_ns=coarse_offset + refine_window,
            bin_ns=_TS_NS*50,
        )

        if cnt.max() == 0:
            continue

        # Require the peak to be a clear outlier (z-score > 3) so that
        # noise-dominated chunks don't corrupt the drift fit.
        c_mean = cnt.mean()
        c_std  = cnt.std()
        z = (cnt.max() - c_mean) / max(c_std, 1.0)
        if z < 3.0:
            if verbose:
                print(f"[sync]   Chunk {c+1:>2}/{n_drift_chunks}: weak peak "
                      f"(z={z:.2f}, peak={cnt.max()}) — skipped")
            continue

        peak = _peak_offset(bc, cnt)
        chunk_times.append(mid)
        chunk_offsets.append(peak)

        if verbose:
            print(f"[sync]   Chunk {c+1:>2}/{n_drift_chunks}: "
                  f"t_mid={mid/1e6:.1f} µs  offset={peak:.1f} ns  "
                  f"peak_count={cnt.max()}  z={z:.1f}")

    if len(chunk_offsets) < 2:
        warnings.warn(
            "Fewer than 2 chunks had enough hits for drift estimation. "
            "Using coarse offset only (no drift correction). "
            "Try reducing n_drift_chunks or min_hits_per_chunk.",
            RuntimeWarning,
        )
        empty = np.array([], dtype=bool)
        drift_model = DriftModel(
            offset_ns        = coarse_offset,
            drift_rate       = 0.0,
            chunk_times_ns   = np.array(chunk_times),
            chunk_offsets_ns = np.array(chunk_offsets),
            r_squared        = float("nan"),
            inlier_mask      = empty,
            n_outliers       = 0,
        )
    else:
        ct = np.array(chunk_times)
        co = np.array(chunk_offsets)
        slope, intercept, r_sq, inlier_mask = _robust_linregress(ct, co)
        n_outliers = int((~inlier_mask).sum())

        if verbose and n_outliers > 0:
            outlier_offsets = co[~inlier_mask]
            print(f"[sync]   Sigma-clipping rejected {n_outliers} chunk(s) as outliers: "
                  f"offsets = {np.round(outlier_offsets, 1).tolist()} ns")

        drift_model = DriftModel(
            offset_ns        = intercept,
            drift_rate       = slope,
            chunk_times_ns   = ct,
            chunk_offsets_ns = co,
            r_squared        = r_sq,
            inlier_mask      = inlier_mask,
            n_outliers       = n_outliers,
        )

    if verbose:
        print(f"[sync]   {drift_model}")

    # ── 6.  Apply drift correction and match coincidences ─────────────────────
    if verbose:
        print("[sync] Step 3/3 — Coincidence matching …")

    match_result = match_hits(
        sensor_a, sensor_b,
        drift_model           = drift_model,
        coincidence_window_ts = coincidence_window_ts,
        coincidence_window_ns = coincidence_window_ns,
        verbose               = False,
    )

    result = SyncResult(
        drift_model      = drift_model,
        matched_a        = match_result.matched_a,
        matched_b        = match_result.matched_b,
        dt_ns            = match_result.dt_ns,
        only_a           = match_result.only_a,
        only_b           = match_result.only_b,
        coarse_offset_ns = coarse_offset,
    )

    if verbose:
        print(result.summary())

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public: coincidence matching against a known drift model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """
    Result of ``match_hits``.

    All indices refer to the arrays that were passed into ``match_hits``,
    so they are valid whether you called it on a subset or the full dataset.

    Attributes
    ----------
    matched_a : np.ndarray[int]
        Indices into sensor_a of matched hits.
    matched_b : np.ndarray[int]
        Indices into sensor_b of their partners.  ``matched_a[i]`` pairs
        with ``matched_b[i]``.
    dt_ns : np.ndarray[float]
        Residual |Δt| for each pair after drift correction (ns).
    only_a : np.ndarray[int]
        Indices into sensor_a of unmatched hits.
    only_b : np.ndarray[int]
        Indices into sensor_b of unmatched hits.
    """
    matched_a: np.ndarray
    matched_b: np.ndarray
    dt_ns:     np.ndarray
    only_a:    np.ndarray
    only_b:    np.ndarray

    @property
    def n_matched(self) -> int:
        return len(self.matched_a)

    def summary(self) -> str:
        total_a = len(self.matched_a) + len(self.only_a)
        total_b = len(self.matched_b) + len(self.only_b)
        pct_a = 100.0 * self.n_matched / total_a if total_a else 0
        pct_b = 100.0 * self.n_matched / total_b if total_b else 0
        med_dt = np.median(self.dt_ns) if len(self.dt_ns) else float("nan")
        return (
            f"─── Match Result ──────────────────────────────\n"
            f"  Matched pairs   : {self.n_matched:>8,}  "
            f"({pct_a:.2f}% of A,  {pct_b:.2f}% of B)\n"
            f"  Only in A       : {len(self.only_a):>8,}\n"
            f"  Only in B       : {len(self.only_b):>8,}\n"
            f"  Median |Δt|     : {med_dt:.1f} ns\n"
            f"───────────────────────────────────────────────"
        )

    def __repr__(self) -> str:
        return self.summary()


def match_hits(
    sensor_a: list | np.ndarray,
    sensor_b: list | np.ndarray,
    drift_model: DriftModel,
    coincidence_window_ts: int            = 4,
    coincidence_window_ns: Optional[float] = None,
    verbose: bool                          = False,
) -> MatchResult:
    """
    Apply a known drift model and find coincident (shared) hits.

    This is the matching step of ``sync_sensors`` exposed as a standalone
    function so you can:

    * Run it on a **subset** of your data (e.g. one spill or time window)
      to study the coincidence rate in that region.
    * Re-run it on the **full dataset** after fitting the drift model on a
      representative subset, without repeating the expensive cross-correlation.

    Returned indices always refer to the arrays you pass in, not to any
    internal sorted copy, so they can be used directly as::

        shared_a_hits = [sensor_a[i] for i in result.matched_a]

    Parameters
    ----------
    sensor_a, sensor_b : list or np.ndarray
        Hit timestamps in TS ticks (integers, 1 tick = 25 ns).
        Need not be pre-sorted.
    drift_model : DriftModel
        Fitted drift model, e.g. from ``sync_sensors(...).drift_model``
        or constructed manually.
    coincidence_window_ts : int
        Half-width of the matching window in TS ticks (default 4 → ±100 ns).
    coincidence_window_ns : float, optional
        Same as coincidence_window_ts but in nanoseconds.  Overrides
        coincidence_window_ts if given.
    verbose : bool
        Print a short summary on completion.

    Returns
    -------
    MatchResult
        See ``MatchResult`` for field documentation.

    Examples
    --------
    Fit on a subset, apply to the full run::

        # Fit drift model on first 10 % of hits (faster, still representative)
        n = len(sensor_a)
        result_sub = sync_sensors(sensor_a[:n//10], sensor_b[:n//10], ...)

        # Apply to the full dataset — indices refer to the full lists
        full_match = match_hits(sensor_a, sensor_b, result_sub.drift_model)
        shared_a = [sensor_a[i] for i in full_match.matched_a]
    """
    # Resolve coincidence window
    cw_ns = float(coincidence_window_ns) if coincidence_window_ns is not None \
            else coincidence_window_ts * _TS_NS

    # Convert to ns, keeping track of the sort order so we can map back to
    # the original input indices at the end.
    arr_a  = np.asarray(sensor_a, dtype=np.float64) * _TS_NS
    arr_b  = np.asarray(sensor_b, dtype=np.float64) * _TS_NS
    orig_a = np.argsort(arr_a)
    orig_b = np.argsort(arr_b)
    ts_a   = arr_a[orig_a]              # sorted ns timestamps for A
    ts_b_corr = drift_model.correct(arr_b)[orig_b]   # drift-corrected, sorted

    # Greedy 1-to-1 nearest-neighbour match via two-pointer sweep.
    matched_sorted_a: list[int] = []
    matched_sorted_b: list[int] = []
    dt_list:          list[float] = []

    used_b  = np.zeros(len(ts_b_corr), dtype=bool)
    j_start = 0

    for i in range(len(ts_a)):
        t_lo = ts_a[i] - cw_ns
        t_hi = ts_a[i] + cw_ns

        while j_start < len(ts_b_corr) and ts_b_corr[j_start] < t_lo:
            j_start += 1

        best_dt = np.inf
        best_j  = -1
        j = j_start
        while j < len(ts_b_corr) and ts_b_corr[j] <= t_hi:
            if not used_b[j]:
                dt = abs(ts_a[i] - ts_b_corr[j])
                if dt < best_dt:
                    best_dt = dt
                    best_j  = j
            j += 1

        if best_j != -1:
            matched_sorted_a.append(i)
            matched_sorted_b.append(best_j)
            dt_list.append(best_dt)
            used_b[best_j] = True

    ms_a = np.array(matched_sorted_a, dtype=np.intp)
    ms_b = np.array(matched_sorted_b, dtype=np.intp)
    dt_arr = np.array(dt_list, dtype=np.float64)

    # Map sorted-array indices back to original input indices
    matched_orig_a = orig_a[ms_a] if len(ms_a) else np.array([], dtype=np.intp)
    matched_orig_b = orig_b[ms_b] if len(ms_b) else np.array([], dtype=np.intp)

    unmatched_sorted_a = np.setdiff1d(np.arange(len(ts_a)), ms_a)
    unmatched_sorted_b = np.setdiff1d(np.arange(len(ts_b_corr)), ms_b)
    only_orig_a = orig_a[unmatched_sorted_a]
    only_orig_b = orig_b[unmatched_sorted_b]

    result = MatchResult(
        matched_a = matched_orig_a,
        matched_b = matched_orig_b,
        dt_ns     = dt_arr,
        only_a    = only_orig_a,
        only_b    = only_orig_b,
    )

    if verbose:
        print(result.summary())

    return result