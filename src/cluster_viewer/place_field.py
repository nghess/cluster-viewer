"""Place-field math: spike alignment, occupancy/rate maps, Skaggs spatial
information, and shuffle-based significance (SSI). Ported from
reference/place_field_basics.py, adapted to work directly on position
arrays (frame_ms/x/y) instead of a mtrack DataFrame, since session_data.py
already has those arrays on hand. Pure numpy/scipy - no plotting here.
"""
import numpy as np
from scipy.stats import norm


def make_bin_edges(x_range, y_range, bin_size_px):
    """
    Spatial bin edges tiling a fixed physical extent exactly (e.g. the known
    arena bounds in config.ARENA_X_RANGE_PX/ARENA_Y_RANGE_PX), rather than
    the tracked positions' own min/max - which run slightly outside the
    arena in places (tracking noise) and would otherwise shift bin edges off
    the true (0, 0) corner and stretch them past the far wall.
    """
    def edges_for(lo, hi):
        n_bins = max(1, round((hi - lo) / bin_size_px))
        return np.linspace(lo, hi, n_bins + 1)

    return edges_for(*x_range), edges_for(*y_range)


def align_spikes(spike_times, frame_ms, x, y):
    """Nearest-tracked-frame (x, y) for each spike time."""
    idx = np.searchsorted(frame_ms, spike_times)
    idx = np.clip(idx, 1, len(frame_ms) - 1)
    left_diff = np.abs(spike_times - frame_ms[idx - 1])
    right_diff = np.abs(spike_times - frame_ms[idx])
    closest = np.where(left_diff < right_diff, idx - 1, idx)
    return x[closest], y[closest]


def compute_occupancy(frame_ms, x, y, x_edges, y_edges):
    """
    Time spent per spatial bin (seconds), from tracked positions.

    Assumes a roughly constant frame interval (uses the median of the first
    100 inter-frame gaps to convert frame counts to seconds).
    """
    occ_hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    median_interval_ms = np.median(np.diff(frame_ms[:100]))
    time_per_frame_s = median_interval_ms / 1000.0
    occupancy_time = occ_hist * time_per_frame_s
    return occupancy_time, time_per_frame_s


def compute_rate_map(spike_x, spike_y, occupancy_time, x_edges, y_edges, min_occupancy_s):
    """
    Firing rate map (Hz): spike count per bin / occupancy time per bin.
    Bins visited less than min_occupancy_s are set to NaN.
    """
    spike_hist, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_edges, y_edges])
    rate_map = np.divide(
        spike_hist, occupancy_time,
        out=np.full_like(spike_hist, np.nan, dtype=float),
        where=occupancy_time >= min_occupancy_s
    )
    return rate_map, spike_hist


def skaggs_si(rate_map_flat, p_i):
    """
    Vectorized Skaggs spatial information.

        SI = sum_i  p_i * (r_i / r_mean) * log2(r_i / r_mean)

    rate_map_flat : flattened rate map (Hz), NaN for unvisited bins
    p_i : flattened occupancy probability per bin (same shape, sums to 1)

    Returns SI in bits/spike, or 0.0 if the mean rate is zero.
    """
    valid = ~np.isnan(rate_map_flat)
    denom = np.sum(p_i[valid])
    if denom == 0:
        return 0.0
    r_mean = np.sum(rate_map_flat[valid] * p_i[valid]) / denom
    if r_mean == 0:
        return 0.0
    active = valid & (rate_map_flat > 0)
    ratio = rate_map_flat[active] / r_mean
    return float(np.sum(p_i[active] * ratio * np.log2(ratio)))


def shuffle_null_si(spike_times, frame_ms, x, y, occupancy_time, x_edges, y_edges,
                     p_i, min_occupancy_s, n_permutations):
    """
    Circular-shuffle permutation test: repeatedly shifts all spike times by a
    random amount (wrapping around the recording duration), recomputes SI
    against the *unshifted* trajectory, and collects the resulting null
    distribution of SI values.
    """
    t_min, t_max = frame_ms.min(), frame_ms.max()
    recording_duration = t_max - t_min

    null_si = np.zeros(n_permutations)
    for perm in range(n_permutations):
        shift = np.random.uniform(0, recording_duration)
        shifted = spike_times + shift
        shifted = np.where(shifted > t_max, shifted - recording_duration, shifted)

        sx, sy = align_spikes(shifted, frame_ms, x, y)
        sh_hist, _, _ = np.histogram2d(sx, sy, bins=[x_edges, y_edges])
        sh_rate = np.divide(
            sh_hist, occupancy_time,
            out=np.full_like(sh_hist, np.nan, dtype=float),
            where=occupancy_time >= min_occupancy_s
        )
        null_si[perm] = skaggs_si(sh_rate.flatten(), p_i)

    return null_si


def ssi_from_null(observed_si, null_si):
    """
    Compare an observed SI to its shuffle null distribution.

    Returns (z_score, p_value_gaussian, p_value_empirical).
    """
    null_mean = np.mean(null_si)
    null_std = np.std(null_si)
    z_score = (observed_si - null_mean) / null_std if null_std > 0 else 0.0
    p_value_gaussian = float(1 - norm.cdf(z_score))
    p_value_empirical = float(np.sum(null_si >= observed_si) / len(null_si))
    return z_score, p_value_gaussian, p_value_empirical


def upsample_rate_map(rate_map, factor=8, order=3):
    """
    Smooth zoom of a rate map for display. NaN regions stay NaN (a bin is
    considered valid in the output only if the upsampled mask is >= 0.5).
    """
    from scipy.ndimage import zoom

    valid_mask = ~np.isnan(rate_map.T)
    rate_filled = np.nan_to_num(rate_map.T, nan=0.0)
    rate_upsampled = zoom(rate_filled, factor, order=order)
    mask_upsampled = zoom(valid_mask.astype(float), factor, order=order)
    rate_upsampled[mask_upsampled < 0.5] = np.nan
    return rate_upsampled
