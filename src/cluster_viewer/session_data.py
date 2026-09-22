"""Discovers sessions and loads one session's position tracking + Kilosort/phy
cluster data (both brain regions). Place fields themselves are computed by
place_field_worker.py, since that's slow enough to want a background thread.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, place_field


@dataclass
class ClusterInfo:
    cluster_id: int      # original Kilosort/phy cluster id
    region: str            # 'hc' or 'ob'
    label: str              # 'good' or 'mua'
    best_channel: int | None
    n_spikes: int           # whole-session spike count
    spike_times_ms: np.ndarray  # ephys-clock ms, whole session (not yet window-restricted)

    # filled in by place_field_worker.py:
    si: float = float('nan')
    ssi: float = float('nan')
    p_value: float = float('nan')            # empirical p from the shuffle null (used for sorting)
    p_value_gaussian: float = float('nan')    # Gaussian-fit p (matches reference plot_place_field's panel)
    rate_map: np.ndarray | None = None
    rate_map_upsampled: np.ndarray | None = None
    spike_hist: np.ndarray | None = None      # raw spike-count histogram (reference's panel [0,2])
    null_si: np.ndarray | None = None         # shuffle-null SI distribution (reference's panel [1,2])
    spike_x: np.ndarray | None = None
    spike_y: np.ndarray | None = None


def discover_sessions(data_root: Path):
    """(animal, session, events_path, hc_dir, ob_dir) tuples for every
    session with events.csv and at least one region's spike_times.npy.
    hc_dir/ob_dir are None when that region has no kilosort output here."""
    sessions = []
    events_root = data_root / config.EVENTS_SUBDIR
    for animal_dir in sorted(events_root.iterdir()):
        if not animal_dir.is_dir():
            continue
        for sess_dir in sorted(animal_dir.iterdir()):
            events_path = sess_dir / config.EVENTS_GLOB
            if not events_path.exists():
                continue

            region_dirs = {}
            for key, subdir, _label in config.REGIONS:
                region_dir = data_root / subdir / animal_dir.name / sess_dir.name
                if (region_dir / 'spike_times.npy').exists():
                    region_dirs[key] = region_dir

            if not region_dirs:
                continue
            sessions.append((animal_dir.name, sess_dir.name, events_path,
                              region_dirs.get('hc'), region_dirs.get('ob')))
    return sessions


def _read_label_column(path: Path) -> dict:
    """Read a phy cluster_*.tsv label file into {cluster_id: label}, using
    whichever of 'group'/'KSLabel' is actually the header - cluster_group.tsv
    is headed 'KSLabel' (and lists every cluster) until someone actually
    curates in phy, at which point it's headed 'group' and lists only the
    clusters that were manually reviewed."""
    df = pd.read_csv(path, sep='\t')
    label_col = 'group' if 'group' in df.columns else 'KSLabel'
    return dict(zip(df['cluster_id'].astype(int), df[label_col]))


def load_region_clusters(region_dir: Path, region_key: str) -> list[ClusterInfo]:
    """Load one region's Kilosort/phy output, keeping only clusters labeled
    per config.INCLUDE_LABELS with at least config.MIN_SPIKES spikes.

    Deliberately does not use cluster_info.tsv - that file is only written
    once a session has been manually opened in phy (verified: several
    sessions here never have it). cluster_KSLabel.tsv (automatic, always
    present) is the base label; cluster_group.tsv overrides it per-cluster
    wherever that file is actually headed 'group' (i.e. curation happened),
    so every cluster ends up labeled good/mua/noise with 'noise' only ever
    coming from manual curation.
    """
    spike_times = np.load(region_dir / 'spike_times.npy').flatten()
    spike_clusters = np.load(region_dir / 'spike_clusters.npy').flatten()
    templates = np.load(region_dir / 'templates.npy')

    labels = _read_label_column(region_dir / 'cluster_KSLabel.tsv')
    group_path = region_dir / 'cluster_group.tsv'
    if group_path.exists():
        group_df = pd.read_csv(group_path, sep='\t')
        if 'group' in group_df.columns:
            labels.update(dict(zip(group_df['cluster_id'].astype(int), group_df['group'])))

    clusters = []
    for cluster_id, label in labels.items():
        if label not in config.INCLUDE_LABELS:
            continue

        mask = spike_clusters == cluster_id
        n_spikes = int(mask.sum())
        if n_spikes < config.MIN_SPIKES:
            continue

        cluster_spike_times_ms = spike_times[mask] / (config.SAMPLING_RATE_HZ / 1000.0)

        best_channel = None
        if cluster_id < len(templates):
            template = templates[cluster_id]
            best_channel = int(np.argmax(np.max(np.abs(template), axis=0)))

        clusters.append(ClusterInfo(
            cluster_id=cluster_id, region=region_key, label=label,
            best_channel=best_channel, n_spikes=n_spikes,
            spike_times_ms=cluster_spike_times_ms,
        ))
    return clusters


def _has_phy_curation_file(region_dir: Path) -> bool:
    """cluster_info.tsv is only written once a session has been manually
    opened in phy - its absence means the region's cluster labels are
    Kilosort's automatic KSLabel only, never reviewed by a person."""
    return (region_dir / 'cluster_info.tsv').exists()


class ClusterSessionData:
    def __init__(self, animal: str, session: str, events_path: Path,
                 hc_dir: Path | None, ob_dir: Path | None):
        self.animal = animal
        self.session = session
        self.events_path = events_path

        x_col, y_col = f'{config.POSITION_POINT}_x', f'{config.POSITION_POINT}_y'
        events = pd.read_csv(events_path, usecols=lambda c: c in
                              {config.TIMESTAMP_COLUMN, x_col, y_col})
        events = events.dropna(subset=[config.TIMESTAMP_COLUMN, x_col, y_col])
        events = events.sort_values(config.TIMESTAMP_COLUMN)

        self.frame_ms = events[config.TIMESTAMP_COLUMN].to_numpy(dtype=float)
        self.x = events[x_col].to_numpy(dtype=float)
        self.y = events[y_col].to_numpy(dtype=float)

        # Occupancy/bin edges depend only on position tracking, not spikes,
        # so they're computed once here and shared across every cluster.
        self.x_edges, self.y_edges = place_field.make_bin_edges(self.x, self.y, config.BIN_SIZE_PX)
        self.occupancy_time, _ = place_field.compute_occupancy(
            self.frame_ms, self.x, self.y, self.x_edges, self.y_edges)
        self.p_i = (self.occupancy_time / np.sum(self.occupancy_time)).flatten()

        # Region keys present but missing phy's manually-curated cluster_info.tsv
        # (labels for these come from Kilosort's automatic KSLabel only) -
        # surfaced as a flag in the UI.
        self.uncurated_regions: list[str] = []

        self.clusters: list[ClusterInfo] = []
        for key, region_dir in (('hc', hc_dir), ('ob', ob_dir)):
            if region_dir is None:
                continue
            self.clusters += load_region_clusters(region_dir, key)
            if not _has_phy_curation_file(region_dir):
                self.uncurated_regions.append(key)
