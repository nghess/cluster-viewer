"""Computes place fields for every cluster in a loaded session on a
background thread, so a session with many clusters (each needing a
shuffle-null permutation test) doesn't freeze the UI."""
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import config, place_field
from .session_data import ClusterSessionData


class PlaceFieldWorker(QThread):
    progress = Signal(int, int)  # (done, total)
    finished_ok = Signal(list)   # list[ClusterInfo], place fields filled in

    def __init__(self, session: ClusterSessionData, parent=None):
        super().__init__(parent)
        self.session = session

    def run(self):
        session = self.session
        clusters = session.clusters
        total = len(clusters)

        for i, cluster in enumerate(clusters):
            spike_times = cluster.spike_times_ms
            # restrict to the tracked time window
            spike_times = spike_times[(spike_times >= session.frame_ms[0]) &
                                       (spike_times <= session.frame_ms[-1])]

            spike_x, spike_y = place_field.align_spikes(
                spike_times, session.frame_ms, session.x, session.y)
            rate_map, spike_hist = place_field.compute_rate_map(
                spike_x, spike_y, session.occupancy_time,
                session.x_edges, session.y_edges, config.MIN_OCCUPANCY_S)

            si = place_field.skaggs_si(rate_map.flatten(), session.p_i)
            null_si = place_field.shuffle_null_si(
                spike_times, session.frame_ms, session.x, session.y,
                session.occupancy_time, session.x_edges, session.y_edges,
                session.p_i, config.MIN_OCCUPANCY_S, config.N_PERMUTATIONS)
            ssi, p_gaussian, p_empirical = place_field.ssi_from_null(si, null_si)

            cluster.si = si
            cluster.ssi = ssi
            cluster.p_value = p_empirical
            cluster.p_value_gaussian = p_gaussian
            cluster.rate_map = rate_map
            cluster.rate_map_upsampled = place_field.upsample_rate_map(
                rate_map, factor=config.UPSAMPLE_FACTOR)
            cluster.spike_hist = spike_hist
            cluster.null_si = null_si
            cluster.spike_x = spike_x
            cluster.spike_y = spike_y

            self.progress.emit(i + 1, total)

        self.finished_ok.emit(clusters)
