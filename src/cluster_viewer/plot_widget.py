"""Matplotlib-backed display for one cluster's place field: the same
6-panel figure as reference/place_field_basics.py's plot_place_field
(trajectory+spikes, occupancy, spike counts, rate map, upsampled rate map,
SSI null distribution), embedded in the app instead of saved to a file."""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtWidgets import QVBoxLayout, QWidget

from . import config


class PlaceFieldPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._draw_placeholder()

    def _new_grid(self):
        # The figure is cleared and rebuilt from scratch on every redraw
        # (rather than reusing persistent Axes) because matplotlib's
        # colorbar keeps its own companion Axes tied to the image's Axes -
        # manually tracking/removing it across repeated clear()+imshow()
        # calls on reused Axes is fragile (breaks on the 2nd+ redraw).
        self.figure.clear()
        return GridSpec(2, 3, figure=self.figure, hspace=0.45, wspace=0.4)

    def _draw_placeholder(self):
        gs = self._new_grid()
        ax = self.figure.add_subplot(gs[:, :])
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Select a cluster", ha='center', va='center',
                 transform=ax.transAxes, color='gray')
        self.canvas.draw_idle()

    def set_cluster(self, session, cluster, title: str = ""):
        if cluster is None or cluster.rate_map is None:
            self._draw_placeholder()
            return

        fontsize = 9
        extent = [session.x_edges[0], session.x_edges[-1],
                  session.y_edges[0], session.y_edges[-1]]
        gs = self._new_grid()
        fig = self.figure

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(session.x, session.y, 'o', ms=1, alpha=0.1, color='C0')
        ax0.scatter(cluster.spike_x, cluster.spike_y, s=2, color='red', alpha=1, zorder=2)
        ax0.set_xlim(extent[0], extent[1])
        ax0.set_ylim(extent[2], extent[3])
        ax0.set_aspect('equal', adjustable='box')
        ax0.set_xlabel('Head X', fontsize=fontsize)
        ax0.set_ylabel('Head Y', fontsize=fontsize)
        ax0.set_title('Trajectory & Spike Positions', fontsize=fontsize, pad=8)
        ax0.tick_params(labelsize=fontsize - 1)

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(session.occupancy_time.T, origin='lower', aspect='equal',
                          cmap='viridis', extent=extent)
        ax1.set_xlabel('Head X', fontsize=fontsize)
        ax1.set_ylabel('Head Y', fontsize=fontsize)
        ax1.set_title('Occupancy Histogram', fontsize=fontsize, pad=8)
        ax1.tick_params(labelsize=fontsize - 1)
        fig.colorbar(im1, ax=ax1, label='Seconds')

        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(cluster.spike_hist.T, origin='lower', aspect='equal',
                          cmap='hot', extent=extent)
        ax2.set_xlabel('Head X', fontsize=fontsize)
        ax2.set_ylabel('Head Y', fontsize=fontsize)
        ax2.set_title('Spike Count Histogram', fontsize=fontsize, pad=8)
        ax2.tick_params(labelsize=fontsize - 1)
        fig.colorbar(im2, ax=ax2, label='Spike Counts')

        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(cluster.rate_map.T, origin='lower', aspect='equal',
                          cmap='jet', extent=extent)
        ax3.set_xlabel('Head X', fontsize=fontsize)
        ax3.set_ylabel('Head Y', fontsize=fontsize)
        ax3.set_title('Spike Rate Map', fontsize=fontsize, pad=8)
        ax3.tick_params(labelsize=fontsize - 1)
        fig.colorbar(im3, ax=ax3, label='Spikes/Sec')

        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(cluster.rate_map_upsampled, origin='lower', aspect='equal',
                          cmap='jet', extent=extent, interpolation='bilinear')
        ax4.set_xlabel('Head X', fontsize=fontsize)
        ax4.set_ylabel('Head Y', fontsize=fontsize)
        ax4.set_title(f'Spike Rate Map (Upsampled {config.UPSAMPLE_FACTOR}x)', fontsize=fontsize, pad=8)
        ax4.tick_params(labelsize=fontsize - 1)
        fig.colorbar(im4, ax=ax4, label='Spikes/Sec')

        null_si = cluster.null_si
        null_mean, null_std = np.mean(null_si), np.std(null_si)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(null_si, bins=50, alpha=0.7, color='gray', edgecolor='black', label='Null Distribution')
        ax5.axvline(cluster.si, color='red', lw=2, ls='--', label=f'SI = {cluster.si:.3f}')
        ax5.axvline(null_mean, color='blue', lw=1, ls=':', label=f'Null Mean = {null_mean:.3f}')
        ax5.axvline(null_mean + 3 * null_std, color='green', lw=1, ls=':',
                    label=f'3σ = {null_mean + 3 * null_std:.3f}')
        ax5.set_xlabel('Spatial Information (bits/spike)', fontsize=fontsize)
        ax5.set_ylabel('Frequency', fontsize=fontsize)
        ax5.set_title(f'SSI Null Distribution\n(z={cluster.ssi:.2f}, p={cluster.p_value_gaussian:.4f})',
                       fontsize=fontsize, pad=8)
        ax5.tick_params(labelsize=fontsize - 1)
        ax5.legend(fontsize=fontsize - 2)

        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold')
        self.canvas.draw_idle()
