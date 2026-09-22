# cluster-viewer

Desktop viewer for cbm-odor sessions' spike-sorted clusters: loads a
session's hippocampus (`hc-ks`) and olfactory-bulb (`ob-ks`) Kilosort/phy
output, computes each cluster's spatial place field, and shows them in a
sortable table (by spatial information and shuffle-corrected significance).

## Setup

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```
.venv\Scripts\python.exe src\cluster_viewer\main.py
```

Pick an animal/session and click Load. Sessions are discovered from
`<DATA_ROOT>/<EVENTS_SUBDIR>/<animal>/<session>/events.csv` (position
tracking) paired with `<DATA_ROOT>/<HC_SUBDIR|OB_SUBDIR>/<animal>/<session>/`
(Kilosort/phy output for each brain region) - a session needs events.csv
and at least one region present. Data root and subdirectory names are set
in `src/cluster_viewer/config.py`.

## What it shows

- On Load, every `good`/`mua` cluster (per phy's `group` label, falling
  back to Kilosort's automatic `KSLabel` for clusters nobody's manually
  reviewed; `noise` is always excluded) from both regions has its place
  field computed: occupancy map, spatial firing-rate map, Skaggs spatial
  information (SI, bits/spike), and a circular-shuffle null distribution
  used to derive SSI (a z-score) and a p-value. This runs on a background
  thread with a progress label, since a session can have dozens of
  clusters and each needs its own shuffle test.
- A sortable table (Cluster, Region, Label, Channel, N Spikes, SI, SSI, p) -
  click a row to plot that cluster's trajectory+spike-position overlay and
  its rate-map heatmap.

Position tracking uses `events.csv`'s sleap-tracked `centroid_x/y` and its
`timestamp_ms` column - the *ephys*-clock timestamp (not `bonsai_ts`, which
is Bonsai's own wall clock), since spike times from `spike_times.npy` are on
the ephys/Kilosort sample clock and need to be compared against something
in that same clock domain.

## Known limitations / next steps

- The shuffle-null test defaults to 500 permutations/cluster (vs. a typical
  one-off analysis's 1000+) to keep per-session load times reasonable when
  computing place fields for every cluster eagerly; raise
  `config.N_PERMUTATIONS` for tighter p-values at the cost of load time.
- Spatial bins use a fixed pixel size (`config.BIN_SIZE_PX`, ~1cm) with
  arena bounds auto-derived from the tracked positions, rather than a fixed
  bin count or a hand-set arena rectangle.
- No cross-region or cross-session comparison views; one session's clusters
  at a time.
# cluster-viewer
