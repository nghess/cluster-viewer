from pathlib import Path

# Root directory containing all session data.
DEFAULT_DATA_ROOT = Path('D:/data/cbm-odor')

# Subdirectories of the data root, mirrored by <animal>/<session>/.
BONSAI_SUBDIR = 'bonsai'  # unused directly (no video needed here) - kept for parity
EVENTS_SUBDIR = 'events'  # <data_root>/events/<animal>/<session>/events.csv
HC_SUBDIR = 'hc-ks'        # <data_root>/hc-ks/<animal>/<session>/ (hippocampus)
OB_SUBDIR = 'ob-ks'        # <data_root>/ob-ks/<animal>/<session>/ (olfactory bulb)

# Brain regions to load clusters from: (key, subdir, display label).
REGIONS = [
    ('hc', HC_SUBDIR, 'HC'),
    ('ob', OB_SUBDIR, 'OB'),
]

EVENTS_GLOB = 'events.csv'

# events.csv column prefix (<POSITION_POINT>_x/_y) used for place-field
# spatial binning - the sleap-tracked centroid, not Bonsai's real-time
# bonsai_centroid (that one's for cbm-viewer's video overlay instead).
POSITION_POINT = 'nose'

# events.csv column holding position timestamps, in the *ephys* clock -
# not 'bonsai_ts' (Bonsai's wall clock). spike_times.npy is recorded on the
# ephys/Kilosort sample clock, so alignment must go through this column.
TIMESTAMP_COLUMN = 'timestamp_ms'

# Kilosort spike_times.npy sample rate (Hz), for converting samples -> ms.
SAMPLING_RATE_HZ = 30000.0

# Clusters below this whole-session spike count are excluded before place
# fields are even computed (too few spikes for a meaningful rate map).
MIN_SPIKES = 250

# Cluster labels to include (from cluster_info.tsv's 'group', falling back
# to 'KSLabel' when a cluster hasn't been manually reviewed in phy).
# 'noise' is always excluded.
INCLUDE_LABELS = ('good', 'mua')

# Known physical arena extent (pixels), used as fixed spatial bin edges.
# Tracked positions occasionally jitter slightly outside this envelope
# (SLEAP noise near the arena walls), but bins should tile the actual
# arena rather than stretch to whatever the noisy tracked extent happens
# to be for a given session.
ARENA_X_RANGE_PX = (0, 888)
ARENA_Y_RANGE_PX = (0, 1968)

# Spatial bin size in pixels for occupancy/rate maps.
BIN_SIZE_PX = 129 # 129/2.54cm 50.7/cm

# Spatial bins visited less than this many seconds are excluded (NaN in the rate map).
MIN_OCCUPANCY_S = 0.1

# Shuffle-null permutations used for the SSI (z-score) significance test.
# Lower than a typical one-off analysis (e.g. 1000) because this runs
# eagerly for every cluster in a session as soon as it's loaded - see the
# benchmark in the project plan. Bump this if you want tighter p-values and
# don't mind a slower session load.
N_PERMUTATIONS = 500

# Zoom factor for the display-only upsampled/smoothed rate map.
UPSAMPLE_FACTOR = 8
