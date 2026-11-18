from pathlib import Path
from typing import Optional, List

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_BUS_SEATED_CAPACITY = 40
STANDARD_BUS_MAX_CAPACITY = 65
MIDI_BUS_SEATED_CAPACITY = 28
MIDI_BUS_MAX_CAPACITY = 43
MIDI_BUS_PREFIXES = ("DL51GD", "DL1PE")

CROWDING_SEATS_AVAILABLE_MAX_CI = 0.85
CROWDING_STANDING_ROOM_MAX_CI = 1.35

PEAK_HOURS = {8, 9, 10, 17, 18, 19}
PASS_HOLDER_PEAK_MULTIPLIER = 1.18
PASS_HOLDER_OFFPEAK_MULTIPLIER = 1.10

HYSTERESIS_MAX_TIME_GAP_MINUTES = 45.0
HYSTERESIS_TURNAROUND_SEQ_DELTA = -3

STOP_MATCH_SCORE_THRESHOLD = 82.0

EB_SHRINKAGE_KAPPA = 5.0
LOW_CONFIDENCE_MIN_TRIPS = 3
MEDIUM_CONFIDENCE_MIN_TRIPS = 10

TIME_BIN_MINUTES = 30
TIME_BINS_PER_DAY = 48

ROUTING_TRANSFER_PENALTY_MINUTES = 15.0
ROUTING_TRANSFER_WALK_BUFFER_MINUTES = 8.0
ROUTING_DEFAULT_INTER_STOP_MINUTES = 2.5

ROUTING_FASTEST_BETA = 0.0
ROUTING_FASTEST_DELTA = 1.0

ROUTING_COMFORT_BETA = 2.5
ROUTING_COMFORT_DELTA = 1.0

ROUTING_SHIFTED_BETA = 3.0
ROUTING_SHIFTED_DELTA = 0.2

CROWDING_PROFILE_PARQUET = OUTPUT_DIR / "crowding_profiles.parquet"
VALIDATION_SUMMARY_JSON = OUTPUT_DIR / "validation_summary.json"


def is_ignored_path(path: Path) -> bool:
    for part in path.parts:
        if part == "__MACOSX" or part.startswith("._"):
            return True
    return False


def discover_csv_path(root_dir: Optional[Path] = None) -> Path:
    search_root = root_dir if root_dir is not None else BASE_DIR
    for item in search_root.rglob("October-Delhi.csv"):
        if not is_ignored_path(item):
            return item
    parent_candidate = search_root.parent / "October-Delhi.csv"
    if parent_candidate.exists() and not is_ignored_path(parent_candidate):
        return parent_candidate
    for item in search_root.rglob("*.csv"):
        if not is_ignored_path(item) and "october" in item.name.lower():
            return item
    raise FileNotFoundError(f"Could not locate October-Delhi.csv in {search_root}")


def discover_gtfs_dir(root_dir: Optional[Path] = None) -> Optional[Path]:
    search_root = root_dir if root_dir is not None else BASE_DIR
    for item in search_root.rglob("stops.txt"):
        if not is_ignored_path(item):
            return item.parent
    parent_gtfs = search_root.parent / "stops.txt"
    if parent_gtfs.exists() and not is_ignored_path(parent_gtfs):
        return parent_gtfs.parent
    return None
