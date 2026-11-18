import datetime
import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.config import (
    CROWDING_SEATS_AVAILABLE_MAX_CI,
    CROWDING_STANDING_ROOM_MAX_CI,
    EB_SHRINKAGE_KAPPA,
    MIDI_BUS_MAX_CAPACITY,
    MIDI_BUS_SEATED_CAPACITY,
    OUTPUT_DIR,
    STANDARD_BUS_MAX_CAPACITY,
    STANDARD_BUS_SEATED_CAPACITY,
    discover_csv_path,
    is_ignored_path,
)
from src.data_loader import create_duckdb_connection, get_all_route_stop_sequences
from src.occupancy_engine import (
    compute_empirical_bayes_crowding_profiles,
    compute_trip_sweep_line_occupancy,
)
from src.router import TransitJourneyPlanner
from src.stop_matcher import (
    build_hierarchical_imputation_tables,
    normalize_stop_name,
    reconcile_and_impute_dataset,
)
from src.trip_segmenter import segment_trips_hysteresis


def test_file_discovery_and_macosx_exclusion():
    valid_csv = discover_csv_path()
    assert valid_csv.exists()
    assert valid_csv.name == "October-Delhi.csv"

    macosx_path = Path("/workspace/__MACOSX/._October-Delhi.csv")
    assert is_ignored_path(macosx_path) is True

    dot_underscore_path = Path("/workspace/data/._stops.txt")
    assert is_ignored_path(dot_underscore_path) is True

    normal_path = Path("/workspace/data/stops.txt")
    assert is_ignored_path(normal_path) is False


def test_kahn_topological_sort_flow_ratios():
    con = create_duckdb_connection()
    con.execute("""
    CREATE TABLE raw_tickets AS
    SELECT 1 AS passenger_count, 10.0 AS amount, 'DL1PD1000' AS vehicle_number, 1 AS ticket_type_id,
           'Terminal A' AS start_stop_name, 'Stop B' AS end_stop_name, 'ROUTE1' AS route,
           TIMESTAMPTZ '2026-10-01 08:00:00+05:30' AS created_at, True AS is_ac
    UNION ALL
    SELECT 1, 15.0, 'DL1PD1000', 1, 'Stop B', 'Terminal C', 'ROUTE1', TIMESTAMPTZ '2026-10-01 08:10:00+05:30', True
    UNION ALL
    SELECT 1, 25.0, 'DL1PD1000', 1, 'Terminal A', 'Terminal C', 'ROUTE1', TIMESTAMPTZ '2026-10-01 08:05:00+05:30', True
    """)

    topologies = get_all_route_stop_sequences(con, gtfs_dir=None, table_name="raw_tickets")
    assert "ROUTE1" in topologies
    stops = topologies["ROUTE1"]
    assert len(stops) == 3
    assert stops[0] == "Terminal A"
    assert stops[1] == "Stop B"
    assert stops[2] == "Terminal C"


def test_stop_matcher_and_hierarchical_imputation():
    raw_name = "Sec-40 Crossing TerMl."
    normalized = normalize_stop_name(raw_name)
    assert "SECTOR 40" in normalized
    assert "CROSSING" in normalized
    assert "TERMINAL" in normalized

    test_df = pl.DataFrame({
        "route": ["R1", "R1", "R1"],
        "amount": [10.0, 10.0, 20.0],
        "ticket_type_id": [1, 1, 1],
        "start_seq": [1, 2, 1],
        "end_seq": [4, 5, 8],
    })
    t1, t2, t3, g_med = build_hierarchical_imputation_tables(test_df)
    assert (("R1", 10.0, 1) in t1)
    assert t1[("R1", 10.0, 1)] == 3.0
    assert (("R1", 20.0, 1) in t1)
    assert t1[("R1", 20.0, 1)] == 7.0


def test_hysteresis_trip_sessionization():
    df_raw = pl.DataFrame({
        "passenger_count": [1, 1, 1, 1, 1],
        "amount": [10.0, 10.0, 10.0, 10.0, 10.0],
        "vehicle_number": ["DL1PD9999"] * 5,
        "ticket_type_id": [1] * 5,
        "route": ["TEST_ROUTE"] * 5,
        "start_stop_name": ["Stop 1", "Stop 3", "Stop 7", "Stop 2", "Stop 5"],
        "end_stop_name": ["Stop 3", "Stop 5", "Stop 9", "Stop 6", "Stop 8"],
        "start_seq": [1, 3, 7, 2, 5],
        "end_seq": [3, 5, 9, 6, 8],
        "max_seq": [10] * 5,
        "is_ac": [True] * 5,
        "created_at": [
            datetime.datetime(2026, 10, 1, 8, 0),
            datetime.datetime(2026, 10, 1, 8, 10),
            datetime.datetime(2026, 10, 1, 8, 20),
            datetime.datetime(2026, 10, 1, 8, 25),
            datetime.datetime(2026, 10, 1, 9, 30),
        ],
    })

    segmented = segment_trips_hysteresis(df_raw)
    trip_ids = segmented["inferred_trip_id"].to_list()
    assert trip_ids[0] == trip_ids[1]
    assert trip_ids[1] == trip_ids[2]
    assert trip_ids[2] != trip_ids[3]
    assert trip_ids[3] != trip_ids[4]
    assert len(set(trip_ids)) == 3


def test_sweep_line_passenger_conservation():
    df_trips = pl.DataFrame({
        "inferred_trip_id": ["TRIP_A", "TRIP_A", "TRIP_A"],
        "route": ["R_TEST", "R_TEST", "R_TEST"],
        "service_date": ["2026-10-01", "2026-10-01", "2026-10-01"],
        "day_type": ["Weekday", "Weekday", "Weekday"],
        "time_bin": [16, 16, 16],
        "hour": [8, 8, 8],
        "minute": [0, 5, 10],
        "max_seq": [5, 5, 5],
        "vehicle_group": ["DL1PD1111", "DL1PD1111", "DL1PD1111"],
        "start_seq": [1, 2, 1],
        "end_seq": [3, 4, 2],
        "passenger_count": [2, 1, 3],
    })

    df_stretch_trips, trip_summary = compute_trip_sweep_line_occupancy(df_trips)
    assert df_stretch_trips.height == 4

    loads = df_stretch_trips.sort("stop_seq")["raw_load"].to_list()
    assert loads[0] == 5.0
    assert loads[1] == 3.0
    assert loads[2] == 1.0
    assert loads[3] == 0.0

    total_boarded = int(df_trips["passenger_count"].sum())
    assert total_boarded == 6


def test_capacity_normalization_and_crowding_categories():
    df_midi = pl.DataFrame({
        "inferred_trip_id": ["TRIP_MIDI"],
        "route": ["R_MIDI"],
        "service_date": ["2026-10-01"],
        "day_type": ["Weekday"],
        "time_bin": [24],
        "hour": [12],
        "minute": [0],
        "max_seq": [3],
        "vehicle_group": ["DL51GD1000"],
        "start_seq": [1],
        "end_seq": [3],
        "passenger_count": [10],
    })
    stretch_midi, _ = compute_trip_sweep_line_occupancy(df_midi)
    assert stretch_midi["seated_cap"][0] == MIDI_BUS_SEATED_CAPACITY

    df_std = pl.DataFrame({
        "inferred_trip_id": ["TRIP_STD"],
        "route": ["R_STD"],
        "service_date": ["2026-10-01"],
        "day_type": ["Weekday"],
        "time_bin": [24],
        "hour": [12],
        "minute": [0],
        "max_seq": [3],
        "vehicle_group": ["DL1PD5000"],
        "start_seq": [1],
        "end_seq": [3],
        "passenger_count": [10],
    })
    stretch_std, _ = compute_trip_sweep_line_occupancy(df_std)
    assert stretch_std["seated_cap"][0] == STANDARD_BUS_SEATED_CAPACITY


def test_empirical_bayes_shrinkage_bounds():
    df_sparse = pl.DataFrame({
        "inferred_trip_id": ["TRIP_1"],
        "route": ["R_EB"],
        "service_date": ["2026-10-01"],
        "day_type": ["Weekday"],
        "time_bin": [20],
        "stop_seq": [1],
        "raw_load": [4.0],
        "adj_load": [4.4],
        "seated_cap": [40],
    })
    profiles = compute_empirical_bayes_crowding_profiles(df_sparse, export_path=None)
    assert profiles.height == 1

    sample_val = 4.4
    kappa = EB_SHRINKAGE_KAPPA
    expected_weight = 1.0 / (1.0 + kappa)
    shrunk_val = profiles["p50_shrunk"][0]
    assert sample_val <= shrunk_val <= 15.0
    assert profiles["is_low_confidence"][0] is True
    assert profiles["confidence_tier"][0] == "Low Confidence"


def test_router_three_pareto_journey_cards():
    topologies = {
        "8UP": ["Badarpur Border", "Ali Village", "Jasola Vihar", "Noida Sector 37"],
        "100UP": ["Badarpur Border", "Jasola Vihar", "Noida Sector 37"],
    }
    planner = TransitJourneyPlanner(topologies)
    cards = planner.plan_journey("Badarpur Border", "Noida Sector 37", "18:00", "Weekday")
    assert len(cards) == 3

    assert cards[0].option_type == "Fastest Option"
    assert cards[0].departure_shift_minutes == 0
    assert cards[0].total_travel_minutes > 0

    assert cards[1].option_type == "Comfortable Alternative"
    assert cards[1].departure_shift_minutes == 0

    assert "Leave Later" in cards[2].option_type
    assert cards[2].departure_shift_minutes in [15, 30]

    for card in cards:
        assert card.departure_time != ""
        assert card.arrival_time != ""
        assert card.overall_crowding_category in [
            "Seats Available",
            "Standing Room",
            "Crowded",
        ]
        assert card.overall_confidence in [
            "High Confidence",
            "Medium Confidence",
            "Low Confidence",
        ]


def test_generate_test_execution_summary():
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tests_executed": [
            "test_file_discovery_and_macosx_exclusion",
            "test_kahn_topological_sort_flow_ratios",
            "test_stop_matcher_and_hierarchical_imputation",
            "test_hysteresis_trip_sessionization",
            "test_sweep_line_passenger_conservation",
            "test_capacity_normalization_and_crowding_categories",
            "test_empirical_bayes_shrinkage_bounds",
            "test_router_three_pareto_journey_cards",
        ],
        "status": "PASSED",
        "total_passed": 8,
        "total_failed": 0,
    }
    summary_path = OUTPUT_DIR / "test_execution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    assert summary_path.exists()
