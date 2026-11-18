from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl

from src.config import (
    CROWDING_PROFILE_PARQUET,
    CROWDING_SEATS_AVAILABLE_MAX_CI,
    CROWDING_STANDING_ROOM_MAX_CI,
    EB_SHRINKAGE_KAPPA,
    LOW_CONFIDENCE_MIN_TRIPS,
    MEDIUM_CONFIDENCE_MIN_TRIPS,
    MIDI_BUS_MAX_CAPACITY,
    MIDI_BUS_PREFIXES,
    MIDI_BUS_SEATED_CAPACITY,
    PASS_HOLDER_OFFPEAK_MULTIPLIER,
    PASS_HOLDER_PEAK_MULTIPLIER,
    PEAK_HOURS,
    STANDARD_BUS_MAX_CAPACITY,
    STANDARD_BUS_SEATED_CAPACITY,
)


def compute_trip_sweep_line_occupancy(
    df_trips: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    trip_summary = (
        df_trips.group_by("inferred_trip_id")
        .agg([
            pl.col("route").first().alias("route"),
            pl.col("service_date").first().alias("service_date"),
            pl.col("day_type").first().alias("day_type"),
            pl.col("time_bin").first().alias("time_bin"),
            pl.col("hour").first().alias("hour"),
            pl.col("max_seq").max().alias("max_seq"),
            pl.col("vehicle_group").first().alias("vehicle_group"),
            pl.col("start_seq").alias("start_seqs"),
            pl.col("end_seq").alias("end_seqs"),
            pl.col("passenger_count").alias("passenger_counts"),
            pl.col("passenger_count").sum().alias("total_boardings"),
        ])
    )

    routes = trip_summary["route"].to_list()
    service_dates = trip_summary["service_date"].to_list()
    day_types = trip_summary["day_type"].to_list()
    time_bins = trip_summary["time_bin"].to_list()
    hours = trip_summary["hour"].to_list()
    max_seqs = trip_summary["max_seq"].to_list()
    vehicle_groups = trip_summary["vehicle_group"].to_list()
    start_seq_lists = trip_summary["start_seqs"].to_list()
    end_seq_lists = trip_summary["end_seqs"].to_list()
    passenger_lists = trip_summary["passenger_counts"].to_list()
    inferred_trip_ids = trip_summary["inferred_trip_id"].to_list()

    stretch_trip_ids: List[str] = []
    stretch_routes: List[str] = []
    stretch_dates: List[str] = []
    stretch_day_types: List[str] = []
    stretch_time_bins: List[int] = []
    stretch_seqs: List[int] = []
    stretch_raw_loads: List[float] = []
    stretch_adj_loads: List[float] = []
    stretch_seated_caps: List[int] = []

    for idx in range(len(routes)):
        route = routes[idx]
        s_date = service_dates[idx]
        d_type = day_types[idx]
        t_bin = int(time_bins[idx])
        hr = int(hours[idx])
        m_seq = int(max_seqs[idx])
        v_grp = vehicle_groups[idx]
        t_id = inferred_trip_ids[idx]

        multiplier = (
            PASS_HOLDER_PEAK_MULTIPLIER
            if hr in PEAK_HOURS
            else PASS_HOLDER_OFFPEAK_MULTIPLIER
        )

        is_midi = any(v_grp.startswith(prefix) for prefix in MIDI_BUS_PREFIXES)
        seated_cap = MIDI_BUS_SEATED_CAPACITY if is_midi else STANDARD_BUS_SEATED_CAPACITY

        diff_len = m_seq + 2
        diff = np.zeros(diff_len, dtype=np.float64)

        s_arr = np.asarray(start_seq_lists[idx], dtype=np.int64)
        e_arr = np.asarray(end_seq_lists[idx], dtype=np.int64)
        p_arr = np.asarray(passenger_lists[idx], dtype=np.float64)

        np.add.at(diff, s_arr, p_arr)
        np.add.at(diff, e_arr, -p_arr)

        prefix_loads = np.cumsum(diff)

        num_stretches = max(1, m_seq - 1)
        for s_idx in range(1, num_stretches + 1):
            raw_val = max(0.0, float(prefix_loads[s_idx]))
            adj_val = raw_val * multiplier

            stretch_trip_ids.append(t_id)
            stretch_routes.append(route)
            stretch_dates.append(s_date)
            stretch_day_types.append(d_type)
            stretch_time_bins.append(t_bin)
            stretch_seqs.append(s_idx)
            stretch_raw_loads.append(raw_val)
            stretch_adj_loads.append(adj_val)
            stretch_seated_caps.append(seated_cap)

    df_stretch_trips = pl.DataFrame({
        "inferred_trip_id": stretch_trip_ids,
        "route": stretch_routes,
        "service_date": stretch_dates,
        "day_type": stretch_day_types,
        "time_bin": stretch_time_bins,
        "stop_seq": stretch_seqs,
        "raw_load": stretch_raw_loads,
        "adj_load": stretch_adj_loads,
        "seated_cap": stretch_seated_caps,
    })

    return df_stretch_trips, trip_summary


def compute_empirical_bayes_crowding_profiles(
    df_stretch_trips: pl.DataFrame,
    export_path: Optional[Path] = None,
) -> pl.DataFrame:
    group_cols = ["route", "stop_seq", "day_type", "time_bin"]

    stretch_aggregates = (
        df_stretch_trips.group_by(group_cols)
        .agg([
            pl.len().alias("trip_count"),
            pl.col("adj_load").median().alias("p50_sample"),
            pl.col("adj_load").quantile(0.85, interpolation="linear").alias("p85_sample"),
            pl.col("adj_load").mean().alias("mean_sample"),
            pl.col("raw_load").median().alias("p50_raw"),
            pl.col("seated_cap").mean().alias("seated_cap"),
        ])
    )

    route_time_priors = (
        df_stretch_trips.group_by(["route", "day_type", "time_bin"])
        .agg([
            pl.col("adj_load").median().alias("p50_route_prior"),
            pl.col("adj_load").quantile(0.85, interpolation="linear").alias("p85_route_prior"),
        ])
    )

    route_global_priors = (
        df_stretch_trips.group_by("route")
        .agg([
            pl.col("adj_load").median().alias("p50_global_prior"),
            pl.col("adj_load").quantile(0.85, interpolation="linear").alias("p85_global_prior"),
        ])
    )

    joined = stretch_aggregates.join(
        route_time_priors,
        on=["route", "day_type", "time_bin"],
        how="left",
    ).join(
        route_global_priors,
        on="route",
        how="left",
    )

    global_p50 = float(df_stretch_trips["adj_load"].median() or 1.1)
    global_p85 = float(
        df_stretch_trips["adj_load"].quantile(0.85, interpolation="linear") or 2.2
    )

    joined = joined.with_columns([
        pl.col("p50_route_prior").fill_null(pl.col("p50_global_prior")).fill_null(global_p50),
        pl.col("p85_route_prior").fill_null(pl.col("p85_global_prior")).fill_null(global_p85),
    ])

    joined = joined.with_columns([
        (pl.col("trip_count") / (pl.col("trip_count") + EB_SHRINKAGE_KAPPA)).alias("confidence_weight"),
    ])

    joined = joined.with_columns([
        (
            pl.col("confidence_weight") * pl.col("p50_sample")
            + (1.0 - pl.col("confidence_weight")) * pl.col("p50_route_prior")
        ).alias("p50_shrunk"),
        (
            pl.col("confidence_weight") * pl.col("p85_sample")
            + (1.0 - pl.col("confidence_weight")) * pl.col("p85_route_prior")
        ).alias("p85_shrunk"),
    ])

    joined = joined.with_columns([
        (pl.col("p50_shrunk") / pl.col("seated_cap")).alias("crowding_index_p50"),
        (pl.col("p85_shrunk") / pl.col("seated_cap")).alias("crowding_index_p85"),
    ])

    joined = joined.with_columns([
        pl.when(pl.col("crowding_index_p50") <= CROWDING_SEATS_AVAILABLE_MAX_CI)
        .then(pl.lit("Seats Available"))
        .when(pl.col("crowding_index_p50") <= CROWDING_STANDING_ROOM_MAX_CI)
        .then(pl.lit("Standing Room"))
        .otherwise(pl.lit("Crowded"))
        .alias("crowding_category"),
        pl.when(pl.col("trip_count") < LOW_CONFIDENCE_MIN_TRIPS)
        .then(pl.lit("Low Confidence"))
        .when(pl.col("trip_count") < MEDIUM_CONFIDENCE_MIN_TRIPS)
        .then(pl.lit("Medium Confidence"))
        .otherwise(pl.lit("High Confidence"))
        .alias("confidence_tier"),
        (pl.col("trip_count") < LOW_CONFIDENCE_MIN_TRIPS).alias("is_low_confidence"),
    ])

    output_path = export_path if export_path is not None else CROWDING_PROFILE_PARQUET
    joined.write_parquet(output_path)

    return joined
