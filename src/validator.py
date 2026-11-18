import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import polars as pl

from src.config import (
    CROWDING_SEATS_AVAILABLE_MAX_CI,
    CROWDING_STANDING_ROOM_MAX_CI,
    VALIDATION_SUMMARY_JSON,
)
from src.occupancy_engine import compute_empirical_bayes_crowding_profiles


def evaluate_out_of_time_holdout(
    df_stretch_trips: pl.DataFrame,
    split_date: str = "2026-10-21",
) -> Dict[str, float]:
    train_trips = df_stretch_trips.filter(pl.col("service_date") <= split_date)
    test_trips = df_stretch_trips.filter(pl.col("service_date") > split_date)

    if train_trips.height == 0 or test_trips.height == 0:
        return {
            "r_squared": 0.0,
            "mae": 0.0,
            "category_accuracy": 0.0,
            "evaluated_segments": 0,
        }

    train_profiles = compute_empirical_bayes_crowding_profiles(train_trips, export_path=None)

    test_actuals = (
        test_trips.group_by(["route", "stop_seq", "day_type", "time_bin"])
        .agg([
            pl.col("adj_load").median().alias("actual_load"),
            pl.col("seated_cap").mean().alias("seated_cap"),
        ])
    )

    merged = test_actuals.join(
        train_profiles.select(["route", "stop_seq", "day_type", "time_bin", "p50_shrunk"]),
        on=["route", "stop_seq", "day_type", "time_bin"],
        how="inner",
    )

    if merged.height == 0:
        return {
            "r_squared": 0.0,
            "mae": 0.0,
            "category_accuracy": 0.0,
            "evaluated_segments": 0,
        }

    y_actual = merged["actual_load"].to_numpy()
    y_pred = merged["p50_shrunk"].to_numpy()
    seated_caps = merged["seated_cap"].to_numpy()

    mae = float(np.mean(np.abs(y_actual - y_pred)))
    total_ss = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    residual_ss = float(np.sum((y_actual - y_pred) ** 2))
    r_squared = float(1.0 - (residual_ss / total_ss)) if total_ss > 1e-6 else 0.0

    actual_ci = y_actual / seated_caps
    pred_ci = y_pred / seated_caps

    actual_cats = np.where(
        actual_ci <= CROWDING_SEATS_AVAILABLE_MAX_CI,
        0,
        np.where(actual_ci <= CROWDING_STANDING_ROOM_MAX_CI, 1, 2),
    )
    pred_cats = np.where(
        pred_ci <= CROWDING_SEATS_AVAILABLE_MAX_CI,
        0,
        np.where(pred_ci <= CROWDING_STANDING_ROOM_MAX_CI, 1, 2),
    )
    cat_accuracy = float(np.mean(actual_cats == pred_cats))

    return {
        "r_squared": round(r_squared, 4),
        "mae": round(mae, 4),
        "category_accuracy": round(cat_accuracy * 100.0, 2),
        "evaluated_segments": int(merged.height),
    }


def verify_diurnal_and_weekly_patterns(
    df_stretch_trips: pl.DataFrame,
) -> Dict[str, Any]:
    df_with_hour = df_stretch_trips.with_columns(
        ((pl.col("time_bin") * 30) // 60).cast(pl.Int32).alias("hour_of_day")
    )

    hourly_loads = (
        df_with_hour.group_by("hour_of_day")
        .agg(pl.col("adj_load").mean().alias("mean_load"))
        .sort("hour_of_day")
    )
    hourly_dict = {
        int(row[0]): float(row[1]) for row in hourly_loads.iter_rows()
    }

    morning_peak_hours = [8, 9, 10]
    midday_hours = [12, 13, 14]
    evening_peak_hours = [17, 18, 19]

    morning_avg = float(
        np.mean([hourly_dict.get(h, 0.0) for h in morning_peak_hours])
    )
    midday_avg = float(
        np.mean([hourly_dict.get(h, 0.0) for h in midday_hours])
    )
    evening_avg = float(
        np.mean([hourly_dict.get(h, 0.0) for h in evening_peak_hours])
    )

    is_bimodal = (morning_avg > midday_avg) and (evening_avg > midday_avg)

    weekday_trips = df_stretch_trips.filter(pl.col("day_type") == "Weekday")
    weekend_trips = df_stretch_trips.filter(pl.col("day_type") == "Weekend")

    weekday_distinct_days = max(1, weekday_trips["service_date"].n_unique())
    weekend_distinct_days = max(1, weekend_trips["service_date"].n_unique())

    weekday_daily_load = (
        float(weekday_trips["adj_load"].sum()) / weekday_distinct_days
        if weekday_trips.height > 0
        else 0.0
    )
    weekend_daily_load = (
        float(weekend_trips["adj_load"].sum()) / weekend_distinct_days
        if weekend_trips.height > 0
        else 0.0
    )

    weekday_to_weekend_ratio = (
        round(weekday_daily_load / weekend_daily_load, 2)
        if weekend_daily_load > 0
        else 1.0
    )

    return {
        "morning_peak_avg_load": round(morning_avg, 2),
        "midday_offpeak_avg_load": round(midday_avg, 2),
        "evening_peak_avg_load": round(evening_avg, 2),
        "is_bimodal_distribution": bool(is_bimodal),
        "weekday_daily_load": round(weekday_daily_load, 2),
        "weekend_daily_load": round(weekend_daily_load, 2),
        "weekday_to_weekend_ratio": weekday_to_weekend_ratio,
        "hourly_load_profile": {str(k): round(v, 2) for k, v in hourly_dict.items()},
    }


def find_top_overloaded_corridors(
    crowding_profiles: pl.DataFrame,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    peak_bins = [16, 17, 18, 19, 20, 21, 34, 35, 36, 37, 38, 39]
    peak_df = crowding_profiles.filter(pl.col("time_bin").is_in(peak_bins))

    if peak_df.height == 0:
        peak_df = crowding_profiles

    sorted_df = (
        peak_df.sort("crowding_index_p85", descending=True)
        .head(top_n)
        .select([
            "route",
            "stop_seq",
            "day_type",
            "time_bin",
            "crowding_index_p85",
            "crowding_index_p50",
            "crowding_category",
            "trip_count",
        ])
    )

    records: List[Dict[str, Any]] = []
    for row in sorted_df.iter_rows(named=True):
        records.append({
            "route": str(row["route"]),
            "stop_seq": int(row["stop_seq"]),
            "day_type": str(row["day_type"]),
            "time_bin": int(row["time_bin"]),
            "p85_crowding_index": round(float(row["crowding_index_p85"]), 2),
            "p50_crowding_index": round(float(row["crowding_index_p50"]), 2),
            "crowding_category": str(row["crowding_category"]),
            "trip_count": int(row["trip_count"]),
        })

    return records


def run_full_validation(
    df_stretch_trips: pl.DataFrame,
    crowding_profiles: pl.DataFrame,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    holdout_metrics = evaluate_out_of_time_holdout(df_stretch_trips)
    diurnal_metrics = verify_diurnal_and_weekly_patterns(df_stretch_trips)
    top_corridors = find_top_overloaded_corridors(crowding_profiles)

    summary: Dict[str, Any] = {
        "out_of_time_holdout": holdout_metrics,
        "diurnal_and_weekly_patterns": diurnal_metrics,
        "top_overloaded_corridors": top_corridors,
    }

    target_file = output_path if output_path is not None else VALIDATION_SUMMARY_JSON
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
