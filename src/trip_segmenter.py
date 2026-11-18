import polars as pl
from src.config import (
    HYSTERESIS_MAX_TIME_GAP_MINUTES,
    HYSTERESIS_TURNAROUND_SEQ_DELTA,
    TIME_BIN_MINUTES,
)


def segment_trips_hysteresis(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("created_at").dt.date().cast(pl.String).alias("service_date"),
        pl.col("created_at").dt.hour().cast(pl.Int32).alias("hour"),
        pl.col("created_at").dt.minute().cast(pl.Int32).alias("minute"),
        pl.col("created_at").dt.weekday().alias("weekday_num"),
        pl.when(pl.col("vehicle_number").is_null() | (pl.col("vehicle_number") == "NULL") | (pl.col("vehicle_number") == ""))
        .then(pl.lit("UNTRACKED_FLEET"))
        .otherwise(pl.col("vehicle_number"))
        .alias("vehicle_group"),
    ])

    df = df.with_columns([
        pl.when(pl.col("weekday_num") >= 6)
        .then(pl.lit("Weekend"))
        .otherwise(pl.lit("Weekday"))
        .alias("day_type"),
        ((pl.col("hour") * 60 + pl.col("minute")) // TIME_BIN_MINUTES).cast(pl.Int32).alias("time_bin"),
    ])

    df = df.sort(["vehicle_group", "route", "service_date", "created_at"])

    partition_cols = ["vehicle_group", "route", "service_date"]

    df = df.with_columns([
        (
            (pl.col("created_at").diff().dt.total_milliseconds() / 60000.0)
            .over(partition_cols)
            .alias("delta_t_minutes")
        ),
        (
            pl.col("start_seq").diff()
            .over(partition_cols)
            .alias("delta_seq")
        ),
    ])

    df = df.with_columns(
        (
            (pl.col("delta_t_minutes") > HYSTERESIS_MAX_TIME_GAP_MINUTES)
            | (pl.col("delta_seq") <= HYSTERESIS_TURNAROUND_SEQ_DELTA)
            | pl.col("delta_t_minutes").is_null()
        ).alias("is_trip_boundary")
    )

    df = df.with_columns(
        pl.col("is_trip_boundary")
        .cum_sum()
        .over(partition_cols)
        .alias("trip_sequence_num")
    )

    df = df.with_columns(
        pl.concat_str([
            pl.col("route"),
            pl.lit("_"),
            pl.col("service_date"),
            pl.lit("_"),
            pl.col("vehicle_group"),
            pl.lit("_TRIP"),
            pl.col("trip_sequence_num").cast(pl.String),
        ]).alias("inferred_trip_id")
    )

    return df
