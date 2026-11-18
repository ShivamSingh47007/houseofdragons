import re
from typing import Dict, List, Optional, Tuple
import duckdb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from src.config import STOP_MATCH_SCORE_THRESHOLD


def normalize_stop_name(raw_name: Optional[str]) -> str:
    if raw_name is None:
        return ""
    text = str(raw_name).upper().strip()
    text = re.sub(r"[\.,\-/\\_#\(\)]", " ", text)
    replacements = {
        r"\bSEC\b": "SECTOR",
        r"\bXING\b": "CROSSING",
        r"\bTERML\b": "TERMINAL",
        r"\bTERM\b": "TERMINAL",
        r"\bDP\b": "DEPOT",
        r"\bDPT\b": "DEPOT",
        r"\bSTN\b": "STATION",
        r"\bRD\b": "ROAD",
        r"\bVLG\b": "VILLAGE",
        r"\bAPT\b": "APARTMENT",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_route_candidate_dictionaries(
    route_topologies: Dict[str, List[str]]
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[str]], Dict[str, int]]:
    route_stop_to_seq: Dict[str, Dict[str, int]] = {}
    route_candidate_names: Dict[str, List[str]] = {}
    route_max_sequences: Dict[str, int] = {}

    for route, stops in route_topologies.items():
        route_stop_to_seq[route] = {}
        route_candidate_names[route] = []
        route_max_sequences[route] = max(1, len(stops))

        for seq, stop_name in enumerate(stops, 1):
            normalized = normalize_stop_name(stop_name)
            if normalized not in route_stop_to_seq[route]:
                route_stop_to_seq[route][normalized] = seq
                route_candidate_names[route].append(normalized)

    return route_stop_to_seq, route_candidate_names, route_max_sequences


def build_memoized_stop_matcher(
    unique_route_stops: List[Tuple[str, str]],
    route_stop_to_seq: Dict[str, Dict[str, int]],
    route_candidate_names: Dict[str, List[str]],
    route_max_sequences: Dict[str, int],
) -> Dict[Tuple[str, str], int]:
    memo_cache: Dict[Tuple[str, str], int] = {}

    for route, raw_stop in unique_route_stops:
        key = (route, raw_stop)
        if key in memo_cache:
            continue

        if route not in route_candidate_names or not route_candidate_names[route]:
            memo_cache[key] = 1
            continue

        normalized_raw = normalize_stop_name(raw_stop)
        candidate_map = route_stop_to_seq[route]

        if normalized_raw in candidate_map:
            memo_cache[key] = candidate_map[normalized_raw]
            continue

        candidates = route_candidate_names[route]
        match_result = process.extractOne(
            normalized_raw,
            candidates,
            scorer=fuzz.token_sort_ratio,
        )

        if match_result is not None:
            matched_name, score, _ = match_result
            if score >= STOP_MATCH_SCORE_THRESHOLD:
                memo_cache[key] = candidate_map[matched_name]
            else:
                memo_cache[key] = candidate_map[matched_name]
        else:
            memo_cache[key] = 1

    return memo_cache


def build_hierarchical_imputation_tables(
    df: pl.DataFrame,
) -> Tuple[Dict[Tuple[str, float, int], float], Dict[Tuple[str, float], float], Dict[str, float], float]:
    valid_hops = df.filter(
        (pl.col("end_seq") > pl.col("start_seq")) & pl.col("start_seq").is_not_null() & pl.col("end_seq").is_not_null()
    ).with_columns((pl.col("end_seq") - pl.col("start_seq")).alias("hop"))

    if valid_hops.height == 0:
        return {}, {}, {}, 3.0

    global_median = float(valid_hops["hop"].median() or 3.0)

    tier1_df = (
        valid_hops.group_by(["route", "amount", "ticket_type_id"])
        .agg(pl.col("hop").median().alias("median_hop"))
    )
    tier1_map: Dict[Tuple[str, float, int], float] = {
        (row[0], float(row[1]), int(row[2])): float(row[3])
        for row in tier1_df.iter_rows()
    }

    tier2_df = (
        valid_hops.group_by(["route", "amount"])
        .agg(pl.col("hop").median().alias("median_hop"))
    )
    tier2_map: Dict[Tuple[str, float], float] = {
        (row[0], float(row[1])): float(row[2])
        for row in tier2_df.iter_rows()
    }

    tier3_df = (
        valid_hops.group_by("route")
        .agg(pl.col("hop").median().alias("median_hop"))
    )
    tier3_map: Dict[str, float] = {
        row[0]: float(row[1]) for row in tier3_df.iter_rows()
    }

    return tier1_map, tier2_map, tier3_map, global_median


def reconcile_and_impute_dataset(
    con: duckdb.DuckDBPyConnection,
    route_topologies: Dict[str, List[str]],
    input_table: str = "raw_tickets",
) -> pl.DataFrame:
    route_stop_to_seq, route_candidates, route_max_seqs = build_route_candidate_dictionaries(
        route_topologies
    )

    unique_pairs_query = f"""
    SELECT DISTINCT route, start_stop_name AS stop_name FROM {input_table}
    UNION
    SELECT DISTINCT route, end_stop_name AS stop_name FROM {input_table}
    """
    unique_pairs = con.execute(unique_pairs_query).fetchall()

    memo_matcher = build_memoized_stop_matcher(
        unique_pairs, route_stop_to_seq, route_candidates, route_max_seqs
    )

    mapping_rows = [
        {"route": route, "raw_stop": raw_stop, "assigned_seq": seq}
        for (route, raw_stop), seq in memo_matcher.items()
    ]
    mapping_df = pl.DataFrame(mapping_rows)
    con.register("stop_mapping", mapping_df.to_arrow())

    max_seq_rows = [
        {"route": route, "max_seq": max_seq}
        for route, max_seq in route_max_seqs.items()
    ]
    max_seq_df = pl.DataFrame(max_seq_rows)
    con.register("route_max_seqs", max_seq_df.to_arrow())

    joined_query = f"""
    SELECT 
        t.passenger_count,
        t.amount,
        t.vehicle_number,
        t.ticket_type_id,
        t.route,
        t.start_stop_name,
        t.end_stop_name,
        t.created_at,
        t.is_ac,
        COALESCE(sm_start.assigned_seq, 1) AS raw_start_seq,
        COALESCE(sm_end.assigned_seq, 1) AS raw_end_seq,
        COALESCE(rms.max_seq, 30) AS max_seq
    FROM {input_table} t
    LEFT JOIN stop_mapping sm_start 
        ON t.route = sm_start.route AND t.start_stop_name = sm_start.raw_stop
    LEFT JOIN stop_mapping sm_end 
        ON t.route = sm_end.route AND t.end_stop_name = sm_end.raw_stop
    LEFT JOIN route_max_seqs rms 
        ON t.route = rms.route
    """
    arrow_table = con.execute(joined_query).arrow()
    df = pl.from_arrow(arrow_table)

    df = df.with_columns([
        pl.col("raw_start_seq").alias("start_seq"),
        pl.col("raw_end_seq").alias("end_seq"),
    ])

    tier1_map, tier2_map, tier3_map, global_median = build_hierarchical_imputation_tables(df)

    routes = df["route"].to_list()
    amounts = df["amount"].to_list()
    ticket_types = df["ticket_type_id"].to_list()
    start_seqs = df["start_seq"].to_numpy().copy()
    end_seqs = df["end_seq"].to_numpy().copy()
    max_seqs = df["max_seq"].to_numpy()

    needs_imputation = (end_seqs <= start_seqs) | (end_seqs == 0)

    for i in np.where(needs_imputation)[0]:
        route = routes[i]
        amount = amounts[i]
        ticket_type = ticket_types[i]
        s_seq = start_seqs[i]
        m_seq = max_seqs[i]

        t1_key = (route, amount, ticket_type)
        if t1_key in tier1_map:
            hop = tier1_map[t1_key]
        elif (route, amount) in tier2_map:
            hop = tier2_map[(route, amount)]
        elif route in tier3_map:
            hop = tier3_map[route]
        else:
            hop = global_median

        hop_int = max(1, int(round(hop)))
        if s_seq >= m_seq:
            s_seq = max(1, m_seq - 1)
            start_seqs[i] = s_seq
            end_seqs[i] = m_seq
        else:
            end_seqs[i] = min(m_seq, s_seq + hop_int)

    df = df.with_columns([
        pl.Series("start_seq", start_seqs),
        pl.Series("end_seq", end_seqs),
    ])

    return df
