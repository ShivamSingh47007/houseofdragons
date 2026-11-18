import heapq
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import duckdb
import polars as pl

from src.config import discover_csv_path, discover_gtfs_dir


def create_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA preserve_insertion_order=false;")
    return con


def load_tickets_into_duckdb(
    con: duckdb.DuckDBPyConnection,
    csv_path: Optional[Path] = None,
    table_name: str = "raw_tickets",
    sample_size: Optional[int] = None,
) -> Path:
    actual_path = csv_path if csv_path is not None else discover_csv_path()
    limit_clause = f"LIMIT {sample_size}" if sample_size is not None else ""
    query = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT 
        CAST(passenger_count AS BIGINT) AS passenger_count,
        CAST(amount AS DOUBLE) AS amount,
        CASE 
            WHEN vehicle_number IS NULL OR vehicle_number = 'NULL' OR vehicle_number = '' THEN NULL 
            ELSE TRIM(vehicle_number) 
        END AS vehicle_number,
        CAST(ticket_type_id AS BIGINT) AS ticket_type_id,
        TRIM(CAST(start_stop_name AS VARCHAR)) AS start_stop_name,
        TRIM(CAST(end_stop_name AS VARCHAR)) AS end_stop_name,
        TRIM(CAST(route AS VARCHAR)) AS route,
        CAST(created_at AS TIMESTAMPTZ) AS created_at,
        CAST(is_ac AS BOOLEAN) AS is_ac
    FROM read_csv_auto('{actual_path.as_posix()}')
    WHERE start_stop_name IS NOT NULL 
      AND end_stop_name IS NOT NULL
      AND start_stop_name != 'NULL'
      AND end_stop_name != 'NULL'
      AND route IS NOT NULL
      AND route != 'NULL'
    {limit_clause}
    """
    con.execute(query)
    return actual_path


def load_static_gtfs(gtfs_dir: Path) -> Dict[str, List[str]]:
    routes_file = gtfs_dir / "routes.txt"
    trips_file = gtfs_dir / "trips.txt"
    stop_times_file = gtfs_dir / "stop_times.txt"
    stops_file = gtfs_dir / "stops.txt"

    if not (routes_file.exists() and trips_file.exists() and stop_times_file.exists() and stops_file.exists()):
        return {}

    con = duckdb.connect()
    con.execute(f"""
    CREATE TABLE gtfs_routes AS SELECT route_id, route_short_name FROM read_csv_auto('{routes_file.as_posix()}');
    CREATE TABLE gtfs_trips AS SELECT trip_id, route_id, shape_id FROM read_csv_auto('{trips_file.as_posix()}');
    CREATE TABLE gtfs_stop_times AS SELECT trip_id, stop_id, stop_sequence FROM read_csv_auto('{stop_times_file.as_posix()}');
    CREATE TABLE gtfs_stops AS SELECT stop_id, stop_name FROM read_csv_auto('{stops_file.as_posix()}');
    """)

    query = """
    WITH representative_trips AS (
        SELECT route_id, trip_id,
               ROW_NUMBER() OVER (PARTITION BY route_id ORDER BY trip_id) AS rn
        FROM gtfs_trips
    ),
    ordered_stops AS (
        SELECT 
            COALESCE(r.route_short_name, r.route_id) AS route_name,
            s.stop_name,
            st.stop_sequence
        FROM representative_trips rt
        JOIN gtfs_routes r ON rt.route_id = r.route_id
        JOIN gtfs_stop_times st ON rt.trip_id = st.trip_id
        JOIN gtfs_stops s ON st.stop_id = s.stop_id
        WHERE rt.rn = 1
        ORDER BY route_name, st.stop_sequence
    )
    SELECT route_name, stop_name FROM ordered_stops
    """
    rows = con.execute(query).fetchall()
    result: Dict[str, List[str]] = {}
    for route_name, stop_name in rows:
        if route_name not in result:
            result[route_name] = []
        result[route_name].append(stop_name)
    con.close()
    return result


def build_fallback_gtfs_dag(
    con: duckdb.DuckDBPyConnection, table_name: str = "raw_tickets"
) -> Dict[str, List[str]]:
    stop_stats_query = f"""
    WITH boardings AS (
        SELECT route, start_stop_name AS stop_name, COUNT(*) AS b_count
        FROM {table_name}
        GROUP BY route, start_stop_name
    ),
    alightings AS (
        SELECT route, end_stop_name AS stop_name, COUNT(*) AS a_count
        FROM {table_name}
        GROUP BY route, end_stop_name
    )
    SELECT 
        COALESCE(b.route, a.route) AS route,
        COALESCE(b.stop_name, a.stop_name) AS stop_name,
        (COALESCE(a.a_count, 0) + 1.0) / (COALESCE(b.b_count, 0) + COALESCE(a.a_count, 0) + 2.0) AS flow_ratio
    FROM boardings b
    FULL OUTER JOIN alightings a
        ON b.route = a.route AND b.stop_name = a.stop_name
    """
    stop_stats_rows = con.execute(stop_stats_query).fetchall()

    route_flow_map: Dict[str, Dict[str, float]] = {}
    for route, stop_name, flow_ratio in stop_stats_rows:
        if route not in route_flow_map:
            route_flow_map[route] = {}
        route_flow_map[route][stop_name] = float(flow_ratio)

    transitions_query = f"""
    SELECT 
        route, 
        start_stop_name AS u, 
        end_stop_name AS v, 
        COUNT(*) AS edge_count
    FROM {table_name}
    WHERE start_stop_name != end_stop_name
    GROUP BY route, start_stop_name, end_stop_name
    """
    transitions_rows = con.execute(transitions_query).fetchall()

    route_transitions: Dict[str, List[Tuple[str, str, int]]] = {}
    for route, u, v, weight in transitions_rows:
        if route not in route_transitions:
            route_transitions[route] = []
        route_transitions[route].append((u, v, int(weight)))

    route_topologies: Dict[str, List[str]] = {}

    for route, flow_map in route_flow_map.items():
        all_stops = list(flow_map.keys())
        if len(all_stops) <= 1:
            route_topologies[route] = all_stops
            continue

        adj: Dict[str, List[str]] = {s: [] for s in all_stops}
        in_degree: Dict[str, int] = {s: 0 for s in all_stops}

        for u, v, _ in route_transitions.get(route, []):
            if u in flow_map and v in flow_map:
                if flow_map[u] <= flow_map[v]:
                    adj[u].append(v)
                    in_degree[v] += 1

        pq: List[Tuple[float, str]] = []
        for s in all_stops:
            if in_degree[s] == 0:
                heapq.heappush(pq, (flow_map[s], s))

        ordered: List[str] = []
        visited: Set[str] = set()

        while pq:
            _, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            ordered.append(u)

            for v in adj[u]:
                if v not in visited:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        heapq.heappush(pq, (flow_map[v], v))

            if not pq and len(visited) < len(all_stops):
                unvisited = [s for s in all_stops if s not in visited]
                if unvisited:
                    unvisited.sort(key=lambda s: (in_degree[s], flow_map[s]))
                    chosen = unvisited[0]
                    in_degree[chosen] = 0
                    heapq.heappush(pq, (flow_map[chosen], chosen))

        unvisited_final = [s for s in all_stops if s not in visited]
        if unvisited_final:
            unvisited_final.sort(key=lambda s: flow_map[s])
            ordered.extend(unvisited_final)

        route_topologies[route] = ordered

    return route_topologies


def get_all_route_stop_sequences(
    con: duckdb.DuckDBPyConnection,
    gtfs_dir: Optional[Path] = None,
    table_name: str = "raw_tickets",
) -> Dict[str, List[str]]:
    actual_gtfs_dir = gtfs_dir if gtfs_dir is not None else discover_gtfs_dir()
    gtfs_topologies: Dict[str, List[str]] = {}
    if actual_gtfs_dir is not None:
        gtfs_topologies = load_static_gtfs(actual_gtfs_dir)

    fallback_topologies = build_fallback_gtfs_dag(con, table_name=table_name)

    unified: Dict[str, List[str]] = {}
    for route, stops in fallback_topologies.items():
        if route in gtfs_topologies and len(gtfs_topologies[route]) > 1:
            unified[route] = gtfs_topologies[route]
        else:
            unified[route] = stops

    for route, stops in gtfs_topologies.items():
        if route not in unified and len(stops) > 1:
            unified[route] = stops

    return unified
