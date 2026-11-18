import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional

from src.config import (
    CROWDING_PROFILE_PARQUET,
    OUTPUT_DIR,
    VALIDATION_SUMMARY_JSON,
    discover_csv_path,
    discover_gtfs_dir,
)
from src.data_loader import (
    create_duckdb_connection,
    get_all_route_stop_sequences,
    load_tickets_into_duckdb,
)
from src.occupancy_engine import (
    compute_empirical_bayes_crowding_profiles,
    compute_trip_sweep_line_occupancy,
)
from src.router import JourneyCard, StretchDetail, TransitJourneyPlanner
from src.stop_matcher import reconcile_and_impute_dataset
from src.trip_segmenter import segment_trips_hysteresis
from src.validator import run_full_validation


def render_ascii_hourly_chart(hourly_profile: Dict[str, float]) -> None:
    print("\n  Hour | Mean Pax Load | Diurnal Peak Distribution")
    print("  -----+---------------+--------------------------------------------------")
    max_val = max(hourly_profile.values()) if hourly_profile else 1.0
    scale = 40.0 / max(1.0, max_val)

    for h in range(6, 23):
        val = hourly_profile.get(str(h), 0.0)
        bar_len = int(round(val * scale))
        bar = "#" * bar_len
        marker = ""
        if h in [8, 9, 10]:
            marker = " [MORNING PEAK]"
        elif h in [12, 13, 14]:
            marker = " [MIDDAY TROUGH]"
        elif h in [17, 18, 19]:
            marker = " [EVENING PEAK]"
        print(f"  {h:02d}:00 | {val:6.2f} pax   | {bar:<40} {marker}")


def render_journey_card_terminal(card: JourneyCard) -> None:
    border = "=" * 80
    sub_border = "-" * 80
    print(f"\n{border}")
    print(f"  OPTION TYPE:       {card.option_type.upper()}")
    print(f"  TRANSIT SERVICE:   {card.route_label}")
    print(f"  SCHEDULE:          Depart {card.departure_time} -> Arrive {card.arrival_time} (Duration: {card.total_travel_minutes} mins)")
    print(f"  TRANSFERS:         {card.transfers_count} transfer(s) | Time-Shift: +{card.departure_shift_minutes} mins")
    print(f"  CROWDING STATUS:   {card.overall_crowding_category.upper()} (Peak CI: {card.peak_crowding_index:.2f}, Avg CI: {card.avg_crowding_index:.2f})")
    print(f"  DATA CONFIDENCE:   {card.overall_confidence}")
    print(f"  GENERALIZED COST:  {card.generalized_cost:.2f}")
    print(sub_border)

    for leg_idx, leg in enumerate(card.legs, 1):
        print(f"  Leg {leg_idx}: Route {leg.route} ({leg.from_stop} -> {leg.to_stop}) | Duration: {leg.duration_minutes} mins")
        if leg.stretches:
            print("    Seq | From Stop                    | To Stop                      | CI   | Category         | Confidence")
            print("    ----+------------------------------+------------------------------+------+------------------+-----------------")
            for s in leg.stretches:
                f_name = s.from_stop[:28]
                t_name = s.to_stop[:28]
                print(f"    {s.stop_seq:3d} | {f_name:<28} | {t_name:<28} | {s.crowding_index:4.2f} | {s.crowding_category:<16} | {s.confidence_tier}")
    print(border)


def render_spacetime_corridor_table(planner: TransitJourneyPlanner, route: str) -> None:
    stops = planner.route_topologies.get(route, [])
    if not stops:
        return
    print(f"\n  SPACE-TIME CROWDING MATRIX: ROUTE {route}")
    print("  Stop Seq | Stop Name                     | 08:30 (M-Peak) | 13:00 (Midday) | 18:30 (E-Peak)")
    print("  ---------+-------------------------------+----------------+----------------+---------------")
    for s_idx in range(1, min(len(stops), 12)):
        s_name = stops[s_idx - 1][:29]
        ci_m, cat_m, _ = planner._get_stretch_crowding(route, s_idx, "Weekday", 17)
        ci_mid, cat_mid, _ = planner._get_stretch_crowding(route, s_idx, "Weekday", 26)
        ci_e, cat_e, _ = planner._get_stretch_crowding(route, s_idx, "Weekday", 37)
        print(f"  {s_idx:8d} | {s_name:<29} | {ci_m:5.2f} ({cat_m[:5]}) | {ci_mid:5.2f} ({cat_mid[:5]}) | {ci_e:5.2f} ({cat_e[:5]})")


def execute_pipeline(
    csv_path: Optional[Path] = None,
    gtfs_dir: Optional[Path] = None,
    sample_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
    query_origin: Optional[str] = None,
    query_dest: Optional[str] = None,
    query_time: str = "18:00",
    query_day: str = "Weekday",
    interactive_mode: bool = False,
) -> None:
    pipeline_start_time = time.time()
    target_output_dir = output_dir if output_dir is not None else OUTPUT_DIR
    target_output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("DELHI BUS CROWDEDNESS-AWARE JOURNEY PLANNER: TERMINAL SYSTEM ENGINE")
    print("================================================================================")

    step_start = time.time()
    actual_csv = csv_path if csv_path is not None else discover_csv_path()
    actual_gtfs = gtfs_dir if gtfs_dir is not None else discover_gtfs_dir()
    print(f"[Stage 1/7] Ingesting ticketing logs from: {actual_csv}")
    if actual_gtfs:
        print(f"            GTFS directory discovered: {actual_gtfs}")
    else:
        print("            GTFS absent. Activating Kahn's Flow-Ratio Topological Sort fallback.")

    con = create_duckdb_connection()
    load_tickets_into_duckdb(con, csv_path=actual_csv, sample_size=sample_size)
    total_tickets = con.execute("SELECT COUNT(*) FROM raw_tickets").fetchone()[0]
    print(f"            Loaded {total_tickets:,} tickets in {time.time() - step_start:.2f}s")

    step_start = time.time()
    print("[Stage 2/7] Constructing route stop topologies (Kahn's DAG Flow-Ratio Sort)...")
    route_topologies = get_all_route_stop_sequences(con, gtfs_dir=actual_gtfs)
    print(f"            Synthesized topologies for {len(route_topologies):,} routes in {time.time() - step_start:.2f}s")

    step_start = time.time()
    print("[Stage 3/7] Reconciling stop names & imputing missing alightings (RapidFuzz 3-Tier)...")
    df_reconciled = reconcile_and_impute_dataset(con, route_topologies)
    print(f"            Reconciled {df_reconciled.height:,} records in {time.time() - step_start:.2f}s")

    step_start = time.time()
    print("[Stage 4/7] Sessionizing trips with spatial-temporal hysteresis scan...")
    df_trips = segment_trips_hysteresis(df_reconciled)
    num_trips = df_trips["inferred_trip_id"].n_unique()
    print(f"            Inferred {num_trips:,} trips in {time.time() - step_start:.2f}s")

    step_start = time.time()
    print("[Stage 5/7] Executing 1D difference-array sweep-line prefix sums & capacity normalization...")
    df_stretch_trips, trip_summary = compute_trip_sweep_line_occupancy(df_trips)
    print(f"            Reconstructed {df_stretch_trips.height:,} stretch-level observations in {time.time() - step_start:.2f}s")

    step_start = time.time()
    print("[Stage 6/7] Aggregating spatio-temporal tensor & Empirical Bayes shrinkage...")
    profile_path = target_output_dir / "crowding_profiles.parquet"
    profiles = compute_empirical_bayes_crowding_profiles(df_stretch_trips, export_path=profile_path)
    print(f"            Exported {profiles.height:,} crowding profile cells to {profile_path} in {time.time() - step_start:.2f}s")

    step_start = time.time()
    print("[Stage 7/7] Running out-of-time cross-validation & diurnal pattern verifier...")
    summary_path = target_output_dir / "validation_summary.json"
    val_summary = run_full_validation(df_stretch_trips, profiles, output_path=summary_path)
    print(f"            Exported validation report to {summary_path} in {time.time() - step_start:.2f}s")

    print("\n--------------------------------------------------------------------------------")
    print("VALIDATION SUMMARY & MODEL METRICS:")
    print("--------------------------------------------------------------------------------")
    holdout = val_summary.get("out_of_time_holdout", {})
    diurnal = val_summary.get("diurnal_and_weekly_patterns", {})
    print("  Holdout Period:            Oct 1-21 (Train) vs. Oct 22-31 (Holdout Test)")
    print(f"  Evaluated Active Segments: {holdout.get('evaluated_segments', 0):,}")
    print(f"  Mean Absolute Error (MAE): {holdout.get('mae', 0.0):.4f} passengers per stretch")
    print(f"  Classification Accuracy:   {holdout.get('category_accuracy', 0.0):.2f}%")
    print(f"  Bimodal Peak Verified:     {'YES (Morning & Evening Peaks Confirmed)' if diurnal.get('is_bimodal_distribution') else 'NO'}")
    print(f"  Morning Peak Load (08-11): {diurnal.get('morning_peak_avg_load', 0.0)} pax/stretch")
    print(f"  Midday Off-Peak Load (12-15): {diurnal.get('midday_offpeak_avg_load', 0.0)} pax/stretch")
    print(f"  Evening Peak Load (17-20): {diurnal.get('evening_peak_avg_load', 0.0)} pax/stretch")
    print(f"  Weekday/Weekend Ratio:     {diurnal.get('weekday_to_weekend_ratio', 1.0)}x")

    render_ascii_hourly_chart(diurnal.get("hourly_load_profile", {}))

    top_corridors = val_summary.get("top_overloaded_corridors", [])[:5]
    if top_corridors:
        print("\n  Top Overloaded Corridors (Peak P85):")
        print("  Route       | Stop Seq | Day Type | Time Bin | P85 CI | P50 CI | Category        | Trips")
        print("  ------------+----------+----------+----------+--------+--------+-----------------+------")
        for tc in top_corridors:
            print(f"  {tc['route']:<11} | {tc['stop_seq']:8d} | {tc['day_type']:<8} | {tc['time_bin']:8d} | {tc['p85_crowding_index']:6.2f} | {tc['p50_crowding_index']:6.2f} | {tc['crowding_category']:<15} | {tc['trip_count']:5d}")

    planner = TransitJourneyPlanner(route_topologies, crowding_profiles=profiles)

    sample_route = "8UP" if "8UP" in route_topologies else list(route_topologies.keys())[0]
    render_spacetime_corridor_table(planner, sample_route)

    print("\n--------------------------------------------------------------------------------")
    print("COMMUTER JOURNEY PLANNER (3 PARETO COMMUTER CARDS):")
    print("--------------------------------------------------------------------------------")

    queries_to_run: List[tuple] = []
    if query_origin and query_dest:
        queries_to_run.append((query_origin, query_dest, query_time, query_day))
    else:
        queries_to_run.append(("Anand Vihar", "Connaught Place", "18:00", "Weekday"))
        queries_to_run.append(("Badarpur Border", "Noida Sector 37", "18:00", "Weekday"))

    for q_orig, q_dest, q_tm, q_dy in queries_to_run:
        print(f"\n>>> QUERY: From '{q_orig}' to '{q_dest}' at {q_tm} ({q_dy})")
        cards = planner.plan_journey(q_orig, q_dest, departure_time_str=q_tm, day_type=q_dy)
        if cards:
            for card in cards:
                render_journey_card_terminal(card)
        else:
            print(f"    No direct or 1-transfer transit connection found between '{q_orig}' and '{q_dest}'.")

    if interactive_mode:
        print("\n================================================================================")
        print("INTERACTIVE CMD JOURNEY PLANNER (Type 'exit' to quit)")
        print("================================================================================")
        while True:
            try:
                user_orig = input("\nEnter Origin Stop: ").strip()
                if user_orig.lower() in ["exit", "quit", "q"]:
                    break
                user_dest = input("Enter Destination Stop: ").strip()
                if user_dest.lower() in ["exit", "quit", "q"]:
                    break
                user_time = input("Enter Departure Time [HH:MM] (default 18:00): ").strip()
                if not user_time:
                    user_time = "18:00"
                user_day = input("Enter Day Type [Weekday/Weekend] (default Weekday): ").strip()
                if not user_day:
                    user_day = "Weekday"

                interactive_cards = planner.plan_journey(
                    user_orig, user_dest, departure_time_str=user_time, day_type=user_day
                )
                if interactive_cards:
                    for c in interactive_cards:
                        render_journey_card_terminal(c)
                else:
                    print(f"No direct or 1-transfer connection found between '{user_orig}' and '{user_dest}'.")
            except (KeyboardInterrupt, EOFError):
                break

    total_time = time.time() - pipeline_start_time
    print("================================================================================")
    print(f"PIPELINE EXECUTION COMPLETED IN {total_time:.2f} SECONDS")
    print("================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delhi Bus Crowdedness-Aware Journey Planner Terminal Engine")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--gtfs-dir", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--origin", type=str, default=None)
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--time", type=str, default="18:00")
    parser.add_argument("--day", type=str, default="Weekday")
    parser.add_argument("--interactive", action="store_true", default=False)

    args = parser.parse_args()

    csv_p = Path(args.csv_path) if args.csv_path else None
    gtfs_p = Path(args.gtfs_dir) if args.gtfs_dir else None
    out_p = Path(args.output_dir) if args.output_dir else None

    execute_pipeline(
        csv_path=csv_p,
        gtfs_dir=gtfs_p,
        sample_size=args.sample_size,
        output_dir=out_p,
        query_origin=args.origin,
        query_dest=args.destination,
        query_time=args.time,
        query_day=args.day,
        interactive_mode=args.interactive,
    )
