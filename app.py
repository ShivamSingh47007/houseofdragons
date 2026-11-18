from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.config import (
    CROWDING_PROFILE_PARQUET,
    VALIDATION_SUMMARY_JSON,
    discover_csv_path,
    discover_gtfs_dir,
)
from src.data_loader import (
    create_duckdb_connection,
    get_all_route_stop_sequences,
    load_tickets_into_duckdb,
)
from src.router import JourneyCard, TransitJourneyPlanner


st.set_page_config(
    page_title="Delhi Bus Crowdedness-Aware Journey Planner",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_crowding_profiles() -> Optional[pl.DataFrame]:
    if CROWDING_PROFILE_PARQUET.exists():
        return pl.read_parquet(CROWDING_PROFILE_PARQUET)
    return None


@st.cache_data
def load_validation_summary() -> Optional[dict]:
    if VALIDATION_SUMMARY_JSON.exists():
        with open(VALIDATION_SUMMARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_planner_and_topologies() -> Tuple[TransitJourneyPlanner, List[str], List[str]]:
    con = create_duckdb_connection()
    try:
        csv_file = discover_csv_path()
        load_tickets_into_duckdb(con, csv_path=csv_file, sample_size=25000)
    except Exception:
        pass
    gtfs_folder = discover_gtfs_dir()
    topologies = get_all_route_stop_sequences(con, gtfs_dir=gtfs_folder)

    df_profiles = load_crowding_profiles()
    planner = TransitJourneyPlanner(topologies, crowding_profiles=df_profiles)
    all_stops = sorted(list(planner.all_canonical_stops))
    all_routes = sorted(list(topologies.keys()))
    return planner, all_stops, all_routes


planner, canonical_stops, available_routes = load_planner_and_topologies()
profiles_df = load_crowding_profiles()
val_summary = load_validation_summary()


st.title("Delhi Bus Crowdedness-Aware Journey Planner")
st.caption("Multi-Criteria Transit Routing with Empirical Bayes & 1D Difference Array Occupancy Reconstruction")

tab_commuter, tab_operator = st.tabs([
    "Commuter Journey Planner",
    "Operator Diagnostics & Validation",
])


with tab_commuter:
    st.subheader("Plan a Crowdedness-Aware Bus Journey")

    col_orig, col_dest, col_day, col_time = st.columns(4)

    default_origin_index = (
        canonical_stops.index("Badarpur Border")
        if "Badarpur Border" in canonical_stops
        else 0
    )
    default_dest_index = (
        canonical_stops.index("Noida Sector 37")
        if "Noida Sector 37" in canonical_stops
        else (1 if len(canonical_stops) > 1 else 0)
    )

    with col_orig:
        origin_stop = st.selectbox(
            "Origin Stop",
            options=canonical_stops,
            index=default_origin_index,
        )

    with col_dest:
        dest_stop = st.selectbox(
            "Destination Stop",
            options=canonical_stops,
            index=default_dest_index,
        )

    with col_day:
        day_type = st.selectbox("Day Type", options=["Weekday", "Weekend"], index=0)

    with col_time:
        dep_time = st.selectbox(
            "Departure Time",
            options=[
                "07:00", "07:30", "08:00", "08:30", "09:00", "09:30", "10:00",
                "12:00", "14:00", "16:00", "17:00", "17:30", "18:00", "18:30",
                "19:00", "19:30", "20:00", "21:00",
            ],
            index=12,
        )

    btn_search = st.button("Find Optimal Journeys", type="primary", use_container_width=True)

    if btn_search or origin_stop:
        cards = planner.plan_journey(
            origin_query=origin_stop,
            destination_query=dest_stop,
            departure_time_str=dep_time,
            day_type=day_type,
        )

        if not cards:
            st.warning(f"No direct or 1-transfer transit connection found between '{origin_stop}' and '{dest_stop}'. Try adjacent stops.")
        else:
            st.write("---")
            col1, col2, col3 = st.columns(3)
            card_cols = [col1, col2, col3]

            category_colors = {
                "Seats Available": "#28a745",
                "Standing Room": "#ffc107",
                "Crowded": "#dc3545",
            }

            for idx, card in enumerate(cards):
                with card_cols[idx]:
                    card_color = category_colors.get(card.overall_crowding_category, "#6c757d")
                    st.markdown(
                        f"""
                        <div style="border: 2px solid {card_color}; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: rgba(255, 255, 255, 0.05);">
                            <h4 style="margin-top: 0; color: {card_color};">{card.option_type}</h4>
                            <p style="font-size: 1.1em; font-weight: bold; margin-bottom: 5px;">Route: {card.route_label}</p>
                            <p style="font-size: 1.4em; font-weight: bold; margin: 5px 0;">Time: {card.departure_time} -> {card.arrival_time} <span style="font-size: 0.7em; color: #aaa;">({card.total_travel_minutes} min)</span></p>
                            <p style="margin: 4px 0;">Transfers: <b>{card.transfers_count}</b> | Shift: <b>+{card.departure_shift_minutes}m</b></p>
                            <p style="margin: 4px 0;">Peak Crowding: <span style="color: {card_color}; font-weight: bold;">{card.overall_crowding_category}</span> (CI: {card.peak_crowding_index:.2f})</p>
                            <p style="margin: 4px 0; font-size: 0.85em; color: #888;">Confidence: <b>{card.overall_confidence}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("View Stretch-by-Stretch Crowding"):
                        for leg in card.legs:
                            st.caption(f"Leg: Route {leg.route} ({leg.from_stop} -> {leg.to_stop})")
                            stretch_rows = []
                            for s in leg.stretches:
                                stretch_rows.append({
                                    "From": s.from_stop,
                                    "To": s.to_stop,
                                    "Seq": s.stop_seq,
                                    "CI": s.crowding_index,
                                    "Crowding": s.crowding_category,
                                    "Confidence": s.confidence_tier,
                                })
                            if stretch_rows:
                                st.dataframe(stretch_rows, use_container_width=True)


with tab_operator:
    st.subheader("Operator Transit Diagnostics & Validation")

    if val_summary:
        holdout = val_summary.get("out_of_time_holdout", {})
        diurnal = val_summary.get("diurnal_and_weekly_patterns", {})

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Evaluated Segments", f"{holdout.get('evaluated_segments', 0):,}")
        m2.metric("MAE (Holdout)", f"{holdout.get('mae', 0.0):.3f} pax")
        m3.metric("Crowding Accuracy", f"{holdout.get('category_accuracy', 0.0):.1f}%")
        m4.metric("Peak Bimodal Status", "Verified Bimodal" if diurnal.get("is_bimodal_distribution") else "Uniform")
        m5.metric("Weekday/Weekend Ratio", f"{diurnal.get('weekday_to_weekend_ratio', 1.0)}x")

        st.write("---")

        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### Diurnal Hourly Passenger Profile (Bimodal Peaks)")
            hourly_profile = diurnal.get("hourly_load_profile", {})
            if hourly_profile:
                hrs = [int(h) for h in sorted(hourly_profile.keys(), key=lambda x: int(x))]
                loads = [hourly_profile[str(h)] for h in hrs]
                fig_diurnal = go.Figure()
                fig_diurnal.add_trace(go.Bar(
                    x=hrs,
                    y=loads,
                    marker_color=["#dc3545" if h in [8, 9, 10, 17, 18, 19] else "#007bff" for h in hrs],
                    name="Avg Load",
                ))
                fig_diurnal.update_layout(
                    xaxis_title="Hour of Day (IST)",
                    yaxis_title="Mean Passenger Load per Stretch",
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_diurnal, use_container_width=True)

        with c_right:
            st.markdown("#### Weekday vs. Weekend Passenger Demand")
            w_day = diurnal.get("weekday_daily_load", 0.0)
            w_end = diurnal.get("weekend_daily_load", 0.0)
            fig_week = go.Figure()
            fig_week.add_trace(go.Bar(
                x=["Weekday", "Weekend"],
                y=[w_day, w_end],
                marker_color=["#28a745", "#ffc107"],
            ))
            fig_week.update_layout(
                yaxis_title="Total Reconstructed Passenger Volume",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_week, use_container_width=True)

    st.write("---")
    st.markdown("#### Route Space-Time Crowding Heatmap (Marey Diagram)")

    selected_route = st.selectbox(
        "Select Route for Space-Time Heatmap",
        options=available_routes,
        index=0 if available_routes else None,
    )

    if profiles_df is not None and selected_route:
        route_profiles = profiles_df.filter(pl.col("route") == selected_route)
        if route_profiles.height > 0:
            pivot_df = (
                route_profiles.pivot(
                    on="time_bin",
                    index="stop_seq",
                    values="crowding_index_p50",
                    aggregate_function="mean",
                )
                .sort("stop_seq")
            )

            time_bin_cols = [c for c in pivot_df.columns if c != "stop_seq"]
            time_bin_cols = sorted(time_bin_cols, key=lambda x: int(x))

            z_vals = pivot_df.select(time_bin_cols).to_numpy()
            y_stops = pivot_df["stop_seq"].to_list()
            x_times = [f"{int(c)*30//60:02d}:{int(c)*30%60:02d}" for c in time_bin_cols]

            fig_heat = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=x_times,
                y=y_stops,
                colorscale="Viridis",
                colorbar=dict(title="Crowding Index (CI)"),
            ))
            fig_heat.update_layout(
                title=f"Space-Time Crowding Heatmap: Route {selected_route}",
                xaxis_title="Time of Day",
                yaxis_title="Stop Sequence (Departure -> Terminal)",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info(f"No aggregated profile data found for route {selected_route}.")

    if val_summary:
        st.markdown("#### Top Overloaded Route Corridors (Peak P85 CI)")
        top_c = val_summary.get("top_overloaded_corridors", [])
        if top_c:
            st.dataframe(top_c, use_container_width=True)
