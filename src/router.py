from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from src.config import (
    CROWDING_PROFILE_PARQUET,
    CROWDING_SEATS_AVAILABLE_MAX_CI,
    CROWDING_STANDING_ROOM_MAX_CI,
    ROUTING_COMFORT_BETA,
    ROUTING_COMFORT_DELTA,
    ROUTING_DEFAULT_INTER_STOP_MINUTES,
    ROUTING_FASTEST_BETA,
    ROUTING_FASTEST_DELTA,
    ROUTING_SHIFTED_BETA,
    ROUTING_SHIFTED_DELTA,
    ROUTING_TRANSFER_PENALTY_MINUTES,
    ROUTING_TRANSFER_WALK_BUFFER_MINUTES,
    STOP_MATCH_SCORE_THRESHOLD,
)


@dataclass
class StretchDetail:
    from_stop: str
    to_stop: str
    stop_seq: int
    crowding_index: float
    crowding_category: str
    confidence_tier: str


@dataclass
class JourneyLeg:
    route: str
    from_stop: str
    to_stop: str
    from_seq: int
    to_seq: int
    duration_minutes: float
    stretches: List[StretchDetail] = field(default_factory=list)


@dataclass
class JourneyCard:
    option_type: str
    route_label: str
    departure_time: str
    arrival_time: str
    total_travel_minutes: float
    departure_shift_minutes: int
    transfers_count: int
    peak_crowding_index: float
    avg_crowding_index: float
    overall_crowding_category: str
    overall_confidence: str
    generalized_cost: float
    legs: List[JourneyLeg] = field(default_factory=list)


class TransitJourneyPlanner:
    def __init__(
        self,
        route_topologies: Dict[str, List[str]],
        crowding_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        self.route_topologies = route_topologies
        self.all_canonical_stops: Set[str] = set()
        self.stop_to_routes: Dict[str, List[Tuple[str, int]]] = {}

        for route, stops in route_topologies.items():
            for seq, stop_name in enumerate(stops, 1):
                self.all_canonical_stops.add(stop_name)
                if stop_name not in self.stop_to_routes:
                    self.stop_to_routes[stop_name] = []
                self.stop_to_routes[stop_name].append((route, seq))

        self.canonical_stop_list = sorted(list(self.all_canonical_stops))

        self.profile_lookup: Dict[Tuple[str, int, str, int], Tuple[float, str, str]] = {}
        if crowding_profiles is not None:
            self._load_profiles(crowding_profiles)
        elif CROWDING_PROFILE_PARQUET.exists():
            df_profiles = pl.read_parquet(CROWDING_PROFILE_PARQUET)
            self._load_profiles(df_profiles)

    def _load_profiles(self, df_profiles: pl.DataFrame) -> None:
        required_cols = {
            "route",
            "stop_seq",
            "day_type",
            "time_bin",
            "crowding_index_p50",
            "crowding_category",
            "confidence_tier",
        }
        if not required_cols.issubset(set(df_profiles.columns)):
            return

        for row in df_profiles.select(list(required_cols)).iter_rows(named=True):
            key = (
                str(row["route"]),
                int(row["stop_seq"]),
                str(row["day_type"]),
                int(row["time_bin"]),
            )
            self.profile_lookup[key] = (
                float(row["crowding_index_p50"]),
                str(row["crowding_category"]),
                str(row["confidence_tier"]),
            )

    def match_stop_name(self, query: str) -> Optional[str]:
        if not query or not self.canonical_stop_list:
            return None
        clean_q = query.strip()
        if clean_q in self.all_canonical_stops:
            return clean_q

        match_res = process.extractOne(
            clean_q,
            self.canonical_stop_list,
            scorer=fuzz.token_sort_ratio,
        )
        if match_res is not None and match_res[1] >= STOP_MATCH_SCORE_THRESHOLD:
            return match_res[0]
        if match_res is not None:
            return match_res[0]
        return None

    def _get_stretch_crowding(
        self,
        route: str,
        stop_seq: int,
        day_type: str,
        time_bin: int,
    ) -> Tuple[float, str, str]:
        key = (route, stop_seq, day_type, time_bin)
        if key in self.profile_lookup:
            return self.profile_lookup[key]
        for delta in [-1, 1, -2, 2]:
            adj_bin = (time_bin + delta) % 48
            adj_key = (route, stop_seq, day_type, adj_bin)
            if adj_key in self.profile_lookup:
                return self.profile_lookup[adj_key]
        return 0.50, "Seats Available", "Low Confidence"

    def _evaluate_candidate_journey(
        self,
        legs_info: List[Tuple[str, str, str, int, int]],
        day_type: str,
        base_dep_minutes: int,
        shift_minutes: int,
        beta: float,
        delta: float,
        option_name: str,
    ) -> JourneyCard:
        dep_total_min = base_dep_minutes + shift_minutes
        current_clock_min = dep_total_min

        journey_legs: List[JourneyLeg] = []
        all_cis: List[float] = []
        confidences: List[str] = []
        total_generalized_cost = 0.0
        total_travel_time = 0.0

        for leg_idx, (r_name, f_stop, t_stop, f_seq, t_seq) in enumerate(legs_info):
            if leg_idx > 0:
                current_clock_min += ROUTING_TRANSFER_WALK_BUFFER_MINUTES
                total_travel_time += ROUTING_TRANSFER_WALK_BUFFER_MINUTES
                total_generalized_cost += ROUTING_TRANSFER_PENALTY_MINUTES

            leg_stretches = max(1, t_seq - f_seq)
            leg_duration = leg_stretches * ROUTING_DEFAULT_INTER_STOP_MINUTES
            total_travel_time += leg_duration

            stretch_details: List[StretchDetail] = []
            route_stops = self.route_topologies.get(r_name, [])

            for s_seq in range(f_seq, t_seq):
                s_from_name = route_stops[s_seq - 1] if s_seq <= len(route_stops) else f"Stop {s_seq}"
                s_to_name = route_stops[s_seq] if s_seq < len(route_stops) else f"Stop {s_seq + 1}"

                curr_time_bin = (int(current_clock_min) // 30) % 48
                ci, cat, conf = self._get_stretch_crowding(
                    r_name, s_seq, day_type, curr_time_bin
                )

                all_cis.append(ci)
                confidences.append(conf)

                stretch_details.append(
                    StretchDetail(
                        from_stop=s_from_name,
                        to_stop=s_to_name,
                        stop_seq=s_seq,
                        crowding_index=round(ci, 2),
                        crowding_category=cat,
                        confidence_tier=conf,
                    )
                )

                stretch_travel_time = ROUTING_DEFAULT_INTER_STOP_MINUTES
                crowding_excess = max(0.0, ci - CROWDING_SEATS_AVAILABLE_MAX_CI)
                stretch_cost = stretch_travel_time * (1.0 + beta * (crowding_excess ** 2))
                total_generalized_cost += stretch_cost

                current_clock_min += stretch_travel_time

            journey_legs.append(
                JourneyLeg(
                    route=r_name,
                    from_stop=f_stop,
                    to_stop=t_stop,
                    from_seq=f_seq,
                    to_seq=t_seq,
                    duration_minutes=round(leg_duration, 1),
                    stretches=stretch_details,
                )
            )

        total_generalized_cost += delta * abs(shift_minutes)

        transfers = max(0, len(legs_info) - 1)
        peak_ci = float(np.max(all_cis)) if all_cis else 0.5
        avg_ci = float(np.mean(all_cis)) if all_cis else 0.5

        if peak_ci <= CROWDING_SEATS_AVAILABLE_MAX_CI:
            overall_cat = "Seats Available"
        elif peak_ci <= CROWDING_STANDING_ROOM_MAX_CI:
            overall_cat = "Standing Room"
        else:
            overall_cat = "Crowded"

        if "Low Confidence" in confidences:
            overall_conf = "Low Confidence"
        elif "Medium Confidence" in confidences:
            overall_conf = "Medium Confidence"
        else:
            overall_conf = "High Confidence"

        dep_hr = (dep_total_min // 60) % 24
        dep_mn = dep_total_min % 60
        arr_total_min = int(current_clock_min)
        arr_hr = (arr_total_min // 60) % 24
        arr_mn = arr_total_min % 60

        route_label = (
            legs_info[0][0]
            if len(legs_info) == 1
            else f"{legs_info[0][0]} -> {legs_info[1][0]} (Transfer at {legs_info[0][2]})"
        )

        return JourneyCard(
            option_type=option_name,
            route_label=route_label,
            departure_time=f"{dep_hr:02d}:{dep_mn:02d}",
            arrival_time=f"{arr_hr:02d}:{arr_mn:02d}",
            total_travel_minutes=round(total_travel_time, 1),
            departure_shift_minutes=shift_minutes,
            transfers_count=transfers,
            peak_crowding_index=round(peak_ci, 2),
            avg_crowding_index=round(avg_ci, 2),
            overall_crowding_category=overall_cat,
            overall_confidence=overall_conf,
            generalized_cost=round(total_generalized_cost, 2),
            legs=journey_legs,
        )

    def plan_journey(
        self,
        origin_query: str,
        destination_query: str,
        departure_time_str: str = "18:00",
        day_type: str = "Weekday",
    ) -> List[JourneyCard]:
        matched_origin = self.match_stop_name(origin_query)
        matched_dest = self.match_stop_name(destination_query)

        if matched_origin is None or matched_dest is None:
            return []

        time_parts = departure_time_str.strip().split(":")
        base_hour = int(time_parts[0]) if len(time_parts) > 0 else 18
        base_minute = int(time_parts[1]) if len(time_parts) > 1 else 0
        base_dep_minutes = base_hour * 60 + base_minute

        direct_candidates: List[Tuple[str, str, str, int, int]] = []
        origin_routes = {
            r: seq for r, seq in self.stop_to_routes.get(matched_origin, [])
        }
        dest_routes = {
            r: seq for r, seq in self.stop_to_routes.get(matched_dest, [])
        }

        for r_name, orig_seq in origin_routes.items():
            if r_name in dest_routes:
                d_seq = dest_routes[r_name]
                if orig_seq < d_seq:
                    direct_candidates.append(
                        (r_name, matched_origin, matched_dest, orig_seq, d_seq)
                    )

        transfer_candidates: List[List[Tuple[str, str, str, int, int]]] = []
        if len(direct_candidates) < 5:
            for r1, s1 in origin_routes.items():
                r1_stops = self.route_topologies.get(r1, [])
                for s_hub_seq in range(s1 + 1, min(len(r1_stops) + 1, s1 + 25)):
                    hub_name = r1_stops[s_hub_seq - 1]
                    for r2, s2 in self.stop_to_routes.get(hub_name, []):
                        if r2 == r1:
                            continue
                        if r2 in dest_routes:
                            s_dest = dest_routes[r2]
                            if s2 < s_dest:
                                transfer_candidates.append([
                                    (r1, matched_origin, hub_name, s1, s_hub_seq),
                                    (r2, hub_name, matched_dest, s2, s_dest),
                                ])
                                if len(transfer_candidates) >= 15:
                                    break
                    if len(transfer_candidates) >= 15:
                        break
                if len(transfer_candidates) >= 15:
                    break

        all_leg_options: List[List[Tuple[str, str, str, int, int]]] = []
        for direct_cand in direct_candidates[:8]:
            all_leg_options.append([direct_cand])
        for trans_cand in transfer_candidates[:8]:
            all_leg_options.append(trans_cand)

        if not all_leg_options:
            if origin_routes:
                r_fallback = list(origin_routes.keys())[0]
                all_leg_options.append([
                    (r_fallback, matched_origin, matched_dest, 1, 5)
                ])
            else:
                return []

        fastest_evals: List[JourneyCard] = []
        for leg_opt in all_leg_options:
            card = self._evaluate_candidate_journey(
                leg_opt,
                day_type,
                base_dep_minutes,
                shift_minutes=0,
                beta=ROUTING_FASTEST_BETA,
                delta=ROUTING_FASTEST_DELTA,
                option_name="Fastest Option",
            )
            fastest_evals.append(card)
        fastest_card = min(fastest_evals, key=lambda c: (c.total_travel_minutes, c.peak_crowding_index))

        comfort_evals: List[JourneyCard] = []
        for leg_opt in all_leg_options:
            card = self._evaluate_candidate_journey(
                leg_opt,
                day_type,
                base_dep_minutes,
                shift_minutes=0,
                beta=ROUTING_COMFORT_BETA,
                delta=ROUTING_COMFORT_DELTA,
                option_name="Comfortable Alternative",
            )
            comfort_evals.append(card)
        comfort_card = min(comfort_evals, key=lambda c: (c.generalized_cost, c.peak_crowding_index))

        shifted_evals: List[JourneyCard] = []
        for shift in [15, 30]:
            for leg_opt in all_leg_options:
                card = self._evaluate_candidate_journey(
                    leg_opt,
                    day_type,
                    base_dep_minutes,
                    shift_minutes=shift,
                    beta=ROUTING_SHIFTED_BETA,
                    delta=ROUTING_SHIFTED_DELTA,
                    option_name=f"Leave Later (+{shift}m)",
                )
                shifted_evals.append(card)
        shifted_card = min(shifted_evals, key=lambda c: (c.peak_crowding_index, c.generalized_cost))

        return [fastest_card, comfort_card, shifted_card]
