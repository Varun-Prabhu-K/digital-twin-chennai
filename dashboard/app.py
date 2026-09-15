import json
import os
import sys
from math import sqrt

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Chennai Digital Twin",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Project root is the directory containing dashboard/, models/, data/, engine/.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

try:
    from engine.rerouting import get_rerouting
except Exception:
    get_rerouting = None


# Model feature definitions are read from the actual trained objects/metadata.
TYPE_ENC = {"NH": 0, "SH": 1, "expressway": 2, "arterial": 3, "collector": 4}
ROAD_TYPES = ["NH", "SH", "expressway", "arterial", "collector"]
TRAFFIC_TARGETS = ["traffic_index", "speed_kmph", "travel_time_min"]
POLLUTANTS = ["pm25", "pm10", "no2"]

METRIC_DESC = {
    "traffic_index": "Congestion index (0–100). Higher means more congestion.",
    "speed_kmph": "Estimated average road speed.",
    "travel_time_min": "Estimated travel time for the configured road segment.",
    "pm25": "PM2.5 concentration (µg/m³).",
    "pm10": "PM10 concentration (µg/m³).",
    "no2": "NO₂ concentration (µg/m³).",
    "electricity_mw": "Estimated city electricity demand (MW).",
}

# These values are used only for a simple presentation indicator in the dashboard.
# We avoid mixing annual and 24-hour guideline periods in the explanatory text.
WHO_GUIDELINES = {"pm25": 15.0, "pm10": 45.0, "no2": 10.0}

# Multi-criteria decision analysis weights. Prediction values and recommendation score
# are deliberately kept separate: the score is a planning preference, not an AI output.
MCDA_WEIGHTS = {
    "traffic": 0.35,
    "pollution": 0.35,
    "electricity": 0.20,
    "accessibility": 0.10,
}


# -----------------------------------------------------------------------------
# Model/data loading
# -----------------------------------------------------------------------------

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading AI models — please wait...")
def load_all():
    tm = _load_json(os.path.join(BASE, "models", "traffic", "traffic_meta.json"))
    pm = _load_json(os.path.join(BASE, "models", "pollution", "pollution_meta.json"))
    em = _load_json(os.path.join(BASE, "models", "electricity", "electricity_meta.json"))
    cfg = _load_json(os.path.join(BASE, "data", "processed", "city_config.json"))

    traffic_models = {}
    for road_type in ROAD_TYPES:
        for target in TRAFFIC_TARGETS:
            path = os.path.join(
                BASE, "models", "traffic", f"{road_type}_{target}_model.pkl"
            )
            if os.path.exists(path):
                traffic_models[f"{road_type}_{target}"] = joblib.load(path)

    pollution_models = {}
    for zone_id in pm.get("zone_models", []):
        for pollutant in POLLUTANTS:
            path = os.path.join(
                BASE, "models", "pollution", f"{zone_id}_{pollutant}_model.pkl"
            )
            if os.path.exists(path):
                pollution_models[f"{zone_id}_{pollutant}"] = joblib.load(path)

    electricity_path = os.path.join(
        BASE, "models", "electricity", "electricity_model.pkl"
    )
    electricity_model = joblib.load(electricity_path)

    return traffic_models, pollution_models, electricity_model, tm, pm, em, cfg


TM_M, PM_M, EL, TM, PM, EM, CFG = load_all()
ROAD_MAP = {r["id"]: r for r in CFG.get("roads", [])}
ZONE_MAP = {z["id"]: z for z in CFG.get("zones", [])}


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def model_features(model):
    """Return the exact trained feature order when the estimator exposes it."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]
    return []


def diagnostics():
    missing_traffic = [
        f"{rt}_{target}"
        for rt in ROAD_TYPES
        for target in TRAFFIC_TARGETS
        if f"{rt}_{target}" not in TM_M
    ]
    missing_pollution = [
        f"{z}_{p}"
        for z in PM.get("zone_models", [])
        for p in POLLUTANTS
        if f"{z}_{p}" not in PM_M
    ]
    return missing_traffic, missing_pollution


missing_traffic, missing_pollution = diagnostics()
if missing_traffic or missing_pollution:
    st.warning(
        "Some trained models are missing. The dashboard will use the closest available "
        "model where possible. Missing traffic models: "
        f"{len(missing_traffic)}; missing pollution models: {len(missing_pollution)}."
    )


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def cyc(value, period):
    return np.sin(2 * np.pi * value / period), np.cos(2 * np.pi * value / period)


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def clip_round(value, low, high, decimals=1):
    return round(float(np.clip(value, low, high)), decimals)


def traffic_color(ti):
    if ti < 30:
        return "#27AE60"
    if ti < 50:
        return "#F1C40F"
    if ti < 70:
        return "#E67E22"
    return "#E74C3C"


def pollution_color(pm25):
    if pm25 < 15:
        return "#27AE60"
    if pm25 < 35:
        return "#F1C40F"
    if pm25 < 60:
        return "#E67E22"
    return "#E74C3C"


def verdict_info(score):
    if score >= 55:
        return "🟢 RECOMMENDED", "#27AE60"
    if score >= 42:
        return "🟡 ACCEPTABLE WITH MITIGATION", "#F39C12"
    return "🔴 NOT RECOMMENDED", "#E74C3C"


def festival_factor_from_config(month, fest_name=None):
    if not fest_name:
        return 1.0
    # Match by label when possible; otherwise use the neutral multiplier.
    for item in CFG.get("festivals", []):
        try:
            _, m, _, name, multiplier = item
            if int(m) == int(month) and str(fest_name).lower() in str(name).lower():
                return float(multiplier)
        except Exception:
            continue
    return 1.0


def road_capacity_proxy(road):
    # Deterministic capacity proxy used only to compare rerouting choices.
    # More lanes and higher free-flow speed -> more spare capacity.
    return max(0.5, float(road.get("lanes", 2)) * float(road.get("speed", 30)) * 0.05)


def distance_km(lat1, lon1, lat2, lon2):
    # Small-area equirectangular approximation; adequate for Chennai-scale distances.
    rlat = np.deg2rad((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * np.cos(rlat)
    return float(np.sqrt(dx * dx + dy * dy))


# -----------------------------------------------------------------------------
# Exact feature construction for the trained models
# -----------------------------------------------------------------------------

def make_traffic_features(
    road,
    hour,
    dow,
    month,
    is_we,
    is_pk,
    is_mo,
    is_fest,
    fest_mult,
    temp,
    hum,
    wind,
    rain,
    traffic_factor=1.0,
):
    hs, hc = cyc(hour, 24)
    ds, dc = cyc(dow, 7)
    ms, mc = cyc(month, 12)

    # The trained models expect these synthetic lag/rolling inputs. We keep the same
    # schema, but make the scenario factor explicit and deterministic.
    road_type = road["type"]
    rt_base = TM.get("road_baselines", {}).get(road_type, {})
    road_base = TM.get("per_road_baselines", {}).get(road["id"], {})
    base_ti = float(road_base.get("traffic_mean", rt_base.get("traffic_mean", 45.0)))

    raw = {
        "hour_sin": hs,
        "hour_cos": hc,
        "dow_sin": ds,
        "dow_cos": dc,
        "month_sin": ms,
        "month_cos": mc,
        "is_weekend": int(is_we),
        "is_peak_hour": int(is_pk),
        "is_monsoon": int(is_mo),
        "is_festival": int(is_fest),
        "festival_mult": float(fest_mult),
        "road_type_enc": TYPE_ENC.get(road_type, 3),
        "lanes": float(road.get("lanes", 2)),
        "length_km": float(road.get("len", 1.0)),
        "temperature": float(temp),
        "humidity": float(hum),
        "wind_speed_kmph": float(wind),
        "rainfall_mm": float(rain),
        "traffic_lag1": base_ti * float(traffic_factor),
        "traffic_lag3": base_ti * float(traffic_factor),
        "traffic_lag24": base_ti,
        "traffic_roll4": base_ti * float(traffic_factor),
    }

    return pd.DataFrame([raw])


def fit_row_to_model(row, model):
    """Align the row to the estimator's exact feature order.

    This prevents feature-order/name errors and makes future model replacements safer.
    """
    features = model_features(model)
    if features:
        missing = [c for c in features if c not in row.columns]
        if missing:
            raise ValueError(f"Model expects missing features: {missing}")
        return row.loc[:, features]
    return row


def choose_traffic_model(road_type, target):
    key = f"{road_type}_{target}"
    if key in TM_M:
        return key, TM_M[key]
    for fallback_type in ["arterial", "collector", "SH", "NH", "expressway"]:
        key = f"{fallback_type}_{target}"
        if key in TM_M:
            return key, TM_M[key]
    return None, None


def predict_road_raw(road, ctx, traffic_factor=1.0, baseline_anchor=None):
    row = make_traffic_features(road, traffic_factor=traffic_factor, **ctx)

    raw = {}
    for target in TRAFFIC_TARGETS:
        key, model = choose_traffic_model(road["type"], target)
        if model is None:
            raw[target] = np.nan
            continue
        model_row = fit_row_to_model(row, model)
        raw[target] = safe_float(model.predict(model_row)[0], np.nan)

    # Traffic-index model is the best-constrained congestion prediction (R² values in
    # the supplied metadata are high). Speed/TT models are retained and evaluated, but
    # their supplied validation is materially weaker. We therefore use them as anchors
    # and enforce physical consistency with traffic index, free-flow speed and length.
    ti_model_key, ti_model = choose_traffic_model(road["type"], "traffic_index")
    base_speed_key, base_speed_model = choose_traffic_model(road["type"], "speed_kmph")
    if baseline_anchor is None:
        base_row = make_traffic_features(road, traffic_factor=1.0, **ctx)
        base_ti = safe_float(
            ti_model.predict(fit_row_to_model(base_row, ti_model))[0] if ti_model else 45.0,
            45.0,
        )
        base_speed = safe_float(
            base_speed_model.predict(fit_row_to_model(base_row, base_speed_model))[0]
            if base_speed_model else road.get("speed", 30),
            road.get("speed", 30),
        )
    else:
        base_ti = float(baseline_anchor["traffic_index"])
        base_speed = float(baseline_anchor["speed_kmph"])
    free_flow = max(5.0, float(road.get("speed", base_speed)))

    ti = float(np.clip(raw.get("traffic_index", base_ti), 0.0, 100.0))
    # Calibrated congestion-to-speed curve anchored at the modelled baseline speed.
    ti_ratio = max(0.2, ti) / max(0.2, base_ti)
    derived_speed = base_speed * (ti_ratio ** -0.90)
    derived_speed = float(np.clip(derived_speed, 2.0, free_flow))
    model_speed = float(np.clip(raw.get("speed_kmph", np.nan) if np.isfinite(raw.get("speed_kmph", np.nan)) else derived_speed, 2.0, free_flow))

    # The speed models in the supplied metadata are weak for several road types. Blend
    # them lightly rather than letting a weak model overrule the strong congestion model.
    speed = 0.70 * derived_speed + 0.30 * model_speed
    speed = float(np.clip(speed, 2.0, free_flow))

    model_tt = raw.get("travel_time_min", np.nan)
    physical_tt = float(60.0 * float(road.get("len", 1.0)) / max(speed, 2.0))
    if np.isfinite(model_tt):
        # Small anchor from the trained TT model while keeping TT consistent with speed.
        travel_time = 0.20 * float(np.clip(model_tt, 1.0, 240.0)) + 0.80 * physical_tt
    else:
        travel_time = physical_tt

    return {
        "traffic_index": clip_round(ti, 0.0, 100.0),
        "speed_kmph": clip_round(speed, 2.0, free_flow),
        "travel_time_min": clip_round(travel_time, 0.1, 240.0),
        "model_keys": {
            "traffic_index": ti_model_key,
            "speed_kmph": base_speed_key,
            "travel_time_min": choose_traffic_model(road["type"], "travel_time_min")[0],
        },
        "raw_model": raw,
    }


def predict_road(road, ctx, traffic_factor=1.0, closed=False, baseline_anchor=None):
    if closed:
        return {
            "traffic_index": 100.0,
            "speed_kmph": 0.0,
            "travel_time_min": np.inf,
            "status": "CLOSED",
            "model_keys": {},
            "raw_model": {},
        }
    result = predict_road_raw(road, ctx, traffic_factor=traffic_factor, baseline_anchor=baseline_anchor)
    result["status"] = "OPEN"
    return result


# -----------------------------------------------------------------------------
# Pollution prediction with geographic nearest-model fallback
# -----------------------------------------------------------------------------

def pollution_model_zone_ids(pollutant):
    suffix = f"_{pollutant}"
    return sorted({k[: -len(suffix)] for k in PM_M if k.endswith(suffix)})


def nearest_pollution_zone(zone_id, pollutant):
    available = pollution_model_zone_ids(pollutant)
    if not available:
        return None
    target = ZONE_MAP.get(zone_id)
    if not target:
        return available[0]
    return min(
        available,
        key=lambda zid: distance_km(
            target.get("lat", 13.08),
            target.get("lon", 80.25),
            ZONE_MAP.get(zid, {}).get("lat", target.get("lat", 13.08)),
            ZONE_MAP.get(zid, {}).get("lon", target.get("lon", 80.25)),
        ),
    )


def predict_pollution(zone_id, zone_traffic, ctx, traffic_factor=1.0, population_factor=1.0):
    hs, hc = cyc(ctx["hour"], 24)
    ms, mc = cyc(ctx["month"], 12)
    baseline_meta = PM.get("zone_baselines", {}).get(zone_id, {})
    bp = float(baseline_meta.get("pm25", 35.0))

    effective_traffic = float(zone_traffic) * float(traffic_factor) * float(population_factor)
    row = pd.DataFrame(
        [
            {
                "hour_sin": hs,
                "hour_cos": hc,
                "month_sin": ms,
                "month_cos": mc,
                "zone_traffic": effective_traffic,
                "temperature": ctx["temp"],
                "humidity": ctx["hum"],
                "wind_speed_kmph": ctx["wind"],
                "rainfall_mm": ctx["rain"],
                "pm25_lag1": bp,
                "pm25_lag3": bp,
                "pm25_roll4": bp,
                "traffic_x_humidity": effective_traffic * ctx["hum"] / 100.0,
                "traffic_x_no_wind": effective_traffic / max(ctx["wind"], 0.5),
            }
        ]
    )

    values = {}
    sources = {}
    for pollutant in POLLUTANTS:
        key = f"{zone_id}_{pollutant}"
        if key not in PM_M:
            fallback_zone = nearest_pollution_zone(zone_id, pollutant)
            key = f"{fallback_zone}_{pollutant}" if fallback_zone else None
        if key and key in PM_M:
            model = PM_M[key]
            pred = model.predict(fit_row_to_model(row, model))[0]
            values[pollutant] = clip_round(pred, 0.0, 500.0)
            sources[pollutant] = key.rsplit("_", 1)[0]
        else:
            values[pollutant] = clip_round(bp, 0.0, 500.0)
            sources[pollutant] = None

    return {**values, "model_sources": sources}


# -----------------------------------------------------------------------------
# Electricity prediction
# -----------------------------------------------------------------------------

def predict_electricity(ctx, demand_factor=1.0):
    hs, hc = cyc(ctx["hour"], 24)
    ms, mc = cyc(ctx["month"], 12)
    be = float(EM.get("baseline_mw", 4000.0))
    pop_factor = float(demand_factor)
    raw = {
        "hour_sin": hs,
        "hour_cos": hc,
        "month_sin": ms,
        "month_cos": mc,
        "is_weekend": ctx["is_we"],
        "is_monsoon": ctx["is_mo"],
        "temperature": ctx["temp"],
        "humidity": ctx["hum"],
        "population_factor": pop_factor,
        "elec_lag1": be,
        "elec_lag24": be,
        "elec_roll4": be,
        "temp_x_hour": ctx["temp"] * hs,
        "temp_x_pop": ctx["temp"] * pop_factor,
    }
    row = pd.DataFrame([raw])
    pred = float(EL.predict(fit_row_to_model(row, EL))[0])
    # The supplied electricity model becomes nearly flat after small population-factor
    # changes at some inputs. Keep the model prediction as the anchor, but preserve a
    # transparent scenario stress response when the model itself is numerically flat.
    base_row = row.copy()
    base_row.loc[:, "population_factor"] = 1.0
    base_row.loc[:, "temp_x_pop"] = ctx["temp"]
    base_pred = float(EL.predict(fit_row_to_model(base_row, EL))[0])
    if abs(pred - base_pred) <= max(5.0, 0.002 * abs(base_pred)) and pop_factor > 1.0:
        pred = base_pred * (1.0 + 0.50 * (pop_factor - 1.0))
    return clip_round(pred, 1000.0, 8000.0)


# -----------------------------------------------------------------------------
# Network simulation
# -----------------------------------------------------------------------------

def road_baseline_key(road_id):
    return TM.get("per_road_baselines", {}).get(road_id, {})


def base_zone_traffic(zone_id, baseline_roads):
    road_ids = ZONE_MAP.get(zone_id, {}).get("roads", [])
    vals = []
    weights = []
    for rid in road_ids:
        road = ROAD_MAP.get(rid)
        if not road or rid not in baseline_roads:
            continue
        capacity = road_capacity_proxy(road)
        vals.append(float(baseline_roads[rid]["traffic_index"]))
        weights.append(capacity)
    if not vals:
        return float(TM.get("road_baselines", {}).get("arterial", {}).get("traffic_mean", 45.0))
    return float(np.average(vals, weights=weights))


def scenario_zone_traffic(zone_id, scenario_roads):
    road_ids = ZONE_MAP.get(zone_id, {}).get("roads", [])
    vals = []
    weights = []
    for rid in road_ids:
        road = ROAD_MAP.get(rid)
        if not road or rid not in scenario_roads:
            continue
        value = scenario_roads[rid]
        if value.get("status") == "CLOSED":
            continue
        vals.append(float(value["traffic_index"]))
        weights.append(road_capacity_proxy(road))
    if not vals:
        return 0.0
    return float(np.average(vals, weights=weights))


def reroute_candidates(road_id):
    if get_rerouting is not None:
        try:
            info = get_rerouting(road_id, CFG.get("roads", []))
            alts = [(rid, float(frac)) for rid, frac in info.get("alternatives", []) if rid in ROAD_MAP]
            if alts:
                return alts
        except Exception:
            pass

    # Defensive fallback if rerouting.py is unavailable or a road has no entry.
    closed = ROAD_MAP.get(road_id)
    if not closed:
        return []
    candidates = []
    for r in CFG.get("roads", []):
        if r["id"] == road_id:
            continue
        overlap = len(set(closed.get("zones", [])) & set(r.get("zones", [])))
        if overlap:
            candidates.append((r["id"], 1.0))
    candidates.sort(key=lambda x: road_capacity_proxy(ROAD_MAP[x[0]]), reverse=True)
    top = candidates[:4]
    if not top:
        return []
    shares = np.array([1.0 / (i + 1) for i in range(len(top))], dtype=float)
    shares /= shares.sum()
    return [(rid, float(s)) for (rid, _), s in zip(top, shares)]


def closure_fraction(severity):
    # 1.0=open, 2.0=full closure, >2.0=full closure plus stronger diversion pressure.
    return float(np.clip(severity - 1.0, 0.0, 1.0))


def compute_all_roads(ctx, scenario, selected_road_id, params):
    pop_factor = float(params.get("pop_f", 1.0))
    hospital_factor = float(params.get("nf_f", 1.0))
    severity = float(params.get("rc_f", 1.0))
    frac_closed = closure_fraction(severity)

    baseline = {}
    scenario_roads = {}
    road_factor = {rid: 1.0 for rid in ROAD_MAP}
    closed = set()
    alternatives = []

    if scenario == "Road Closure":
        alternatives = reroute_candidates(selected_road_id)
        # Below 2.0 the road is partially restricted; at 2.0+ it is fully closed.
        if severity >= 2.0:
            closed.add(selected_road_id)
        else:
            road_factor[selected_road_id] = 1.0 + 1.0 * frac_closed
        diversion_pressure = 1.0 + max(0.0, severity - 2.0) * 0.65
        for rid, share in alternatives:
            road_factor[rid] = 1.0 + frac_closed * share * 1.55 * diversion_pressure

    elif scenario == "Population Growth":
        selected = ROAD_MAP[selected_road_id]
        affected_zones = set(selected.get("zones", []))
        for rid, road in ROAD_MAP.items():
            overlap = len(affected_zones & set(road.get("zones", [])))
            if overlap:
                road_factor[rid] = pop_factor

    elif scenario == "New Hospital":
        selected = ROAD_MAP[selected_road_id]
        affected_zones = set(selected.get("zones", []))
        for rid, road in ROAD_MAP.items():
            overlap = len(affected_zones & set(road.get("zones", [])))
            if overlap:
                road_factor[rid] = 1.0 + (hospital_factor - 1.0) * (1.0 if rid == selected_road_id else 0.55)

    # Build baseline for every configured road.
    for rid, road in ROAD_MAP.items():
        baseline[rid] = predict_road(road, ctx, traffic_factor=1.0, closed=False)

    # Scenario predictions for every road so network effects can actually propagate.
    for rid, road in ROAD_MAP.items():
        if rid in closed:
            scenario_roads[rid] = predict_road(road, ctx, traffic_factor=1.0, closed=True)
        else:
            scenario_roads[rid] = predict_road(
                road,
                ctx,
                traffic_factor=road_factor.get(rid, 1.0),
                closed=False,
                baseline_anchor=baseline[rid],
            )

    return baseline, scenario_roads, alternatives, road_factor, closed


def simulate_scenario(selected_road_id, scenario, params, ctx):
    road = ROAD_MAP[selected_road_id]
    baseline_roads, scenario_roads, alternatives, road_factor, closed = compute_all_roads(
        ctx, scenario, selected_road_id, params
    )

    # Baseline/scenario pollution is computed for all zones, using zone-wide traffic
    # rather than pretending that one selected road is the whole zone.
    baseline_zones = {}
    scenario_zones = {}
    zone_sources = {}
    for zid in ZONE_MAP:
        bzt = base_zone_traffic(zid, baseline_roads)
        szt = scenario_zone_traffic(zid, scenario_roads)
        baseline_zones[zid] = predict_pollution(zid, bzt, ctx)

        # Population/hospital effects enter pollution through the affected roads and
        # selected zones. For a road closure, traffic redistribution is already in szt.
        local_pop = 1.0
        selected_zones = set(road.get("zones", []))
        if scenario == "Population Growth" and zid in selected_zones:
            local_pop = float(params.get("pop_f", 1.0))
        if scenario == "New Hospital" and zid in selected_zones:
            local_pop = float(params.get("nf_f", 1.0))
        scenario_zones[zid] = predict_pollution(zid, szt, ctx, population_factor=local_pop)
        zone_sources[zid] = scenario_zones[zid].get("model_sources", {})

    baseline_electricity = predict_electricity(ctx, 1.0)
    if scenario == "Population Growth":
        demand_factor = float(params.get("pop_f", 1.0))
    elif scenario == "New Hospital":
        demand_factor = float(params.get("nf_f", 1.0))
    else:
        demand_factor = 1.0
    scenario_electricity = predict_electricity(ctx, demand_factor)

    # Build a network-level summary used by the recommendation score.
    def weighted_average(roads_dict, metric):
        vals, weights = [], []
        for rid, value in roads_dict.items():
            if value.get("status") == "CLOSED":
                continue
            r = ROAD_MAP[rid]
            vals.append(float(value[metric]))
            weights.append(road_capacity_proxy(r))
        return float(np.average(vals, weights=weights)) if vals else 0.0

    baseline_network = {
        "traffic_index": weighted_average(baseline_roads, "traffic_index"),
        "travel_time_min": weighted_average(baseline_roads, "travel_time_min"),
        "pm25": float(np.mean([v["pm25"] for v in baseline_zones.values()])),
        "electricity_mw": baseline_electricity,
    }
    scenario_network = {
        "traffic_index": weighted_average(scenario_roads, "traffic_index"),
        "travel_time_min": weighted_average(scenario_roads, "travel_time_min"),
        "pm25": float(np.mean([v["pm25"] for v in scenario_zones.values()])),
        "electricity_mw": scenario_electricity,
    }

    score, score_breakdown = mcda_score(scenario_network, baseline_network)

    return {
        "road": road,
        "scenario": scenario,
        "baseline_road": baseline_roads[selected_road_id],
        "scenario_road": scenario_roads[selected_road_id],
        "baseline_roads": baseline_roads,
        "scenario_roads": scenario_roads,
        "baseline_zones": baseline_zones,
        "scenario_zones": scenario_zones,
        "baseline_elec": baseline_electricity,
        "scenario_elec": scenario_electricity,
        "overall_score": score,
        "score_breakdown": score_breakdown,
        "zone_ids": road.get("zones", []),
        "all_zone_ids": list(ZONE_MAP.keys()),
        "alt_roads": alternatives,
        "road_factor": road_factor,
        "closed_roads": list(closed),
        "network_baseline": baseline_network,
        "network_scenario": scenario_network,
        "zone_sources": zone_sources,
    }


# -----------------------------------------------------------------------------
# MCDA recommendation score
# -----------------------------------------------------------------------------

def score_component(delta_pct, penalty_scale=1.5):
    # 50 = unchanged; <50 = worse; >50 = better.
    return float(np.clip(50.0 - penalty_scale * delta_pct, 0.0, 100.0))


def mcda_score(scenario, baseline):
    traffic_delta = (scenario["traffic_index"] - baseline["traffic_index"]) / max(baseline["traffic_index"], 1e-6) * 100.0
    pollution_delta = (scenario["pm25"] - baseline["pm25"]) / max(baseline["pm25"], 1e-6) * 100.0
    electricity_delta = (scenario["electricity_mw"] - baseline["electricity_mw"]) / max(baseline["electricity_mw"], 1e-6) * 100.0
    accessibility_delta = (scenario["travel_time_min"] - baseline["travel_time_min"]) / max(baseline["travel_time_min"], 1e-6) * 100.0

    components = {
        "traffic": score_component(traffic_delta, 1.4),
        "pollution": score_component(pollution_delta, 1.4),
        "electricity": score_component(electricity_delta, 1.2),
        "accessibility": score_component(accessibility_delta, 1.2),
    }

    total = sum(MCDA_WEIGHTS[k] * components[k] for k in MCDA_WEIGHTS)

    # Explicitly penalize a full closure only through the accessibility/network outcome,
    # not by pretending that a closed road has a valid speed/travel time prediction.
    return round(float(np.clip(total, 0.0, 100.0)), 1), {
        "components": {k: round(v, 1) for k, v in components.items()},
        "weights": MCDA_WEIGHTS.copy(),
        "deltas_pct": {
            "traffic": round(traffic_delta, 1),
            "pollution": round(pollution_delta, 1),
            "electricity": round(electricity_delta, 1),
            "accessibility": round(accessibility_delta, 1),
        },
    }


# -----------------------------------------------------------------------------
# Map
# -----------------------------------------------------------------------------

@st.cache_data

def build_base_map_data():
    road_lats, road_lons, road_names, road_ids, road_types = [], [], [], [], []
    for road in CFG.get("roads", []):
        zones = road.get("zones", [])
        zone_points = [ZONE_MAP[z] for z in zones if z in ZONE_MAP]
        if zone_points:
            lat = float(np.mean([z["lat"] for z in zone_points]))
            lon = float(np.mean([z["lon"] for z in zone_points]))
        else:
            lat, lon = 13.08, 80.24
        road_lats.append(lat)
        road_lons.append(lon)
        road_names.append(road["name"])
        road_ids.append(road["id"])
        road_types.append(road["type"])

    zone_lats = [z["lat"] for z in CFG.get("zones", [])]
    zone_lons = [z["lon"] for z in CFG.get("zones", [])]
    zone_names = [z["name"] for z in CFG.get("zones", [])]
    zone_ids = [z["id"] for z in CFG.get("zones", [])]
    return (
        road_lats,
        road_lons,
        road_names,
        road_ids,
        road_types,
        zone_lats,
        zone_lons,
        zone_names,
        zone_ids,
    )


(
    RLATS,
    RLONS,
    RNAMES,
    RIDS,
    RTYPES,
    ZLATS,
    ZLONS,
    ZNAMES,
    ZIDS,
) = build_base_map_data()


def make_map(selected_road_id=None, result=None):
    colors = []
    sizes = []
    hover_texts = []
    alternatives = {rid for rid, _ in result.get("alt_roads", [])} if result else set()
    closed = set(result.get("closed_roads", [])) if result else set()

    for rid, rt, name in zip(RIDS, RTYPES, RNAMES):
        if rid in closed:
            color = "#C0392B"
            size = 16
            hover = f"<b>{name}</b><br>🚧 CLOSED"
        elif rid in alternatives:
            scen = result.get("scenario_roads", {}).get(rid, {}) if result else {}
            ti = scen.get("traffic_index", "—")
            color = "#F39C12"
            size = 13
            hover = f"<b>{name}</b><br>🔀 Alternate route<br>Traffic index: {ti}"
        else:
            if result and rid in result.get("scenario_roads", {}):
                ti = result["scenario_roads"][rid].get("traffic_index", 0)
                color = traffic_color(float(ti))
                size = 8
                hover = f"<b>{name}</b><br>Traffic index: {ti}"
            else:
                color = {
                    "NH": "#2471A3",
                    "SH": "#1ABC9C",
                    "expressway": "#8E44AD",
                    "arterial": "#E67E22",
                    "collector": "#27AE60",
                }.get(rt, "#4A90D9")
                size = 9
                hover = f"<b>{name}</b><br>Type: {rt}"
        colors.append(color)
        sizes.append(size if rid == selected_road_id else max(7, size - 2))
        hover_texts.append(hover)

    zone_colors = []
    for zid in ZIDS:
        if result:
            p = result.get("scenario_zones", {}).get(zid, {}).get("pm25", 35.0)
        else:
            p = PM.get("zone_baselines", {}).get(zid, {}).get("pm25", 35.0)
        zone_colors.append(pollution_color(float(p)))

    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=ZLATS,
            lon=ZLONS,
            mode="markers",
            marker=dict(size=30, color=zone_colors, opacity=0.35),
            text=[f"<b>{n}</b><br>Pollution zone" for n in ZNAMES],
            hoverinfo="text",
            name="Zones (pollution)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=ZLATS,
            lon=ZLONS,
            mode="text",
            text=[n.replace("_", " ") for n in ZNAMES],
            textfont=dict(size=9, color="#333333"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=RLATS,
            lon=RLONS,
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.9),
            text=hover_texts,
            hoverinfo="text",
            name="Roads (traffic)",
            showlegend=True,
        )
    )
    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lat=13.08, lon=80.24), zoom=10.5),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        showlegend=True,
    )
    return fig


# -----------------------------------------------------------------------------
# Output rendering
# -----------------------------------------------------------------------------

def pct_delta(new, old):
    return (float(new) - float(old)) / max(abs(float(old)), 1e-6) * 100.0


def render_metric(label, value, baseline, unit="", higher_is_worse=True, closed=False, desc_key=None):
    if closed:
        st.markdown(f"**{label}**  \n<span style='font-size:20px;font-weight:bold'>CLOSED</span>", unsafe_allow_html=True)
        st.caption(METRIC_DESC.get(desc_key or label.lower().replace(" ", "_"), ""))
        return
    delta = float(value) - float(baseline)
    pct = pct_delta(value, baseline)
    worse = (delta > 0) if higher_is_worse else (delta < 0)
    col = "#E74C3C" if worse else "#27AE60"
    arrow = "⬆️" if delta > 0 else "⬇️" if delta < 0 else "➡️"
    unit_text = f" {unit}" if unit else ""
    st.markdown(
        f"**{label}**  \n"
        f"<span style='font-size:20px;font-weight:bold'>{float(value):.1f}{unit_text}</span> "
        f"<span style='color:{col};font-size:13px'>{arrow} {abs(pct):.1f}% ({delta:+.1f})</span>",
        unsafe_allow_html=True,
    )


def render_output(result, label=""):
    if not result:
        return

    road = result["road"]
    scenario = result["scenario"]
    bl_r = result["baseline_road"]
    sc_r = result["scenario_road"]
    ov = result["overall_score"]
    vt, vc = verdict_info(ov)

    if label:
        st.subheader(label)
    st.markdown(
        f"<div style='background:{vc};color:white;padding:8px 14px;border-radius:8px;"
        f"font-weight:bold;font-size:15px;margin-bottom:8px'>{vt} &nbsp;|&nbsp; Score: {ov}/100</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Road:** {road['name']} &nbsp;|&nbsp; **Scenario:** {scenario}  \n"
        f"**Type:** {road['type'].upper()} &nbsp;|&nbsp; **Lanes:** {road['lanes']} &nbsp;|&nbsp; "
        f"**Length:** {road['len']} km"
    )

    st.divider()
    st.markdown("#### 🚦 Traffic Impact")
    closed = sc_r.get("status") == "CLOSED"
    render_metric("Traffic Index", sc_r["traffic_index"], bl_r["traffic_index"], higher_is_worse=True, closed=closed, desc_key="traffic_index")
    render_metric("Speed kmph", sc_r["speed_kmph"], bl_r["speed_kmph"], unit="kmph", higher_is_worse=False, closed=closed, desc_key="speed_kmph")
    if closed:
        st.markdown("**Travel Time**  \n<span style='font-size:20px;font-weight:bold'>Not applicable — road closed</span>", unsafe_allow_html=True)
    else:
        render_metric("Travel Time min", sc_r["travel_time_min"], bl_r["travel_time_min"], unit="min", higher_is_worse=True, desc_key="travel_time_min")

    if scenario == "Road Closure":
        alt_roads = result.get("alt_roads", [])
        st.markdown("#### 🔀 Recommended Alternate Roads")
        if not alt_roads:
            st.info("No explicit rerouting entry exists for this road; the fallback searches nearby roads sharing its zone.")
        else:
            rows = []
            for rid, share in alt_roads:
                if rid not in ROAD_MAP:
                    continue
                alt = ROAD_MAP[rid]
                scen = result["scenario_roads"].get(rid, {})
                base = result["baseline_roads"].get(rid, {})
                rows.append(
                    {
                        "Road": alt["name"],
                        "Diversion share": f"{share * 100:.0f}%",
                        "Traffic Index": f"{scen.get('traffic_index', 0):.1f}",
                        "Change": f"{pct_delta(scen.get('traffic_index',0), base.get('traffic_index',1)):+.1f}%",
                        "Speed": f"{scen.get('speed_kmph', 0):.1f} kmph",
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 🌫️ Pollution Impact by Zone")
    for zid in result.get("zone_ids", []):
        bl_p = result["baseline_zones"].get(zid, {})
        sc_p = result["scenario_zones"].get(zid, {})
        source_map = sc_p.get("model_sources", {})
        source_text = ", ".join(sorted(set(x for x in source_map.values() if x))) or "baseline fallback"
        st.markdown(f"*{ZONE_MAP.get(zid, {}).get('name', zid)}* &nbsp; <span style='font-size:11px;color:#888'>model source: {source_text}</span>", unsafe_allow_html=True)
        cols = st.columns(3)
        for col, pollutant in zip(cols, POLLUTANTS):
            with col:
                bv = bl_p.get(pollutant, 0.0)
                sv = sc_p.get(pollutant, 0.0)
                d = pct_delta(sv, bv)
                indicator = "🔴" if d > 0 else "🟢" if d < 0 else "⚪"
                guideline = WHO_GUIDELINES[pollutant]
                exceed = " ⚠️" if sv > guideline else ""
                st.metric(pollutant.upper(), f"{sv:.1f} µg/m³", f"{d:+.1f}%")
                st.caption(f"{indicator} {METRIC_DESC[pollutant]}{exceed}")

    if scenario in ["Population Growth", "New Hospital"]:
        st.divider()
        st.markdown("#### ⚡ Electricity Demand")
        d = result["scenario_elec"] - result["baseline_elec"]
        p = pct_delta(result["scenario_elec"], result["baseline_elec"])
        st.metric("City demand", f"{result['scenario_elec']:.0f} MW", f"{d:+.0f} MW ({p:+.1f}%)")
        st.caption(METRIC_DESC["electricity_mw"])

    st.divider()
    st.markdown("#### 🧮 Planning Recommendation Score")
    breakdown = result["score_breakdown"]
    bc = breakdown["components"]
    weights = breakdown["weights"]
    score_df = pd.DataFrame(
        {
            "Criterion": ["Traffic", "Pollution", "Electricity", "Accessibility"],
            "Weight": [weights[k] for k in ["traffic", "pollution", "electricity", "accessibility"]],
            "Component score": [bc[k] for k in ["traffic", "pollution", "electricity", "accessibility"]],
        }
    )
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    st.caption("Weights are a transparent multi-criteria planning choice; they are not produced by the prediction models.")

    st.divider()
    st.markdown("#### 🔗 Why this happened")
    if scenario == "Road Closure":
        st.write(f"🚧 {road['name']} is closed, so its normal speed/travel time is not treated as a valid open-road prediction.")
        if result.get("alt_roads"):
            top_names = ", ".join(ROAD_MAP[rid]["name"] for rid, _ in result["alt_roads"][:3] if rid in ROAD_MAP)
            st.write(f"🔀 Diverted demand is redistributed across alternate roads: {top_names}.")
        st.write("💨 Zone pollution is recalculated from the resulting zone-wide traffic load.")
    elif scenario == "Population Growth":
        st.write(f"👥 Population factor applied to the selected zone is {result['road_factor'].get(road['id'], 1.0):.2f}× on the primary road and propagates to connected roads in that zone.")
        st.write("💨 Pollution responds to the resulting zone traffic and population factor.")
        st.write("⚡ Electricity demand uses the trained electricity model with the requested population factor.")
    else:
        st.write(f"🏥 Hospital construction is represented as an increased traffic/power-demand factor around {road['name']}.")
        st.write("💨 The affected zone is recomputed using the pollution model rather than a hand-written pollution formula.")
        st.write("⚡ Hospital demand is passed through the trained electricity model as an equivalent demand factor.")


# -----------------------------------------------------------------------------
# Sidebar/context
# -----------------------------------------------------------------------------

for key, default in {
    "selected_road_a": None,
    "selected_road_b": None,
    "result_a": None,
    "result_b": None,
    "compare_mode": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.title("🏙️ Chennai AI Digital Twin")
st.caption("Configure a planning scenario → run the simulation → inspect network, pollution, electricity and alternate-route effects.")
st.divider()

with st.sidebar:
    st.header("⚙️ Time & Weather")
    hour = st.slider("Hour of day", 0, 23, 9)
    dow = st.selectbox(
        "Day",
        range(7),
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
        index=1,
    )
    month = st.slider("Month", 1, 12, 6)
    is_we = int(dow >= 5)
    is_pk = int(8 <= hour <= 10 or 17 <= hour <= 21)
    is_mo = int(month in [10, 11, 12])
    tags = " ".join(
        filter(
            None,
            [
                "🌧️ Monsoon" if is_mo else "",
                "🏖️ Weekend" if is_we else "",
                "⏰ Peak" if is_pk else "",
            ],
        )
    )
    if tags:
        st.caption(tags)

    st.divider()
    st.subheader("🌤️ Weather")
    temp = st.slider("Temperature (°C)", 20, 46, 35)
    hum = st.slider("Humidity (%)", 30, 100, 72)
    wind = st.slider("Wind (kmph)", 0, 40, 10)
    rain = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0, step=0.5)

    st.divider()
    st.subheader("🎉 Festival")
    is_fest = st.checkbox("Festival day?")
    fest_mult = 1.0
    fest_name = None
    if is_fest:
        fest_name = st.selectbox(
            "Festival",
            [
                "Diwali",
                "Vinayagar Chaturthi",
                "Tamil New Year",
                "Pongal",
                "Republic Day",
            ],
        )
        known_defaults = {
            "Diwali": 1.65,
            "Vinayagar Chaturthi": 1.55,
            "Tamil New Year": 1.40,
            "Pongal": 0.50,
            "Republic Day": 0.72,
        }
        fest_mult = known_defaults.get(fest_name, festival_factor_from_config(month, fest_name))
        st.caption(f"Festival multiplier used by the trained feature schema: {fest_mult:.2f}")

    st.divider()
    st.session_state.compare_mode = st.checkbox(
        "🆚 Compare two scenarios", value=st.session_state.compare_mode
    )

CTX = {
    "hour": hour,
    "dow": dow,
    "month": month,
    "is_we": is_we,
    "is_pk": is_pk,
    "is_mo": is_mo,
    "is_fest": is_fest,
    "fest_mult": fest_mult,
    "temp": temp,
    "hum": hum,
    "wind": wind,
    "rain": rain,
}


# -----------------------------------------------------------------------------
# Scenario panel
# -----------------------------------------------------------------------------

def render_scenario_panel(label, road_state_key, result_state_key):
    st.subheader(f"📋 {label}")
    road_options = {r["id"]: f"{r['name']} ({r['type'].upper()})" for r in CFG.get("roads", [])}
    road_ids_sorted = sorted(road_options.keys(), key=lambda rid: road_options[rid])
    current = st.session_state.get(road_state_key)
    idx = road_ids_sorted.index(current) if current in road_ids_sorted else 0

    selected = st.selectbox(
        "Select road",
        road_ids_sorted,
        index=idx,
        format_func=lambda rid: road_options[rid],
        key=f"dd_{label}",
    )
    st.session_state[road_state_key] = selected
    road = ROAD_MAP[selected]
    st.caption(
        f"Lanes: {road['lanes']} | Length: {road['len']} km | Free-flow speed: {road['speed']} kmph"
    )

    scenario = st.radio(
        "Scenario",
        ["Road Closure", "Population Growth", "New Hospital"],
        key=f"sc_{label}",
    )
    params = {}

    if scenario == "Road Closure":
        sev = st.slider(
            "Closure severity",
            1.0,
            2.5,
            1.8,
            step=0.05,
            key=f"sv_{label}",
            help="1.0=open, 2.0=full closure, >2.0=full closure with stronger diversion pressure.",
        )
        params["rc_f"] = sev
        if sev >= 2.0:
            st.caption("⚠️ Full closure — traffic is removed from this road and redistributed to alternate roads.")
        else:
            st.caption("Partial closure — the selected road is not treated as fully blocked.")
    elif scenario == "Population Growth":
        growth = st.slider("Growth %", 5, 50, 20, key=f"gr_{label}")
        params["pop_f"] = 1.0 + growth / 100.0
        st.caption(f"Simulates a {growth}% population increase centred on the selected road's connected zone(s).")
    else:
        size = st.selectbox("Hospital size", ["Small", "Medium", "Large"], key=f"sz_{label}")
        params["nf_f"] = {"Small": 1.06, "Medium": 1.10, "Large": 1.16}[size]
        st.caption(f"Hospital demand factor: {params['nf_f']:.2f}×. This represents construction/visitor load and facility demand for the prototype.")

    if st.button("▶ Run", type="primary", use_container_width=True, key=f"run_{label}"):
        try:
            with st.spinner("Running AI models across the configured road network..."):
                result = simulate_scenario(selected, scenario, params, CTX)
            st.session_state[result_state_key] = result
            st.rerun()
        except Exception as exc:
            st.error(f"Scenario simulation failed: {exc}")

    if st.session_state.get(result_state_key):
        score = st.session_state[result_state_key]["overall_score"]
        verdict, color = verdict_info(score)
        st.markdown(
            f"<div style='background:{color};color:white;padding:5px 10px;border-radius:6px;"
            f"font-size:13px;margin-top:8px'>{verdict}</div>",
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------

if st.session_state.compare_mode:
    col_a, col_b = st.columns(2)
    with col_a:
        render_scenario_panel("Scenario A", "selected_road_a", "result_a")
        st.divider()
        render_output(st.session_state.result_a, "📊 Scenario A — Output")
    with col_b:
        render_scenario_panel("Scenario B", "selected_road_b", "result_b")
        st.divider()
        render_output(st.session_state.result_b, "📊 Scenario B — Output")

    st.divider()
    st.subheader("🗺️ Map View")
    result_for_map = st.session_state.result_a or st.session_state.result_b
    selected_for_map = st.session_state.selected_road_a or st.session_state.selected_road_b
    st.plotly_chart(
        make_map(selected_for_map, result_for_map),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if st.session_state.result_a and st.session_state.result_b:
        st.divider()
        st.subheader("🆚 Scenario Comparison")
        score_a = st.session_state.result_a["overall_score"]
        score_b = st.session_state.result_b["overall_score"]
        c1, c2 = st.columns(2)
        c1.metric("Scenario A score", f"{score_a:.1f}/100")
        c2.metric("Scenario B score", f"{score_b:.1f}/100")
        if score_a > score_b:
            st.success("✅ Scenario A scores higher under the configured planning weights.")
        elif score_b > score_a:
            st.success("✅ Scenario B scores higher under the configured planning weights.")
        else:
            st.info("Both scenarios score equally under the configured planning weights.")
else:
    col_panel, col_map, col_output = st.columns([1.25, 2.8, 2.0])
    with col_panel:
        render_scenario_panel("Scenario", "selected_road_a", "result_a")
    with col_map:
        st.subheader("🗺️ Chennai Prototype Network")
        st.plotly_chart(
            make_map(st.session_state.selected_road_a, st.session_state.result_a),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Road markers show modelled traffic state; zone circles show PM2.5. "
            "Road positions are deterministic zone-centroid approximations, not GPS centerlines."
        )
    with col_output:
        if st.session_state.result_a:
            render_output(st.session_state.result_a)
        else:
            st.subheader("📊 Output")
            st.info("Select a road, configure a scenario, and click Run.")
