
"""
engine/predictor.py
--------------------
Core prediction engine for the Chennai Digital Twin.
Loads all models once at startup. Called by the dashboard.
"""
import numpy as np, pandas as pd, joblib, json, os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_all_models():
    with open(f"{_BASE}/models/traffic/traffic_meta.json") as f: TM=json.load(f)
    with open(f"{_BASE}/models/pollution/pollution_meta.json") as f: PM=json.load(f)
    with open(f"{_BASE}/models/electricity/electricity_meta.json") as f: EM=json.load(f)
    with open(f"{_BASE}/data/processed/city_config.json") as f: CFG=json.load(f)

    traffic_m, pollution_m = {}, {}
    for rtype in ["NH","SH","expressway","arterial","collector"]:
        for target in ["traffic_index","speed_kmph","travel_time_min"]:
            p = f"{_BASE}/models/traffic/{rtype}_{target}_model.pkl"
            if os.path.exists(p): traffic_m[f"{rtype}_{target}"] = joblib.load(p)
    for zone in PM["zone_models"]:
        for poll in ["pm25","pm10","no2"]:
            p = f"{_BASE}/models/pollution/{zone}_{poll}_model.pkl"
            if os.path.exists(p): pollution_m[f"{zone}_{poll}"] = joblib.load(p)
    elec_m = joblib.load(f"{_BASE}/models/electricity/electricity_model.pkl")
    return traffic_m, pollution_m, elec_m, TM, PM, EM, CFG

_TM_MODELS, _PM_MODELS, _ELEC_MODEL, _TM_META, _PM_META, _EM_META, _CFG = _load_all_models()

def _c(v,mx): return np.sin(2*np.pi*v/mx), np.cos(2*np.pi*v/mx)

def predict_road(road_id, road_type, lanes, length_km, hour, dow, month,
                 is_weekend, is_peak_hour, is_monsoon, is_festival, festival_mult,
                 temp, humidity, wind, rainfall,
                 road_closure_factor=1.0, population_factor=1.0):
    hs,hc=_c(hour,24); ds,dc=_c(dow,7); ms,mc=_c(month,12)
    bl = _TM_META["road_baselines"].get(road_type, {})
    base_t = bl.get("traffic_mean", 45.0)
    row = pd.DataFrame([{
        "hour_sin":hs,"hour_cos":hc,"dow_sin":ds,"dow_cos":dc,"month_sin":ms,"month_cos":mc,
        "is_weekend":is_weekend,"is_peak_hour":is_peak_hour,"is_monsoon":is_monsoon,
        "is_festival":int(is_festival),"festival_mult":festival_mult,
        "road_type_enc":{"NH":0,"SH":1,"expressway":2,"arterial":3,"collector":4}.get(road_type,3),
        "lanes":lanes,"length_km":length_km,
        "temperature":temp,"humidity":humidity,"wind_speed_kmph":wind,"rainfall_mm":rainfall,
        "traffic_lag1": base_t*road_closure_factor*population_factor,
        "traffic_lag3": base_t*road_closure_factor,
        "traffic_lag24":base_t,
        "traffic_roll4":base_t*road_closure_factor,
    }])
    key_t = f"{road_type}_traffic_index"
    key_s = f"{road_type}_speed_kmph"
    key_tt= f"{road_type}_travel_time_min"
    if key_t not in _TM_MODELS: key_t=key_s=key_tt="arterial_traffic_index"
    ti  = float(np.clip(_TM_MODELS.get(key_t,  _TM_MODELS.get("arterial_traffic_index")).predict(row)[0],2,100))
    sp  = float(np.clip(_TM_MODELS.get(key_s,  _TM_MODELS.get("arterial_speed_kmph")).predict(row)[0],2,80))
    tt  = float(np.clip(_TM_MODELS.get(key_tt, _TM_MODELS.get("arterial_travel_time_min")).predict(row)[0],1,200))
    return {"traffic_index":round(ti,1),"speed_kmph":round(sp,1),"travel_time_min":round(tt,1)}

def predict_zone_pollution(zone_id, zone_traffic, hour, month,
                            is_monsoon, temp, humidity, wind, rainfall,
                            road_closure_factor=1.0, population_factor=1.0):
    hs,hc=_c(hour,24); ms,mc=_c(month,12)
    bl  = _PM_META["zone_baselines"].get(zone_id,{})
    bp  = bl.get("pm25", 35.0)
    eff_traffic = zone_traffic * road_closure_factor * population_factor
    row = pd.DataFrame([{
        "hour_sin":hs,"hour_cos":hc,"month_sin":ms,"month_cos":mc,
        "zone_traffic":eff_traffic,
        "temperature":temp,"humidity":humidity,"wind_speed_kmph":wind,"rainfall_mm":rainfall,
        "pm25_lag1":bp,"pm25_lag3":bp,"pm25_roll4":bp,
        "traffic_x_humidity":eff_traffic*humidity/100,
        "traffic_x_no_wind": eff_traffic/max(wind,0.5),
    }])
    def get_pred(poll):
        key = f"{zone_id}_{poll}"
        if key not in _PM_MODELS:
            # fallback to nearest zone
            fallback = [k for k in _PM_MODELS if k.endswith(f"_{poll}")]
            if not fallback: return bp
            key = fallback[0]
        return float(np.clip(_PM_MODELS[key].predict(row)[0],0,500))
    return {"pm25":round(get_pred("pm25"),1),"pm10":round(get_pred("pm10"),1),"no2":round(get_pred("no2"),1)}

def predict_electricity(hour, month, is_weekend, is_monsoon,
                         temp, humidity, population_factor=1.0):
    hs,hc=_c(hour,24); ms,mc=_c(month,12)
    be = _EM_META["baseline_mw"]
    row = pd.DataFrame([{
        "hour_sin":hs,"hour_cos":hc,"month_sin":ms,"month_cos":mc,
        "is_weekend":is_weekend,"is_monsoon":is_monsoon,
        "temperature":temp,"humidity":humidity,
        "population_factor":population_factor,
        "elec_lag1":be,"elec_lag24":be,"elec_roll4":be,
        "temp_x_hour":temp*hs,"temp_x_pop":temp*population_factor,
    }])
    return {"electricity_mw": round(float(np.clip(_ELEC_MODEL.predict(row)[0],1000,8000)),1)}
