
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib, json, os

st.set_page_config(
    page_title="Chennai Digital Twin",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Metric descriptions ──────────────────────────────────────────────────────
METRIC_DESC = {
    "traffic_index":   "Congestion level 0–100. Higher = more congested. Above 70 is severe.",
    "speed_kmph":      "Average vehicle speed on this road. Lower speed = more congestion.",
    "travel_time_min": "Estimated travel time for the full length of this road segment.",
    "pm25":  "PM2.5 — Fine particles under 2.5 µm. Penetrate deep into lungs. WHO 24h limit: 15 µg/m³.",
    "pm10":  "PM10 — Coarse particles under 10 µm. Cause respiratory irritation. WHO 24h limit: 45 µg/m³.",
    "no2":   "NO₂ — Nitrogen dioxide. Mainly from vehicle exhaust. WHO annual limit: 10 µg/m³.",
    "electricity_mw": "City electricity demand in MW. Chennai peak is ~6,000 MW (TANGEDCO).",
}
WHO = {"pm25": 15.0, "pm10": 45.0, "no2": 25.0}
TYPE_ENC = {"NH":0,"SH":1,"expressway":2,"arterial":3,"collector":4}

# ── Load models ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource(show_spinner="Loading AI models — please wait...")
def load_all():
    with open(f"{BASE}/models/traffic/traffic_meta.json") as f:   TM=json.load(f)
    with open(f"{BASE}/models/pollution/pollution_meta.json") as f: PM=json.load(f)
    with open(f"{BASE}/models/electricity/electricity_meta.json") as f: EM=json.load(f)
    with open(f"{BASE}/data/processed/city_config.json") as f:    CFG=json.load(f)
    TM_M, PM_M = {}, {}
    for rt in ["NH","SH","expressway","arterial","collector"]:
        for tg in ["traffic_index","speed_kmph","travel_time_min"]:
            p = f"{BASE}/models/traffic/{rt}_{tg}_model.pkl"
            if os.path.exists(p): TM_M[f"{rt}_{tg}"] = joblib.load(p)
    for zone in PM["zone_models"]:
        for poll in ["pm25","pm10","no2"]:
            p = f"{BASE}/models/pollution/{zone}_{poll}_model.pkl"
            if os.path.exists(p): PM_M[f"{zone}_{poll}"] = joblib.load(p)
    EL = joblib.load(f"{BASE}/models/electricity/electricity_model.pkl")
    return TM_M, PM_M, EL, TM, PM, EM, CFG

TM_M, PM_M, EL, TM, PM, EM, CFG = load_all()
ROAD_MAP = {r["id"]: r for r in CFG["roads"]}
ZONE_MAP = {z["id"]: z for z in CFG["zones"]}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _c(v, mx): return np.sin(2*np.pi*v/mx), np.cos(2*np.pi*v/mx)

def traffic_color(ti):
    if ti < 30: return "#27AE60"
    if ti < 50: return "#F1C40F"
    if ti < 70: return "#E67E22"
    return "#E74C3C"

def pollution_color(pm25):
    if pm25 < 15:  return "#27AE60"
    if pm25 < 35:  return "#F1C40F"
    if pm25 < 60:  return "#E67E22"
    return "#E74C3C"

def verdict_info(score):
    if score >= 55: return "🟢 RECOMMENDED", "#27AE60"
    if score >= 42: return "🟡 ACCEPTABLE WITH MITIGATION", "#F39C12"
    return "🔴 NOT RECOMMENDED", "#E74C3C"

def make_traffic_row(road, hour, dow, month, is_we, is_pk, is_mo,
                     is_fest, fest_mult, temp, hum, wind, rain, rc_f, pop_f):
    hs,hc=_c(hour,24); ds,dc=_c(dow,7); ms,mc=_c(month,12)
    rt = road["type"]
    bt = TM["road_baselines"].get(rt,{}).get("traffic_mean",45.0)
    return pd.DataFrame([{
        "hour_sin":hs,"hour_cos":hc,"dow_sin":ds,"dow_cos":dc,
        "month_sin":ms,"month_cos":mc,
        "is_weekend":is_we,"is_peak_hour":is_pk,"is_monsoon":is_mo,
        "is_festival":int(is_fest),"festival_mult":float(fest_mult),
        "road_type_enc":TYPE_ENC.get(rt,3),
        "lanes":road["lanes"],"length_km":road["len"],
        "temperature":temp,"humidity":hum,"wind_speed_kmph":wind,"rainfall_mm":rain,
        "traffic_lag1":bt*rc_f*pop_f,"traffic_lag3":bt*rc_f,
        "traffic_lag24":bt,"traffic_roll4":bt*rc_f,
    }])

def predict_road(road, hour, dow, month, is_we, is_pk, is_mo,
                 is_fest, fest_mult, temp, hum, wind, rain, rc_f=1.0, pop_f=1.0):
    rt  = road["type"]
    row = make_traffic_row(road,hour,dow,month,is_we,is_pk,is_mo,
                           is_fest,fest_mult,temp,hum,wind,rain,rc_f,pop_f)
    def get(tgt):
        k = f"{rt}_{tgt}"
        if k not in TM_M: k = f"arterial_{tgt}"
        return float(TM_M[k].predict(row)[0])
    ti  = round(float(np.clip(get("traffic_index"),2,100)),1)
    sp  = round(float(np.clip(get("speed_kmph"),2,road["speed"])),1)
    tt  = round(float(np.clip(get("travel_time_min"),1,200)),1)
    return {"traffic_index":ti,"speed_kmph":sp,"travel_time_min":tt}

def predict_poll(zone_id, zone_traffic, hour, month, is_mo,
                 temp, hum, wind, rain, rc_f=1.0, pop_f=1.0):
    hs,hc=_c(hour,24); ms,mc=_c(month,12)
    bl  = PM["zone_baselines"].get(zone_id,{})
    bp  = bl.get("pm25",35.0)
    eff = zone_traffic * rc_f * pop_f
    row = pd.DataFrame([{
        "hour_sin":hs,"hour_cos":hc,"month_sin":ms,"month_cos":mc,
        "zone_traffic":eff,"temperature":temp,"humidity":hum,
        "wind_speed_kmph":wind,"rainfall_mm":rain,
        "pm25_lag1":bp,"pm25_lag3":bp,"pm25_roll4":bp,
        "traffic_x_humidity":eff*hum/100,
        "traffic_x_no_wind":eff/max(wind,0.5),
    }])
    def gp(poll):
        k = f"{zone_id}_{poll}"
        if k not in PM_M:
            fb = [x for x in PM_M if x.endswith(f"_{poll}")]
            k  = fb[0] if fb else None
        return round(float(np.clip(PM_M[k].predict(row)[0],0,500)),1) if k else bp
    return {"pm25":gp("pm25"),"pm10":gp("pm10"),"no2":gp("no2")}

def predict_elec(hour, month, is_we, is_mo, temp, hum, pop_f=1.0):
    hs,hc=_c(hour,24); ms,mc=_c(month,12)
    be = EM["baseline_mw"]
    row = pd.DataFrame([{
        "hour_sin":hs,"hour_cos":hc,"month_sin":ms,"month_cos":mc,
        "is_weekend":is_we,"is_monsoon":is_mo,"temperature":temp,"humidity":hum,
        "population_factor":pop_f,"elec_lag1":be,"elec_lag24":be,"elec_roll4":be,
        "temp_x_hour":temp*hs,"temp_x_pop":temp*pop_f,
    }])
    return round(float(np.clip(EL.predict(row)[0],1000,8000)),1)

def score(result, baseline):
    tp =(result["traffic_index"]  -baseline["traffic_index"])  /max(baseline["traffic_index"],1) *100
    pp =(result["pm25"]           -baseline["pm25"])            /max(baseline["pm25"],1)          *100
    ep =(result.get("electricity_mw",4000)-baseline.get("electricity_mw",4000))/max(baseline.get("electricity_mw",4000),1)*100
    tc = result["travel_time_min"]-baseline["travel_time_min"]
    ts = max(0,min(100,50-tp*1.2)); ps=max(0,min(100,50-pp*1.5))
    es = max(0,min(100,50-ep*1.0)); ac=max(0,min(100,50-tc*2))
    if result["pm25"]>WHO["pm25"]: ps=max(0,ps-20)
    return round(ts*0.30+ps*0.40+es*0.20+ac*0.10,1)

# ── Build stable plotly map ───────────────────────────────────────────────────
@st.cache_data
def build_base_map_data():
    """Pre-compute all road and zone positions for the map. Cached so it never rerenders."""
    road_lats, road_lons, road_names, road_ids, road_types = [], [], [], [], []
    for road in CFG["roads"]:
        zones = road.get("zones",[])
        if not zones: continue
        z = ZONE_MAP.get(zones[0],{})
        lat = z.get("lat",13.08) + np.random.uniform(-0.018,0.018)
        lon = z.get("lon",80.27) + np.random.uniform(-0.018,0.018)
        road_lats.append(lat); road_lons.append(lon)
        road_names.append(road["name"]); road_ids.append(road["id"])
        road_types.append(road["type"])

    zone_lats = [z["lat"] for z in CFG["zones"]]
    zone_lons = [z["lon"] for z in CFG["zones"]]
    zone_names= [z["name"] for z in CFG["zones"]]
    zone_ids  = [z["id"]  for z in CFG["zones"]]
    return (road_lats,road_lons,road_names,road_ids,road_types,
            zone_lats,zone_lons,zone_names,zone_ids)

(RLATS,RLONS,RNAMES,RIDS,RTYPES,
 ZLATS,ZLONS,ZNAMES,ZIDS) = build_base_map_data()

def make_map(selected_road_id=None, result=None):
    """Build plotly map. Only called when user explicitly runs a scenario — no auto-rerender."""

    # Road colours based on result or defaults
    if result:
        sc_ti = result["scenario_road"]["traffic_index"]
        bl_ti = result["baseline_road"]["traffic_index"]
        colors = []
        for rid in RIDS:
            if rid == selected_road_id:
                colors.append("#E74C3C")          # selected road: red
            elif rid in [r for r,_ in result.get("alt_roads",[])]:
                colors.append("#E67E22")           # rerouted: orange
            else:
                colors.append("#4A90D9")           # others: blue
        sizes = [14 if rid==selected_road_id else 8 for rid in RIDS]
    else:
        colors = [{"NH":"#2471A3","SH":"#1ABC9C","expressway":"#8E44AD",
                   "arterial":"#E67E22","collector":"#27AE60"}.get(rt,"#4A90D9") for rt in RTYPES]
        sizes  = [10]*len(RIDS)

    # Zone pollution circles
    zone_colors = []
    for zid in ZIDS:
        if result:
            pm = result["scenario_zones"].get(zid,{}).get("pm25",
                 result["baseline_zones"].get(zid,{}).get("pm25",35))
        else:
            pm = PM["zone_baselines"].get(zid,{}).get("pm25",35)
        zone_colors.append(pollution_color(pm))

    fig = go.Figure()

    # Zone circles
    fig.add_trace(go.Scattermapbox(
        lat=ZLATS, lon=ZLONS,
        mode="markers",
        marker=dict(size=30, color=zone_colors, opacity=0.35),
        text=[f"<b>{n}</b><br>Click road markers for traffic" for n in ZNAMES],
        hoverinfo="text",
        name="Zones (pollution)",
        showlegend=True,
    ))

    # Zone labels
    fig.add_trace(go.Scattermapbox(
        lat=ZLATS, lon=ZLONS,
        mode="text",
        text=[n.replace("_"," ").replace("Nagar","Ngr") for n in ZNAMES],
        textfont=dict(size=9, color="#333333"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Road markers
    hover_texts = []
    for i, rid in enumerate(RIDS):
        rt = RTYPES[i]
        if result and rid == selected_road_id:
            ti = result["scenario_road"]["traffic_index"]
            sp = result["scenario_road"]["speed_kmph"]
            hover_texts.append(f"<b>{RNAMES[i]}</b><br>⚠️ SELECTED<br>Traffic: {ti} | Speed: {sp} kmph")
        else:
            hover_texts.append(f"<b>{RNAMES[i]}</b><br>Type: {rt}")

    fig.add_trace(go.Scattermapbox(
        lat=RLATS, lon=RLONS,
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.9),
        text=hover_texts,
        hoverinfo="text",
        name="Roads (traffic)",
        showlegend=True,
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=13.08, lon=80.24),
            zoom=10.5,
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=480,
        legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ccc",borderwidth=1),
        showlegend=True,
    )
    return fig

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "selected_road_a":None,"selected_road_b":None,
    "result_a":None,"result_b":None,
    "map_built_a":False,"map_built_b":False,
    "compare_mode":False,
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏙️ Chennai AI Digital Twin")
st.caption("Select a road from the dropdown → configure scenario → Run → see predictions on the map and panel")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Time & Weather")
    hour  = st.slider("Hour of day",0,23,9)
    dow   = st.selectbox("Day",range(7),
             format_func=lambda x:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],index=1)
    month = st.slider("Month",1,12,6)
    is_we = int(dow>=5); is_pk=int(8<=hour<=10 or 17<=hour<=21); is_mo=int(month in [10,11,12])
    tags  = " ".join(filter(None,["🌧️ Monsoon" if is_mo else "","🏖️ Weekend" if is_we else "","⏰ Peak" if is_pk else ""]))
    if tags: st.caption(tags)
    st.divider()
    st.subheader("🌤️ Weather")
    temp = st.slider("Temperature (°C)",20,46,35)
    hum  = st.slider("Humidity (%)",30,100,72)
    wind = st.slider("Wind (kmph)",0,40,10)
    rain = st.slider("Rainfall (mm)",0.0,50.0,0.0,step=0.5)
    st.divider()
    st.subheader("🎉 Festival")
    is_fest = st.checkbox("Festival day?")
    fest_mult = 1.0
    if is_fest:
        ftype = st.selectbox("Festival",
            ["Diwali / New Year's Eve","Vinayagar Chaturthi",
             "Tamil New Year Eve","Pongal (city quiet)","Republic / Independence Day"])
        fest_mult = [1.65,1.55,1.40,0.50,0.72][["Diwali","Vinayagar","Tamil","Pongal","Republic"]
                    .index(next(k for k in ["Diwali","Vinayagar","Tamil","Pongal","Republic"] if k in ftype))]
    st.divider()
    st.session_state.compare_mode = st.checkbox("🆚 Compare two scenarios",
                                                  value=st.session_state.compare_mode)

CTX = dict(hour=hour,dow=dow,month=month,is_we=is_we,is_pk=is_pk,is_mo=is_mo,
           is_fest=is_fest,fest_mult=fest_mult,temp=temp,hum=hum,wind=wind,rain=rain)

# ── Rerouting logic ───────────────────────────────────────────────────────────
ADJACENCY = {
    "Anna_Salai":      [("Nungambakkam_HR",0.3),("Inner_Ring_Road",0.4),("Kodambakkam_HR",0.2),("NSC_Bose_Road",0.1)],
    "NH16_GST":        [("NH48_Bypass",0.4),("SH104_Vandalur",0.3),("Tambaram_Mudichur",0.2),("Nanganallur_Road",0.1)],
    "NH48_Bypass":     [("NH16_GST",0.4),("CTH_Road",0.3),("Arcot_Road",0.2),("ORR_West",0.1)],
    "NH32_Poonamallee":[("Arcot_Road",0.35),("CTH_Road",0.35),("Poonamallee_HR",0.3)],
    "SH49_OMR":        [("Rajiv_Gandhi_IT",0.4),("Rajiv_Gandhi_Salai",0.3),("NH716_ECR",0.3)],
    "Rajiv_Gandhi_IT": [("SH49_OMR",0.5),("Rajiv_Gandhi_Salai",0.3),("Perungudi_Road",0.2)],
    "Poonamallee_HR":  [("Arcot_Road",0.4),("NH32_Poonamallee",0.3),("Jawaharlal_Nehru_Rd",0.3)],
    "Sardar_Patel_Road":[("Inner_Ring_Road",0.4),("Jawaharlal_Nehru_Sal",0.35),("LB_Road_Adyar",0.25)],
    "Inner_Ring_Road": [("Anna_Salai",0.25),("Sardar_Patel_Road",0.25),("Jawaharlal_Nehru_Sal",0.25),("Rajaji_Salai",0.25)],
    "Thiruvottiyur_HR":[("Madhavaram_HR",0.4),("Tondiarpet_HR",0.3),("Royapuram_HR",0.3)],
    "LB_Road_Adyar":   [("Adyar_Bridge_Road",0.4),("Santhome_HR",0.3),("Dr_Radha_Salai",0.3)],
    "SH_Ennore_Exp":   [("Thiruvottiyur_HR",0.5),("Manali_Road",0.3),("Madhavaram_HR",0.2)],
    "NH716_ECR":       [("SH49_OMR",0.5),("Rajiv_Gandhi_Salai",0.5)],
    "ORR_South":       [("NH16_GST",0.4),("Sardar_Patel_Road",0.3),("Velachery_Tambaram",0.3)],
    "ORR_North":       [("Redhills_Road",0.4),("CTH_Road",0.3),("Madhavaram_HR",0.3)],
}

def get_alt_roads(road_id):
    return ADJACENCY.get(road_id, [])

# ── Run scenario ──────────────────────────────────────────────────────────────
def run_scenario(road_id, scenario, params, ctx):
    road   = ROAD_MAP[road_id]
    rc_f   = params.get("rc_f",1.0)
    pop_f  = params.get("pop_f",1.0)
    nf_f   = params.get("nf_f",1.0)
    eff_pop= pop_f * nf_f

    bl_road = predict_road(road,rc_f=1.0,pop_f=1.0,**ctx)
    sc_road = predict_road(road,rc_f=rc_f,pop_f=eff_pop,**ctx)

    zone_ids = road.get("zones",[])
    bl_zones, sc_zones = {}, {}
    for zid in zone_ids:
        poll_ctx = dict(hour=ctx["hour"],month=ctx["month"],is_mo=ctx["is_mo"],
                        temp=ctx["temp"],hum=ctx["hum"],wind=ctx["wind"],rain=ctx["rain"])
        bl_zones[zid] = predict_poll(zid, bl_road["traffic_index"],rc_f=1.0,  pop_f=1.0,    **poll_ctx)
        sc_zones[zid] = predict_poll(zid, sc_road["traffic_index"],rc_f=rc_f, pop_f=pop_f,  **poll_ctx)

    bl_elec = predict_elec(ctx["hour"],ctx["month"],ctx["is_we"],ctx["is_mo"],ctx["temp"],ctx["hum"])
    sc_elec = predict_elec(ctx["hour"],ctx["month"],ctx["is_we"],ctx["is_mo"],ctx["temp"],ctx["hum"],pop_f=eff_pop) \
              if scenario in ["Population Growth","New Hospital"] else bl_elec

    pz = zone_ids[0] if zone_ids else None
    bl_p = bl_zones.get(pz,{"pm25":35,"pm10":70,"no2":22}) if pz else {"pm25":35,"pm10":70,"no2":22}
    sc_p = sc_zones.get(pz, bl_p) if pz else bl_p

    baseline = {**bl_road,**bl_p,"electricity_mw":bl_elec}
    result   = {**sc_road,**sc_p,"electricity_mw":sc_elec}
    ov       = score(result, baseline)

    alt_roads = get_alt_roads(road_id) if scenario=="Road Closure" else []

    return {
        "road":road,"scenario":scenario,
        "baseline_road":bl_road,"scenario_road":sc_road,
        "baseline_zones":bl_zones,"scenario_zones":sc_zones,
        "baseline_elec":bl_elec,"scenario_elec":sc_elec,
        "overall_score":ov,"zone_ids":zone_ids,"alt_roads":alt_roads,
    }

# ── Output panel ──────────────────────────────────────────────────────────────
def render_output(res, label=""):
    if res is None: return
    road=res["road"]; sc_name=res["scenario"]
    bl_r=res["baseline_road"]; sc_r=res["scenario_road"]
    bl_z=res["baseline_zones"]; sc_z=res["scenario_zones"]
    bl_e=res["baseline_elec"]; sc_e=res["scenario_elec"]
    ov=res["overall_score"]; zids=res["zone_ids"]
    alt_roads=res.get("alt_roads",[])

    if label: st.subheader(label)
    vt,vc = verdict_info(ov)
    st.markdown(
        f'<div style="background:{vc};color:white;padding:8px 14px;border-radius:8px;'
        f'font-weight:bold;font-size:15px;margin-bottom:8px">{vt} &nbsp;|&nbsp; Score: {ov}/100</div>',
        unsafe_allow_html=True)

    st.markdown(f"**Road:** {road['name']} &nbsp;|&nbsp; **Scenario:** {sc_name}  \n"
                f"**Type:** {road['type'].upper()} &nbsp;|&nbsp; "
                f"**Lanes:** {road['lanes']} &nbsp;|&nbsp; **Length:** {road['len']} km")
    st.divider()

    # Traffic
    st.markdown("#### 🚦 Traffic Impact")
    for key,lbl,unit in [("traffic_index","Traffic Index",""),
                          ("speed_kmph","Speed","kmph"),
                          ("travel_time_min","Travel Time","min")]:
        bv=bl_r[key]; sv=sc_r[key]; delta=sv-bv; pct=delta/max(bv,1)*100
        worse = (key!="speed_kmph" and delta>0) or (key=="speed_kmph" and delta<0)
        col="#E74C3C" if worse else "#27AE60"
        arrow="⬆️" if delta>0 else "⬇️"
        u=f" {unit}" if unit else ""
        st.markdown(
            f"**{lbl}**  \n"
            f"<span style='font-size:20px;font-weight:bold'>{sv}{u}</span> "
            f"<span style='color:{col};font-size:13px'>{arrow} {abs(pct):.1f}% &nbsp;({delta:+.1f})</span>  \n"
            f"<span style='color:#888;font-size:11px'>{METRIC_DESC[key]}</span>",
            unsafe_allow_html=True)

    # Rerouting note for road closure
    if sc_name=="Road Closure" and alt_roads:
        st.markdown(
            f"<div style='background:#FEF9E7;border-left:3px solid #F39C12;"
            f"padding:8px;border-radius:4px;font-size:12px;margin-top:6px'>"
            f"🔀 <b>Traffic rerouted to:</b> "
            + ", ".join(ROAD_MAP[rid]["name"] for rid,_ in alt_roads if rid in ROAD_MAP)
            + "</div>", unsafe_allow_html=True)

    st.divider()

    # Pollution
    st.markdown("#### 🌫️ Pollution Impact by Zone")
    for zid in zids[:2]:
        zname=zid.replace("_"," ")
        bl_p=bl_z.get(zid,{}); sc_p=sc_z.get(zid,{})
        src=PM["zone_baselines"].get(zid,{}).get("data_source","interpolated")
        badge="📡 Real CPCB/TNPCB data" if src=="real" else "📐 Interpolated"
        st.markdown(f"*{zname}* &nbsp; <span style='font-size:11px;color:#888'>{badge}</span>",
                    unsafe_allow_html=True)
        for poll,unit in [("pm25","µg/m³"),("pm10","µg/m³"),("no2","µg/m³")]:
            bv=bl_p.get(poll,0); sv=sc_p.get(poll,0)
            delta=sv-bv; pct=delta/max(bv,1)*100
            who_v=WHO.get(poll,999); breach=" ⚠️ Exceeds WHO limit" if sv>who_v else ""
            col="#E74C3C" if delta>0 else "#27AE60"
            arrow="⬆️" if delta>0 else "⬇️"
            st.markdown(
                f"**{poll.upper()}** ({unit})  \n"
                f"<span style='font-size:20px;font-weight:bold'>{sv}</span> "
                f"<span style='color:{col};font-size:13px'>{arrow} {abs(pct):.1f}%{breach}</span>  \n"
                f"<span style='color:#888;font-size:11px'>{METRIC_DESC[poll]}</span>",
                unsafe_allow_html=True)

    # Electricity
    if sc_name in ["Population Growth","New Hospital"]:
        st.divider()
        st.markdown("#### ⚡ Electricity Demand")
        delta_e=sc_e-bl_e; pct_e=delta_e/max(bl_e,1)*100
        col="#E74C3C" if delta_e>0 else "#27AE60"
        arrow="⬆️" if delta_e>0 else "⬇️"
        st.markdown(
            f"**City Demand** (MW)  \n"
            f"<span style='font-size:20px;font-weight:bold'>{sc_e:.0f} MW</span> "
            f"<span style='color:{col};font-size:13px'>{arrow} {abs(pct_e):.1f}% &nbsp;({delta_e:+.0f} MW)</span>  \n"
            f"<span style='color:#888;font-size:11px'>{METRIC_DESC['electricity_mw']}</span>",
            unsafe_allow_html=True)

    st.divider()

    # Causal chain
    st.markdown("#### 🔗 Why this happened")
    pz=zids[0] if zids else None
    tp=(sc_r["traffic_index"]-bl_r["traffic_index"])/max(bl_r["traffic_index"],1)*100
    tc=sc_r["travel_time_min"]-bl_r["travel_time_min"]
    pp=(sc_z.get(pz,{}).get("pm25",0)-bl_z.get(pz,{}).get("pm25",0))/max(bl_z.get(pz,{}).get("pm25",1),1)*100 if pz else 0
    ep=(sc_e-bl_e)/max(bl_e,1)*100

    chains = {
        "Road Closure":[
            f"🚧 {road['name']} closed → traffic forced onto alternate roads",
            f"🚗 Traffic {'increased' if tp>0 else 'decreased'} by {abs(tp):.1f}% on affected routes",
            f"⏱️ Travel time {'increased' if tc>0 else 'decreased'} by {abs(tc):.1f} min",
            f"💨 More idling vehicles → PM2.5 {'up' if pp>0 else 'down'} {abs(pp):.1f}% in zone",
        ],
        "Population Growth":[
            f"👥 More residents → more vehicles on all roads",
            f"🚗 Traffic on {road['name']} {'up' if tp>0 else 'down'} {abs(tp):.1f}%",
            f"💨 More vehicle emissions → PM2.5 {'up' if pp>0 else 'down'} {abs(pp):.1f}%",
            f"⚡ More households + vehicles → electricity {'up' if ep>0 else 'down'} {abs(ep):.1f}%",
        ],
        "New Hospital":[
            f"🏥 Hospital near {road['name']} → construction and visitor traffic",
            f"🚗 Traffic {'up' if tp>0 else 'down'} {abs(tp):.1f}% during construction phase",
            f"🏗️ Construction dust → PM10 increases in zone",
            f"⚡ New facility power demand → electricity {'up' if ep>0 else 'down'} {abs(ep):.1f}%",
        ],
    }
    for step in chains.get(sc_name,[]):
        st.write(step)

# ── Scenario panel ────────────────────────────────────────────────────────────
def render_scenario_panel(label, road_key, result_key):
    st.subheader(f"📋 {label}")

    # Road selector — primary method (no map click needed)
    road_options = {r["id"]: f"{r['name']} ({r['type'].upper()})" for r in CFG["roads"]}
    road_ids_sorted = sorted(road_options.keys(), key=lambda x: road_options[x])

    current = st.session_state.get(road_key)
    idx = road_ids_sorted.index(current) if current in road_ids_sorted else 0

    selected = st.selectbox(
        "Select road",
        road_ids_sorted,
        index=idx,
        format_func=lambda x: road_options[x],
        key=f"dd_{label}",
    )
    st.session_state[road_key] = selected

    road = ROAD_MAP[selected]
    st.caption(f"Lanes: {road['lanes']} | Length: {road['len']} km | "
               f"Free-flow speed: {road['speed']} kmph")

    scenario = st.radio("Scenario",["Road Closure","Population Growth","New Hospital"],
                         key=f"sc_{label}")
    params = {}
    if scenario=="Road Closure":
        sev = st.slider("Severity",1.0,2.5,1.4,step=0.05,
                         help="1.0=fully open  1.4=partial  2.0=full closure  2.5=multiple blocks",
                         key=f"sv_{label}")
        params["rc_f"]=sev
        if sev < 1.2:
            st.caption("ℹ️ Below 1.2 — minimal impact. Try 1.4+ for visible change.")
        elif sev >= 2.0:
            st.caption("⚠️ Full closure — severe rerouting expected.")
    elif scenario=="Population Growth":
        gr = st.slider("Growth %",5,50,20,key=f"gr_{label}")
        params["pop_f"]=1+gr/100
        st.caption(f"Simulates {gr}% more people in this zone.")
    else:
        sz = st.selectbox("Hospital size",["Small","Medium","Large"],key=f"sz_{label}")
        params["nf_f"]={"Small":1.06,"Medium":1.10,"Large":1.16}[sz]
        st.caption(f"{sz} hospital — affects construction traffic and power demand.")

    if st.button(f"▶ Run",type="primary",use_container_width=True,key=f"run_{label}"):
        with st.spinner("Running AI models..."):
            res = run_scenario(selected, scenario, params, CTX)
            st.session_state[result_key] = res
        st.rerun()

    if st.session_state.get(result_key):
        ov = st.session_state[result_key]["overall_score"]
        vt,vc = verdict_info(ov)
        st.markdown(
            f'<div style="background:{vc};color:white;padding:5px 10px;'
            f'border-radius:6px;font-size:13px;margin-top:8px">{vt}</div>',
            unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
if st.session_state.compare_mode:
    # Compare: two columns, each with panel + output
    cola, colb = st.columns(2)
    with cola:
        render_scenario_panel("Scenario A","selected_road_a","result_a")
        st.divider()
        render_output(st.session_state.result_a,"📊 Scenario A — Output")
    with colb:
        render_scenario_panel("Scenario B","selected_road_b","result_b")
        st.divider()
        render_output(st.session_state.result_b,"📊 Scenario B — Output")

    # Map below (shows both selections)
    st.divider()
    st.subheader("🗺️ Map View")
    res_for_map = st.session_state.result_a or st.session_state.result_b
    sel_for_map = st.session_state.selected_road_a or st.session_state.selected_road_b
    fig = make_map(sel_for_map, res_for_map)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # Comparison summary
    if st.session_state.result_a and st.session_state.result_b:
        st.divider()
        st.subheader("🆚 Which scenario is better?")
        sc_a=st.session_state.result_a["overall_score"]
        sc_b=st.session_state.result_b["overall_score"]
        va,_=verdict_info(sc_a); vb,_=verdict_info(sc_b)
        c1,c2=st.columns(2)
        c1.metric("Scenario A score",f"{sc_a}/100",delta=None)
        c2.metric("Scenario B score",f"{sc_b}/100",delta=None)
        if sc_a > sc_b:   st.success("✅ Scenario A scores higher — the better option.")
        elif sc_b > sc_a: st.success("✅ Scenario B scores higher — the better option.")
        else:             st.info("Both scenarios score equally.")

else:
    # Single scenario: panel | map | output
    col_panel, col_map, col_out = st.columns([1.2, 2.8, 2.0])

    with col_panel:
        render_scenario_panel("Scenario","selected_road_a","result_a")

    with col_map:
        st.subheader("🗺️ Chennai Road Network")
        res_now = st.session_state.result_a
        sel_now = st.session_state.selected_road_a
        # Map only rebuilds when a scenario is run — not on sidebar slider changes
        fig = make_map(sel_now, res_now)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        # Legend
        st.caption(
            "🔵 NH/SH/Expressway &nbsp; 🟠 Arterial &nbsp; 🟢 Collector &nbsp;|&nbsp;"
            " Map colours update after running scenario &nbsp;|&nbsp;"
            " Zone circles = pollution level")

    with col_out:
        if res_now:
            render_output(res_now)
        else:
            st.subheader("📊 Output")
            st.info("Select a road and run a scenario to see predictions here.")
            st.markdown("""
**How to use:**
1. Set time, weather and festival in the sidebar
2. Select a road from the dropdown
3. Choose a scenario and adjust parameters
4. Click **▶ Run**
5. See predictions, verdict, and causal chain

**Scenarios:**
- 🚧 **Road Closure** — close a road, see congestion and pollution impact
- 👥 **Population Growth** — add residents, see demand across all domains
- 🏥 **New Hospital** — add facility, see construction and operational impact

**Toggle 🆚 Compare** in sidebar to run two scenarios side by side.
            """)
