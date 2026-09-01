
"""
engine/rerouting.py
--------------------
Calculates which roads absorb diverted traffic when a road is closed.
Uses a simple zone-adjacency graph — when a road closes, adjacent zone
roads get a share of the diverted traffic.
"""
import json, os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f"{_BASE}/data/processed/city_config.json") as f: CFG = json.load(f)

# Road adjacency — which roads can absorb traffic from each road
# Format: road_id -> list of (road_id, absorption_fraction)
# Fraction = how much of closed road traffic goes to this alternative
ADJACENCY = {
    "Anna_Salai":     [("Nungambakkam_HR",0.3),("Inner_Ring_Road",0.4),("Kodambakkam_HR",0.2),("NSC_Bose_Road",0.1)],
    "NH16_GST":       [("NH48_Bypass",0.4),("SH104_Vandalur",0.3),("Tambaram_Mudichur",0.2),("Nanganallur_Road",0.1)],
    "NH48_Bypass":    [("NH16_GST",0.4),("CTH_Road",0.3),("Arcot_Road",0.2),("ORR_West",0.1)],
    "NH32_Poonamallee":[("Arcot_Road",0.35),("CTH_Road",0.35),("Poonamallee_HR",0.3)],
    "SH49_OMR":       [("Rajiv_Gandhi_IT",0.4),("Rajiv_Gandhi_Salai",0.3),("NH716_ECR",0.3)],
    "Rajiv_Gandhi_IT":[("SH49_OMR",0.5),("Rajiv_Gandhi_Salai",0.3),("Perungudi_Road",0.2)],
    "Poonamallee_HR": [("Arcot_Road",0.4),("NH32_Poonamallee",0.3),("Jawaharlal_Nehru_Rd",0.3)],
    "Sardar_Patel_Road":[("Inner_Ring_Road",0.4),("Jawaharlal_Nehru_Sal",0.35),("LB_Road_Adyar",0.25)],
    "Inner_Ring_Road":[("Anna_Salai",0.25),("Sardar_Patel_Road",0.25),("Jawaharlal_Nehru_Sal",0.25),("Rajaji_Salai",0.25)],
    "ORR_North":      [("Redhills_Road",0.4),("CTH_Road",0.3),("Madhavaram_HR",0.3)],
    "ORR_South":      [("NH16_GST",0.4),("Sardar_Patel_Road",0.3),("Velachery_Tambaram",0.3)],
    "ORR_West":       [("NH48_Bypass",0.4),("Arcot_Road",0.3),("NH32_Poonamallee",0.3)],
    "Thiruvottiyur_HR":[("Madhavaram_HR",0.4),("Tondiarpet_HR",0.3),("Royapuram_HR",0.3)],
    "Madhavaram_HR":  [("Thiruvottiyur_HR",0.4),("Redhills_Road",0.3),("Trunk_Road_Perambur",0.3)],
    "LB_Road_Adyar":  [("Adyar_Bridge_Road",0.4),("Santhome_HR",0.3),("Dr_Radha_Salai",0.3)],
    "SH_Ennore_Exp":  [("Thiruvottiyur_HR",0.5),("Manali_Road",0.3),("Madhavaram_HR",0.2)],
    "NH716_ECR":      [("SH49_OMR",0.5),("Rajiv_Gandhi_Salai",0.5)],
}

def get_rerouting(closed_road_id, road_config):
    """
    Returns a dict of road_id -> traffic_factor_increase for roads
    that absorb diverted traffic when closed_road_id is closed.
    Also returns which zones are primarily impacted.
    """
    alternatives = ADJACENCY.get(closed_road_id, [])
    if not alternatives:
        # Generic fallback: all roads in same zones get +15% traffic
        closed_road = next((r for r in road_config if r["id"]==closed_road_id), None)
        if closed_road:
            return {"alternatives": [(r["id"],0.15) for r in road_config
                                      if any(z in r["zones"] for z in closed_road["zones"])
                                      and r["id"]!=closed_road_id][:5],
                    "impacted_zones": closed_road.get("zones",[])}
        return {"alternatives":[], "impacted_zones":[]}

    closed_road = next((r for r in road_config if r["id"]==closed_road_id), {})
    all_impacted_zones = set(closed_road.get("zones",[]))
    for alt_id, _ in alternatives:
        alt_road = next((r for r in road_config if r["id"]==alt_id), {})
        all_impacted_zones.update(alt_road.get("zones",[]))

    return {"alternatives": alternatives,
            "impacted_zones": list(all_impacted_zones)}
