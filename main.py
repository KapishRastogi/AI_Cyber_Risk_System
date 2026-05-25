"""
TawasolPay — Complete AI Cyber Risk System
==========================================
Full pipeline in one command:

  Step 1–6 : Load CSVs → join → score → top 5     (~10 sec)
  Step 7   : Load mxbai-embed-large-v1 locally     (~30 sec first run)
  Step 8   : Embed 1189 NIST chunks locally         (~1-3 min CPU)
  Step 9   : RAG for each top-5 risk                (~30 sec, 5 Gemini calls)
  Step 10  : Print complete readable report         (instant)
  Step 11  : Save CSVs                              (instant)

Requirements:
    pip install sentence-transformers torch requests pandas numpy scipy

Environment variables:
    GEMINI_API_KEY  — required for generation step only
    (mxbai embeddings run locally, no key needed)

Run:
    export GEMINI_API_KEY=your_key_here
    python3 main.py
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# PATHS — update to your local folders
# ─────────────────────────────────────────────

DATA_DIR     = Path("/Users/kapish.rastogi/Downloads/ai_associate_assignment/Dataset")
OUT_DIR      = Path("/Users/kapish.rastogi/Downloads/ai_associate_assignment/Result_final")
CATALOG_PATH = Path("/Users/kapish.rastogi/Downloads/NIST_SP-800-53_rev5_catalog_load.csv")
ASSESS_PATH  = Path("/Users/kapish.rastogi/Downloads/sp800-53ar5-assessment-procedures.csv")

# ── MDR Report (optional) ──────────────────────────────────────────────────
# Set to path of MDR advisory to enable threat intel override pipeline.
# Set to None to skip — system uses threat_intelligence.csv data only.
MDR_REPORT_PATH = DATA_DIR / "synthetic_threat_report.md"   # or None to skip

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# ─────────────────────────────────────────────
# IMPORT RAG MODULE
# ─────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from rag_nist import NISTRag, add_nist_guidance_to_top5, print_nist_section, save_rag_queries_json
from mdr_processor import parse_mdr_report, apply_mdr_overrides, build_ioc_map

# ─────────────────────────────────────────────
# SCORING CONFIG (v3 calibrated)
# ─────────────────────────────────────────────

DIM_WEIGHTS = {
    "technical": 0.25, "exposure": 0.25,
    "threat": 0.30, "business": 0.15, "control": 0.05
}
SEVERITY_SCORE  = {"Critical":100,"High":75,"Medium":50,"Low":25}
MATURITY_SCORE  = {"Active Exploitation":100,"Weaponized":85,"Commodity Exploit":65,
                   "Proof of Concept":40,"Social Engineering":25,"Not Applicable":10}
# Active Exploitation = 100: confirmed real attacks happening now
# Weaponized          =  85: exploit ready but not confirmed in active use
INTERNET_SCORE  = 45
PRODUCTION_SCORE= 25
HIGH_RISK_ASSET_SCORE = {
    "VPN Gateway":15,"Load Balancer":15,"Firewall":15,
    "API Server":12,"Web Server":12,"Web Application":12,
    "Mail Server":10,"Application Server":8,"Build Server":8,
    "Kubernetes Cluster":8,"Database":6,"Object Storage":6,
    "Endpoint":4,"Unknown":5,
}
KEV_BONUS=25; RANSOMWARE_BONUS=15; MULTI_ACTOR_BONUS=5; MATURITY_SCALE=0.35
REGION_BONUS={"Middle East":12,"Global":4}
CONFIDENCE_MULT={"High":1.5,"Medium":1.3,"Low":1.1,None:1.0}
CRITICALITY_SCORE={"Critical":100,"High":75,"Medium":50,"Low":25}
REVENUE_SCORE    ={"Critical":100,"High":75,"Medium":50,"Low":25}
COMPLIANCE_BONUSES={"PCI":20,"GDPR":15,"PDPL":12,"ISO":8,"SOC":8}
RTO_SCORE={1:100,2:85,4:65,8:45,12:35,24:20,48:10}
NO_EDR_SCORE=35; NO_AUTH_SCORE=30; NO_PATCH_SCORE=25
STALE_SCORE=15;  NO_OWNER_SCORE=10
AGE_PENALTY={180:20,90:12,30:5}

# ─────────────────────────────────────────────
# PIPELINE STEPS 1–6 : SCORING
# ─────────────────────────────────────────────

def load_csvs():
    print("\n[1/11] Loading CSV files...")
    assets  = pd.read_csv(DATA_DIR/"assets.csv")
    vulns   = pd.read_csv(DATA_DIR/"vulnerabilities.csv")
    biz     = pd.read_csv(DATA_DIR/"business_services.csv")
    threats = pd.read_csv(DATA_DIR/"threat_intelligence.csv")
    print(f"  assets:{len(assets)}  vulns:{len(vulns)}  biz:{len(biz)}  threats:{len(threats)}")
    return assets, vulns, biz, threats


def fetch_cisa_kev():
    print("\n[2/11] Fetching CISA KEV (live)...")
    try:
        resp = requests.get(CISA_KEV_URL, timeout=15)
        resp.raise_for_status()
        kev = {}
        for e in resp.json().get("vulnerabilities", []):
            kev[e["cveID"]] = {
                "on_kev": True,
                "kev_ransomware": e.get("knownRansomwareCampaignUse","").lower()=="known",
            }
        print(f"  {len(kev)} KEV entries fetched")
        return kev
    except Exception as ex:
        print(f"  WARNING: {ex} — scoring continues without KEV")
        return {}


def aggregate_threats(threats):
    print("\n[3/11] Aggregating threat campaigns per CVE...")
    t  = threats.rename(columns={"matched_cve_or_control":"cve"})
    mr = {"Weaponized":6,"Active Exploitation":5,"Commodity Exploit":4,
          "Proof of Concept":3,"Social Engineering":2,"Not Applicable":1}
    cr = {"High":3,"Medium":2,"Low":1}

    def agg(g):
        actors = g["threat_actor"].dropna().unique().tolist()
        camps  = g["campaign_name"].dropna().unique().tolist()
        mats   = g["exploit_maturity"].dropna().tolist()
        confs  = g["confidence"].dropna().tolist()
        return pd.Series({
            "all_actors":      ", ".join(actors) if actors else None,
            "all_campaigns":   ", ".join(camps)  if camps  else None,
            "ransomware_any":  (g["ransomware_association"].fillna("No")=="Yes").any(),
            "max_confidence":  max(confs,key=lambda x:cr.get(x,0),default=None),
            "max_maturity":    max(mats, key=lambda x:mr.get(x,0),default=None),
            "region_specific": (g["target_region"].fillna("")=="Middle East").any(),
            "campaign_count":  len(g["campaign_name"].dropna().unique()),
            "last_active":     g["active_last_seen"].dropna().max() if "active_last_seen" in g.columns else None,
        })

    agg_df = t.groupby("cve").apply(agg).reset_index()
    print(f"  {len(agg_df)} unique CVEs  |  {(agg_df['campaign_count']>1).sum()} with 2+ campaigns")
    return agg_df


def build_unified(assets, vulns, biz, agg_threats, kev_dict):
    print("\n[4/11] Building unified dataframe (LEFT JOINs)...")
    df = vulns.merge(assets, on="asset_id", how="left", suffixes=("","_a"))
    df = df.merge(biz, on="business_service", how="left", suffixes=("","_b"))
    df = df.merge(agg_threats, on="cve", how="left")
    df["has_threat_match"] = df["all_actors"].notna()
    if kev_dict:
        kdf = pd.DataFrame([{"cve":k,**v} for k,v in kev_dict.items()])
        df  = df.merge(kdf, on="cve", how="left")
        df["on_kev"]         = df["on_kev"].fillna(False)
        df["kev_ransomware"] = df["kev_ransomware"].fillna(False)
    else:
        df["on_kev"] = False; df["kev_ransomware"] = False
    df = df[df["status"]=="Open"].copy()
    print(f"  {len(df)} open vulns  "
          f"|  {df['has_threat_match'].sum()} with campaign  "
          f"|  {df['on_kev'].sum()} on KEV")
    return df


def calculate_scores(df):
    print("\n[5/11] Calculating risk scores (v3 calibrated)...")
    df = df.copy()

    # Technical
    df["dim_technical"] = (
        (df["cvss"].fillna(0)/10)*60
        + df["severity"].map(SEVERITY_SCORE).fillna(0)*0.20
        + np.where(df["exploit_available"].fillna("No")=="Yes",20,0)
    ).clip(0,100).round(2)

    # Exposure (C1)
    internet = np.where(
        (df["internet_exposed"].fillna("No")=="Yes")|(df["asset_exposure"].fillna("Internal")=="Internet"),
        INTERNET_SCORE, 0)
    env   = df["environment"].map({"Production":PRODUCTION_SCORE,"Staging":10,"Development":0}).fillna(0)
    atype = df["asset_type"].map(HIGH_RISK_ASSET_SCORE).fillna(5)
    df["dim_exposure"] = (internet+env+atype).clip(0,100).round(2)

    # Threat (C2)
    ts  = df["max_maturity"].map(MATURITY_SCORE).fillna(0)*MATURITY_SCALE
    ts += np.where(df["on_kev"],KEV_BONUS,0)
    ts += np.where(df["ransomware_any"].fillna(False)|df["kev_ransomware"].fillna(False),RANSOMWARE_BONUS,0)
    ts += df["region_specific"].fillna(False).map({True:REGION_BONUS["Middle East"],False:REGION_BONUS["Global"]}).fillna(0)
    ts += np.where(df["campaign_count"].fillna(0)>1,MULTI_ACTOR_BONUS,0)
    cm  = df["max_confidence"].map(CONFIDENCE_MULT).fillna(CONFIDENCE_MULT[None])
    df["dim_threat"] = (ts*cm).clip(0,100).round(2)

    # Business
    crit = df["criticality"].map(CRITICALITY_SCORE).fillna(0)*0.30
    rev  = df["revenue_impact"].map(REVENUE_SCORE).fillna(0)*0.30
    sc   = df["compliance_scope"].fillna("")
    comp = sum(np.where(sc.str.contains(k,case=False,na=False),v,0) for k,v in COMPLIANCE_BONUSES.items())
    comp = pd.Series(comp,index=df.index).clip(0,25)
    def rto_lkp(v):
        if pd.isna(v): return 5
        vi=int(v)
        for t,p in sorted(RTO_SCORE.items()):
            if vi<=t: return p*0.15
        return RTO_SCORE[48]*0.15
    rto = df["rto_hours"].apply(rto_lkp)
    df["dim_business"] = (crit+rev+comp+rto).clip(0,100).round(2)

    # Control gap (C3)
    cs  = np.where(df["edr_installed"].fillna("Yes")=="No",  NO_EDR_SCORE,  0)
    cs += np.where(df["auth_required"].fillna("Yes")=="No",  NO_AUTH_SCORE, 0)
    cs += np.where(df["patch_available"].fillna("Yes")=="No",NO_PATCH_SCORE,0)
    cs += np.where(df["last_seen_days"].fillna(0)>30,        STALE_SCORE,   0)
    cs += np.where(df["owner_team"].isna(),                  NO_OWNER_SCORE,0)
    days = df["days_open"].fillna(0)
    cs  += np.where(days>180,AGE_PENALTY[180],np.where(days>90,AGE_PENALTY[90],np.where(days>30,AGE_PENALTY[30],0)))
    df["dim_control"] = pd.Series(cs,index=df.index).clip(0,100).round(2)

    # Final (C4 — no normalisation)
    df["raw_score"] = (
        df["dim_technical"]*DIM_WEIGHTS["technical"]
        +df["dim_exposure"] *DIM_WEIGHTS["exposure"]
        +df["dim_threat"]   *DIM_WEIGHTS["threat"]
        +df["dim_business"] *DIM_WEIGHTS["business"]
        +df["dim_control"]  *DIM_WEIGHTS["control"]
    ).round(2)
    df["final_score"] = df["raw_score"].clip(0,100).round(1)

    s = df["final_score"]
    print(f"  Range: {s.min():.1f}–{s.max():.1f}  "
          f"|  80+:{(s>=80).sum()}  60-79:{((s>=60)&(s<80)).sum()}  "
          f"40-59:{((s>=40)&(s<60)).sum()}  <40:{(s<40).sum()}")
    return df


def build_explanation(row, ioc_map=None):
    p=[]
    if row.get("internet_exposed")=="Yes" or row.get("asset_exposure")=="Internet":
        p.append("internet-exposed asset")
    if row.get("on_kev"): p.append("CISA KEV — confirmed exploited in wild")
    mat=row.get("max_maturity")
    if pd.notna(mat) and mat: p.append(f"exploit maturity: {mat}")
    elif row.get("exploit_available")=="Yes": p.append("working exploit available")
    actors=row.get("all_actors")
    if pd.notna(actors):
        t=f"targeted by {actors}"
        c=row.get("all_campaigns")
        if pd.notna(c): t+=f" ({c})"
        if row.get("ransomware_any"): t+=" — ransomware"
        if row.get("region_specific"): t+=" / Middle East"
        p.append(t)
    crit=row.get("criticality",""); rev=row.get("revenue_impact","")
    if crit in("Critical","High"):
        b=f"{crit.lower()} criticality"
        if rev in("Critical","High"): b+=f", {rev.lower()} revenue"
        p.append(b)
    scope=str(row.get("compliance_scope",""))
    flags=[f for f in["PCI DSS","GDPR","UAE PDPL","ISO 27001"] if any(k in scope for k in[f.split()[0]])]
    if flags: p.append(f"compliance: {', '.join(flags)}")
    gaps=[]
    if row.get("edr_installed")=="No":   gaps.append("no EDR")
    if row.get("patch_available")=="No": gaps.append("no patch/EOL")
    if row.get("auth_required")=="No":   gaps.append("no auth")
    if gaps: p.append("gaps: "+" | ".join(gaps))
    d=row.get("days_open",0)
    if d>=180: p.append(f"unpatched {int(d)} days")
    elif d>=90: p.append(f"open {int(d)} days")
    rto=row.get("rto_hours")
    if pd.notna(rto) and rto<=2: p.append(f"RTO {int(rto)}hr")
    base = "; ".join(p)+"." if p else "multiple elevated risk factors."

    # Append IOCs from MDR report if available for this actor
    if ioc_map and pd.notna(actors):
        matched_iocs = []
        for actor_name in str(actors).split(", "):
            if actor_name.strip() in ioc_map:
                matched_iocs.extend(ioc_map[actor_name.strip()])
        if matched_iocs:
            ioc_lines = " | ".join(matched_iocs)
            base += f" MDR IOCs: {ioc_lines}"
    return base


def get_priority(row):
    s=row.get("final_score",0); internet=row.get("internet_exposed")=="Yes"
    kev=row.get("on_kev",False); ransom=row.get("ransomware_any",False)
    no_edr=row.get("edr_installed")=="No"; crit=row.get("criticality") in("Critical","High")
    matched=row.get("has_threat_match",False)
    if s>=80:                       return "P0"
    if kev and internet:            return "P0"
    if ransom and internet and no_edr: return "P0"
    if matched and crit and no_edr: return "P0"
    if s>=60:                       return "P1"
    if internet and crit:           return "P1"
    return "P2"


def select_top5(df, ioc_map=None):
    global _ioc_map_ref
    _ioc_map_ref = [ioc_map or {}]  # thread-safe pass to apply lambda
    print("\n[6/11] Selecting top 5 risks...")
    ds = df.sort_values(["final_score","on_kev","days_open","rto_hours"],
                        ascending=[False,False,False,True]).reset_index(drop=True)
    t5 = ds.drop_duplicates(subset=["asset_id"],keep="first").head(5).copy()
    t5["rank"]               = range(1,6)
    t5["explanation"]        = t5.apply(lambda r: build_explanation(r, ioc_map=_ioc_map_ref[0]),axis=1)
    t5["effective_priority"] = t5.apply(get_priority,axis=1)
    print(f"  Top 5 across {t5['asset_id'].nunique()} unique assets")
    return t5, ds


# ─────────────────────────────────────────────
# PRINT FULL REPORT (Thing 3)
# ─────────────────────────────────────────────

def print_report(top5):
    print("\n"+"="*70)
    print("  TAWASOL PAY — AI CYBER RISK REPORT")
    print(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Scope      : 60 assets · 114 vulnerabilities · 40 threat campaigns")
    print(f"  Scoring    : weighted additive, 5 dimensions, absolute 0–100")
    print(f"  Embeddings : mxbai-embed-large-v1 (local, HuggingFace)")
    print(f"  Guidance   : NIST SP 800-53 Rev. 5 via RAG + Gemini 1.5 Flash")
    print("="*70)

    for _,row in top5.iterrows():
        rank=int(row["rank"]); score=row["final_score"]; pri=row["effective_priority"]
        print(f"\n{'─'*70}")
        print(f"  RISK #{rank}  |  Score: {score}/100  |  Priority: {pri}")
        print(f"{'─'*70}")

        print(f"\n  ASSET")
        print(f"    Name    : {row.get('asset_name','N/A')}")
        print(f"    Type    : {row.get('asset_type','N/A')}")
        print(f"    Env     : {row.get('environment','N/A')}  |  Owner: {row.get('owner_team','UNASSIGNED')}")
        print(f"    Internet: {row.get('internet_exposed','N/A')}  |  EDR: {row.get('edr_installed','N/A')}")
        ls=row.get("last_seen_days","N/A")
        stale=" ⚠ STALE" if pd.notna(ls) and ls>30 else ""
        print(f"    Last seen: {ls} days ago{stale}")

        print(f"\n  VULNERABILITY")
        print(f"    Name    : {row.get('vulnerability_name','N/A')}")
        print(f"    CVE     : {row.get('cve','N/A')}")
        print(f"    CVSS    : {row.get('cvss','N/A')}  |  Severity: {row.get('severity','N/A')}")
        print(f"    Days open: {row.get('days_open','N/A')}  |  Exploit: {row.get('exploit_available','N/A')}  |  Patch: {row.get('patch_available','N/A')}")
        print(f"    CISA KEV: {'YES — confirmed exploited in wild' if row.get('on_kev') else 'No'}")

        print(f"\n  THREAT INTELLIGENCE")
        if row.get("has_threat_match"):
            print(f"    Actors  : {row.get('all_actors','N/A')}")
            print(f"    Campaign: {row.get('all_campaigns','N/A')}")
            print(f"    Ransom  : {'Yes' if row.get('ransomware_any') else 'No'}  |  Maturity: {row.get('max_maturity','N/A')}")
            print(f"    Conf    : {row.get('max_confidence','N/A')}  |  Region: {'Middle East' if row.get('region_specific') else 'Global'}")
            print(f"    Last active: {row.get('last_active','N/A')}")
        else:
            print(f"    No active campaign — technical scoring only")

        print(f"\n  BUSINESS SERVICE")
        print(f"    Service    : {row.get('business_service','N/A')}")
        print(f"    Revenue    : {row.get('revenue_impact','N/A')}  |  RTO: {row.get('rto_hours','N/A')} hours")
        print(f"    Compliance : {row.get('compliance_scope','None')}")
        print(f"    Cust-facing: {row.get('customer_facing','No')}")

        t=row.get('dim_technical',0); ex=row.get('dim_exposure',0)
        th=row.get('dim_threat',0);   bu=row.get('dim_business',0); co=row.get('dim_control',0)
        print(f"\n  SCORE BREAKDOWN")
        print(f"    Technical  ×0.25 : {t:.1f} → {t*0.25:.1f} pts")
        print(f"    Exposure   ×0.25 : {ex:.1f} → {ex*0.25:.1f} pts")
        print(f"    Threat     ×0.30 : {th:.1f} → {th*0.30:.1f} pts")
        print(f"    Business   ×0.15 : {bu:.1f} → {bu*0.15:.1f} pts")
        print(f"    Control gap×0.05 : {co:.1f} → {co*0.05:.1f} pts")
        print(f"    Final            : {score}/100")

        print(f"\n  WHY THIS RANKS HERE")
        print(f"    {row.get('explanation','N/A')}")

        # NIST RAG guidance — Thing 2
        print_nist_section(row)

    print(f"\n{'='*70}")
    print(f"  End of report — {len(top5)} risks require immediate action")
    print(f"  Guidance source: NIST SP 800-53 Rev. 5 (RAG · mxbai + Gemini)")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_outputs(top5, df_all):
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    cols = [
        "vuln_id","asset_id","asset_name","cve","vulnerability_name","cvss","severity",
        "internet_exposed","edr_installed","criticality","environment","days_open",
        "on_kev","has_threat_match","all_actors","all_campaigns","ransomware_any",
        "max_confidence","max_maturity","region_specific","campaign_count",
        "business_service","revenue_impact","compliance_scope","rto_hours",
        "dim_technical","dim_exposure","dim_threat","dim_business","dim_control",
        "raw_score","final_score"
    ]
    avail = [c for c in cols if c in df_all.columns]
    df_all[avail].sort_values("final_score",ascending=False).to_csv(
        OUT_DIR/"all_vulnerabilities_scored.csv", index=False, encoding="utf-8-sig")

    t5cols = avail+["rank","effective_priority","explanation","mdr_overrides",
                    "nist_primary_control","nist_top_controls","nist_guidance",
                    "nist_prompt","nist_evidence"]
    t5av = [c for c in t5cols if c in top5.columns]
    top5[t5av].to_csv(OUT_DIR/"top5_risks_with_nist.csv", index=False, encoding="utf-8-sig")

    print(f"\n[11/11] Saved to {OUT_DIR}")
    print(f"  all_vulnerabilities_scored.csv — all 114 rows")
    print(f"  top5_risks_with_nist.csv       — top 5 + NIST guidance")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\nTawasolPay AI Cyber Risk System")
    print("="*70)
    print("Embeddings : BAAI/bge-large-en-v1.5 (local, HuggingFace)")
    print("Reranker   : BAAI/bge-reranker-base (cross-encoder)")
    print("Generation : Gemini 2.0 Flash (Google API)")
    print("="*70)

    api_key = ""
    if not api_key:
        print("\nERROR: GEMINI_API_KEY not set.")
        print("  Mac/Linux : export GEMINI_API_KEY=your_key")
        print("  Windows   : set GEMINI_API_KEY=your_key")
        sys.exit(1)

    # ── Step 1: Load CSVs ──────────────────────────────────────────────────
    assets, vulns, biz, threats = load_csvs()

    # ── Step 2: MDR Report (optional) ─────────────────────────────────────
    # If MDR_REPORT_PATH is set, parse the advisory and override stale
    # threat intel fields (confidence, maturity, ransomware) where MDR
    # contains stronger/newer intelligence than the CSV.
    ioc_map    = {}   # actor -> [ioc strings], populated if MDR loaded
    mdr_data   = None
    change_log = []

    if MDR_REPORT_PATH and Path(MDR_REPORT_PATH).exists():
        print(f"\n[MDR] MDR report found: {MDR_REPORT_PATH}")

        # Create a temporary Gemini client for MDR parsing
        from google import genai as _genai
        _client = _genai.Client(api_key=api_key)

        # Parse the report into structured JSON
        mdr_data = parse_mdr_report(
            report_path=str(MDR_REPORT_PATH),
            gemini_client=_client,
            gemini_model="gemini-2.0-flash",
            out_dir=str(OUT_DIR),
        )

        # Apply overrides to threats dataframe — returns updated df + log
        threats, change_log = apply_mdr_overrides(
            threats_df=threats,
            mdr_data=mdr_data,
            out_dir=str(OUT_DIR),
        )

        # Build IOC map for explanation enrichment
        ioc_map = build_ioc_map(mdr_data)
        print(f"[MDR] IOC map built for {len(ioc_map)} actors")

    else:
        if MDR_REPORT_PATH:
            print(f"\n[MDR] Report not found at {MDR_REPORT_PATH} — skipping")
        else:
            print("\n[MDR] No MDR report path set — using CSV data only")

    # ── Steps 3–7: scoring ────────────────────────────────────────────────
    kev         = fetch_cisa_kev()
    agg_threats = aggregate_threats(threats)   # uses MDR-updated threats if applied
    df          = build_unified(assets, vulns, biz, agg_threats, kev)
    df          = calculate_scores(df)
    top5, df_all= select_top5(df, ioc_map=ioc_map)

    # Steps 7–9: RAG
    print("\n[7/11] Loading mxbai-embed-large-v1 from HuggingFace...")
    print("[8/11] Embedding 1189 NIST controls locally...")
    rag = NISTRag(
        gemini_api_key=api_key,
        catalog_path=str(CATALOG_PATH),
        assess_path=str(ASSESS_PATH),
        cache_dir=str(OUT_DIR / "rag_cache"),   # saves chunks JSON + embedding NPZ
    )

    # Tag each top-5 row with MDR override details if any overrides applied
    if change_log:
        def get_mdr_changes(row):
            cve    = row.get("cve","")
            actors = str(row.get("all_actors",""))
            changes = [
                c for c in change_log
                if c["cve"] == cve or c["actor"] in actors
            ]
            if not changes:
                return "no changes"
            return " | ".join(
                f"{c['field']}: {c['old_value']} -> {c['new_value']} ({c['reason']})"
                for c in changes
            )
        top5["mdr_overrides"] = top5.apply(get_mdr_changes, axis=1)
    else:
        top5["mdr_overrides"] = "MDR not loaded" if not mdr_data else "no changes"

    print("\n[9/11] Running NIST RAG for each top-5 risk...")
    top5, rag_results = add_nist_guidance_to_top5(top5, rag)

    # Step 10: save RAG queries JSON
    save_rag_queries_json(top5, rag_results, str(OUT_DIR))

    # Step 11: print
    print_report(top5)

    # Step 11: save
    save_outputs(top5, df_all)

    return top5, df_all


if __name__ == "__main__":
    top5, df_all = main()
