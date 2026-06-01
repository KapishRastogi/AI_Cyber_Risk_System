"""
TawasolPay — AI Cyber Risk Dashboard
=====================================
Self-contained Streamlit app.

Flow:
  1. User uploads data files + enters API key
  2. App checks if FAISS vector database already exists in results folder
  3. If YES → asks user: use existing FAISS? (Yes/No)
     - Yes → catalog not required, loads in <1 second
     - No  → catalog required, rebuilds embeddings (~5 min)
  4. If NO  → catalog required, builds FAISS for first time
  5. Pipeline runs → results saved to "AI Cyber Risk System results/" folder
  6. Outputs: top5 CSV, all_vulns CSV, rag JSON, mdr JSON, mdr_changes JSON

Results folder (created next to app.py):
  AI Cyber Risk System results/
    ├── rag_cache/                    ← FAISS index, persists across runs
    ├── uploaded_data/                ← copy of uploaded CSVs saved here
    ├── top5_risks_with_nist.csv
    ├── all_vulnerabilities_scored.csv
    ├── rag_queries_and_retrieved.json
    ├── mdr_parsed.json
    └── mdr_changes.json
"""

import streamlit as st
import pandas as pd
import json, os, sys, tempfile, traceback, shutil
from pathlib import Path
import importlib

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "AI Cyber Risk System results"
DATA_SAVED  = RESULTS_DIR / "uploaded_data"
RAG_CACHE   = RESULTS_DIR / "rag_cache"
FAISS_INDEX = RAG_CACHE / "nist_faiss.index"
FAISS_FP    = RAG_CACHE / "nist_faiss_fingerprint.json"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="TawasolPay — Cyber Risk",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.stApp{background-color:#0d1117;color:#e6edf3;}
[data-testid="stSidebar"]{background-color:#161b22;border-right:1px solid #30363d;}
.risk-card{background:#161b22;border:1px solid #30363d;border-left:4px solid #f85149;border-radius:6px;padding:20px 24px;margin-bottom:16px;}
.risk-card.p0{border-left-color:#f85149;}.risk-card.p1{border-left-color:#e3b341;}.risk-card.p2{border-left-color:#3fb950;}
.score-badge{display:inline-block;background:#1f6feb;color:#fff;font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;padding:4px 14px;border-radius:4px;margin-right:10px;}
.priority-badge{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;padding:3px 10px;border-radius:3px;text-transform:uppercase;letter-spacing:1px;}
.priority-p0{background:#3d1a1a;color:#f85149;border:1px solid #f85149;}
.priority-p1{background:#3d2e00;color:#e3b341;border:1px solid #e3b341;}
.priority-p2{background:#1a3d1a;color:#3fb950;border:1px solid #3fb950;}
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;}
.metric-box{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#8b949e;}
.metric-box span{display:block;font-size:16px;font-weight:600;color:#e6edf3;margin-top:2px;}
.tag{display:inline-block;font-size:11px;font-family:'IBM Plex Mono',monospace;padding:2px 8px;border-radius:12px;margin:2px;font-weight:600;}
.tag-kev{background:#3d1a1a;color:#f85149;}.tag-ransomware{background:#3d1a1a;color:#ff9800;}
.tag-actor{background:#1a1a3d;color:#79c0ff;}.tag-region{background:#1a3d1a;color:#3fb950;}.tag-normal{background:#21262d;color:#8b949e;}
.section-header{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#8b949e;border-bottom:1px solid #30363d;padding-bottom:6px;margin:16px 0 10px 0;}
.nist-box{background:#0d1117;border:1px solid #1f6feb;border-radius:4px;padding:14px 18px;font-size:14px;line-height:1.7;white-space:pre-wrap;color:#c9d1d9;}
.evidence-box{background:#0d1117;border:1px solid #3fb950;border-radius:4px;padding:14px 18px;font-size:13px;line-height:1.7;white-space:pre-wrap;font-family:'IBM Plex Mono',monospace;color:#7ee787;}
.ioc-box{background:#1a1200;border:1px solid #e3b341;border-radius:4px;padding:12px 16px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#e3b341;line-height:1.8;}
.mdr-pill{background:#1a1200;border:1px solid #e3b341;border-radius:3px;padding:4px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#e3b341;}
.dim-bar-container{margin:4px 0;display:flex;align-items:center;gap:8px;}
.dim-label{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;width:80px;text-align:right;flex-shrink:0;}
.dim-bar-bg{flex:1;background:#21262d;border-radius:2px;height:8px;overflow:hidden;}
.dim-bar-fill{height:100%;border-radius:2px;}
.dim-value{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#e6edf3;width:35px;text-align:right;flex-shrink:0;}
.faiss-banner{background:#0d2137;border:1px solid #1f6feb;border-radius:6px;padding:14px 18px;margin:12px 0;}
.faiss-banner.new{background:#0d2a0d;border-color:#3fb950;}
.faiss-banner.cached{background:#1a1200;border-color:#e3b341;}
h1,h2,h3,h4{color:#e6edf3!important;}p,li{color:#c9d1d9;}
[data-testid="stExpander"]{background:#161b22;border:1px solid #30363d;border-radius:6px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

DIM_COLORS = {"Technical":"#1f6feb","Exposure":"#f85149","Threat":"#ff9800","Business":"#3fb950","Control":"#a371f7"}

def safe(val, default="N/A"):
    try:
        if pd.isna(val): return default
    except: pass
    return default if str(val).strip() in ("nan","","None") else str(val)

def priority_class(p):
    return {"P0":"priority-p0","P1":"priority-p1","P2":"priority-p2"}.get(p,"priority-p2")

def dim_bar(label, value, color):
    try: v = float(value)
    except: v = 0
    return (f"<div class='dim-bar-container'><span class='dim-label'>{label}</span>"
            f"<div class='dim-bar-bg'><div class='dim-bar-fill' style='width:{v}%;background:{color}'></div></div>"
            f"<span class='dim-value'>{v:.0f}</span></div>")

def extract_iocs(explanation):
    if "MDR IOCs:" not in str(explanation): return []
    return [i.strip() for i in str(explanation).split("MDR IOCs:")[1].split("|") if i.strip()]

def faiss_exists():
    return FAISS_INDEX.exists() and FAISS_FP.exists()

def get_faiss_info():
    """Returns dict of fingerprint info if FAISS exists, else None."""
    if not faiss_exists(): return None
    try:
        with open(FAISS_FP) as f:
            return json.load(f)
    except: return None

def save_uploaded_file(name: str, fbytes: bytes):
    """Saves uploaded file bytes to uploaded_data/ folder."""
    DATA_SAVED.mkdir(parents=True, exist_ok=True)
    (DATA_SAVED / name).write_bytes(fbytes)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

defaults = {
    "results":       None,
    "pipeline_ran":  False,
    "use_existing_faiss": None,   # True / False / None (not decided yet)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

has_results = st.session_state.results is not None

# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline(file_map: dict, api_key: str,
                 mdr_bytes=None, use_existing_faiss: bool = False):
    """
    file_map: {filename: bytes} — uploaded files
    use_existing_faiss: if True, loads FAISS from RAG_CACHE instead of rebuilding
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SAVED.mkdir(parents=True, exist_ok=True)
    RAG_CACHE.mkdir(parents=True, exist_ok=True)

    # Save uploaded files permanently to DATA_SAVED/
    for fname, fbytes in file_map.items():
        save_uploaded_file(fname, fbytes)
        st.write(f"  💾 Saved {fname} → {DATA_SAVED / fname}")
    if mdr_bytes:
        save_uploaded_file("synthetic_threat_report.md", mdr_bytes)
        st.write(f"  💾 Saved synthetic_threat_report.md")

    os.environ["GEMINI_API_KEY"] = api_key
    sys.path.insert(0, str(BASE))

    import main as pm
    importlib.reload(pm)

    # Point pipeline at saved data files
    pm.DATA_DIR     = DATA_SAVED
    pm.OUT_DIR      = RESULTS_DIR
    pm.CATALOG_PATH = DATA_SAVED / "NIST_SP-800-53_rev5_catalog_load.csv"
    pm.ASSESS_PATH  = DATA_SAVED / "sp800-53ar5-assessment-procedures.csv"
    pm.MDR_REPORT_PATH = (DATA_SAVED / "synthetic_threat_report.md") if mdr_bytes else None

    # Step 1 — load CSVs
    st.write("⚙️ **Step 1/6** — Loading CSV files...")
    assets, vulns, biz, threats = pm.load_csvs()

    # Step 2 — MDR
    st.write("⚙️ **Step 2/6** — MDR Advisory...")
    ioc_map = {}; mdr_result = None; change_log = []
    if mdr_bytes and (DATA_SAVED / "synthetic_threat_report.md").exists():
        from google import genai as _g
        from mdr_processor import parse_mdr_report, apply_mdr_overrides, build_ioc_map
        cl = _g.Client(api_key=api_key)
        mdr_result  = parse_mdr_report(str(DATA_SAVED/"synthetic_threat_report.md"),
                                        cl, "gemini-2.5-flash", str(RESULTS_DIR))
        threats, change_log = apply_mdr_overrides(threats, mdr_result, str(RESULTS_DIR))
        ioc_map = build_ioc_map(mdr_result)
        st.write(f"   ✅ {len(change_log)} overrides applied | {len(ioc_map)} actors with IOCs")
    else:
        st.write("   ⏭️ No MDR report — skipping")

    # Step 3 — CISA KEV
    st.write("⚙️ **Step 3/6** — Fetching CISA KEV (live)...")
    kev = pm.fetch_cisa_kev()
    st.write(f"   ✅ {len(kev)} KEV entries fetched")

    # Step 4 — Score
    st.write("⚙️ **Step 4/6** — Scoring vulnerabilities...")
    agg  = pm.aggregate_threats(threats)
    df   = pm.build_unified(assets, vulns, biz, agg, kev)
    df   = pm.calculate_scores(df)
    top5, df_all = pm.select_top5(df, ioc_map=ioc_map)
    st.write(f"   ✅ Top 5 scores: {top5['final_score'].tolist()}")

    # Step 5 — RAG
    if use_existing_faiss:
        st.write("⚙️ **Step 5/6** — Loading FAISS vector database from cache...")
        st.markdown(f"""
        <div class='faiss-banner cached'>
        ⚡ <strong>FAISS index loaded from cache</strong> — skipped re-embedding<br>
        <span style='font-size:12px;color:#8b949e'>
        Index: {FAISS_INDEX}<br>
        To rebuild: select "No, rebuild embeddings" next time
        </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.write("⚙️ **Step 5/6** — Building FAISS vector database (first time ~5 min)...")
        st.info("⏳ Downloading bge-large-en-v1.5 (~1.3GB) and embedding 1009 NIST controls... please wait")

    from rag_nist import NISTRag, add_nist_guidance_to_top5, save_rag_queries_json
    rag = NISTRag(
        gemini_api_key=api_key,
        catalog_path=str(pm.CATALOG_PATH) if pm.CATALOG_PATH.exists() else None,
        assess_path=str(pm.ASSESS_PATH)   if pm.ASSESS_PATH.exists() else None,
        cache_dir=str(RAG_CACHE),          # always points to persistent results folder
    )

    if not use_existing_faiss:
        st.markdown(f"""
        <div class='faiss-banner new'>
        ✅ <strong>FAISS vector database created</strong><br>
        <span style='font-size:12px;color:#8b949e'>
        Saved to: {RAG_CACHE}<br>
        Next time you run, select "Yes, use existing FAISS" — catalog upload not required.
        </span>
        </div>""", unsafe_allow_html=True)

    # Step 6 — NIST guidance
    st.write("⚙️ **Step 6/6** — Generating NIST guidance for top-5 risks...")

    # MDR overrides column
    if change_log:
        def _get_ch(row):
            cve = row.get("cve",""); actors = str(row.get("all_actors",""))
            ch = [c for c in change_log if c["cve"]==cve or c["actor"] in actors]
            return " | ".join(f"{c['field']}: {c['old_value']}->{c['new_value']}" for c in ch) if ch else "no changes"
        top5["mdr_overrides"] = top5.apply(_get_ch, axis=1)
    else:
        top5["mdr_overrides"] = "MDR not loaded" if not mdr_bytes else "no changes"

    top5, rag_results = add_nist_guidance_to_top5(top5, rag)

    # Save all outputs to RESULTS_DIR
    top5.to_csv(RESULTS_DIR/"top5_risks_with_nist.csv",          index=False, encoding="utf-8-sig")
    df_all.to_csv(RESULTS_DIR/"all_vulnerabilities_scored.csv",  index=False, encoding="utf-8-sig")
    save_rag_queries_json(top5, rag_results, str(RESULTS_DIR))

    rag_json     = json.load(open(RESULTS_DIR/"rag_queries_and_retrieved.json"))
    mdr_ch_data  = json.load(open(RESULTS_DIR/"mdr_changes.json")) if (RESULTS_DIR/"mdr_changes.json").exists() else {}

    st.success(f"✅ **Pipeline complete!** All results saved to:\n`{RESULTS_DIR}`")

    return {
        "top5":    top5,
        "all_v":   df_all,
        "rag":     rag_json,
        "mdr":     mdr_result,
        "changes": mdr_ch_data,
    }

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛡️ TawasolPay")
    st.markdown("**AI Cyber Risk System**")
    st.markdown("---")

    if has_results:
        page = st.radio("Navigation", [
            "Top 5 Risks", "All Vulnerabilities",
            "Threat Intelligence", "MDR Advisory", "About"
        ], label_visibility="collapsed")
        st.caption(f"📁 Results: `{RESULTS_DIR.name}/`")
    else:
        page = "Setup"

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    # ── API Key ──
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Free at aistudio.google.com — only used this session, never stored"
    )

    # ── FAISS decision ──
    st.markdown("**Vector Database (FAISS)**")
    faiss_info = get_faiss_info()

    if faiss_info:
        st.markdown(f"""
        <div style='background:#1a1200;border:1px solid #e3b341;border-radius:4px;
                    padding:8px 12px;font-size:12px;margin-bottom:8px'>
        ⚡ <strong style='color:#e3b341'>FAISS index exists</strong><br>
        <span style='color:#8b949e'>
        Model: {faiss_info.get('model','')}<br>
        Vectors: {faiss_info.get('n_vectors','')} controls<br>
        Similarity: cosine (IndexFlatIP on normalised vectors)
        </span>
        </div>
        """, unsafe_allow_html=True)

        use_existing = st.radio(
            "Use existing FAISS?",
            ["Yes — use existing (catalog not needed)",
             "No — rebuild embeddings (catalog required)"],
            index=0,
            help="Select Yes to skip re-embedding (~5 min saved). Select No to rebuild from a new catalog."
        )
        use_existing_faiss = use_existing.startswith("Yes")
    else:
        st.info("No FAISS index found — will be built on first run.\nNIST catalog required.")
        use_existing_faiss = False

    # ── File uploaders ──
    st.markdown("**Required data files**")
    f_assets   = st.file_uploader("assets.csv",              type="csv", key="u_assets")
    f_vulns    = st.file_uploader("vulnerabilities.csv",     type="csv", key="u_vulns")
    f_biz      = st.file_uploader("business_services.csv",   type="csv", key="u_biz")
    f_threats  = st.file_uploader("threat_intelligence.csv", type="csv", key="u_threats")

    # Catalog only required when rebuilding
    if not use_existing_faiss:
        st.markdown("**NIST Catalog** *(required — no existing FAISS)*")
    else:
        st.markdown("**NIST Catalog** *(optional — FAISS already built)*")
    f_catalog  = st.file_uploader(
        "NIST_SP-800-53_rev5_catalog_load.csv",
        type="csv", key="u_catalog",
        help="Required only when building FAISS for the first time or rebuilding."
    )

    st.markdown("**Optional files**")
    f_assess   = st.file_uploader("sp800-53ar5-assessment-procedures.csv", type="csv",        key="u_assess")
    f_mdr      = st.file_uploader("synthetic_threat_report.md",            type=["md","txt"], key="u_mdr")

    # ── Readiness check ──
    catalog_needed = not use_existing_faiss
    data_files_ready = all([f_assets, f_vulns, f_biz, f_threats])
    catalog_ready  = (not catalog_needed) or bool(f_catalog)

    missing = []
    if not api_key:    missing.append("API key")
    if not f_assets:   missing.append("assets.csv")
    if not f_vulns:    missing.append("vulnerabilities.csv")
    if not f_biz:      missing.append("business_services.csv")
    if not f_threats:  missing.append("threat_intelligence.csv")
    if catalog_needed and not f_catalog:
        missing.append("NIST catalog (required to build FAISS)")

    st.markdown("---")
    if missing:
        st.warning(f"Missing: {', '.join(missing)}")
    else:
        st.success("✅ Ready to run")

    if not missing and st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        file_map = {
            "assets.csv":              f_assets.read(),
            "vulnerabilities.csv":     f_vulns.read(),
            "business_services.csv":   f_biz.read(),
            "threat_intelligence.csv": f_threats.read(),
        }
        if f_catalog: file_map["NIST_SP-800-53_rev5_catalog_load.csv"] = f_catalog.read()
        if f_assess:  file_map["sp800-53ar5-assessment-procedures.csv"] = f_assess.read()
        mdr_bytes = f_mdr.read() if f_mdr else None

        try:
            results = run_pipeline(file_map, api_key, mdr_bytes, use_existing_faiss)
            st.session_state.results      = results
            st.session_state.pipeline_ran = True
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.code(traceback.format_exc())

    # ── Download buttons ──
    if has_results:
        st.markdown("---")
        R = st.session_state.results
        st.markdown("**Download Results**")

        st.download_button(
            "⬇️ top5_risks.csv",
            R["top5"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "top5_risks_with_nist.csv", "text/csv", use_container_width=True
        )
        st.download_button(
            "⬇️ all_vulnerabilities.csv",
            R["all_v"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "all_vulnerabilities_scored.csv", "text/csv", use_container_width=True
        )
        if R.get("rag"):
            st.download_button(
                "⬇️ rag_detail.json",
                json.dumps(R["rag"], indent=2).encode(),
                "rag_queries_and_retrieved.json", "application/json", use_container_width=True
            )
        st.caption(f"Files also saved to:\n`{RESULTS_DIR}`")


# ─────────────────────────────────────────────
# SETUP PAGE
# ─────────────────────────────────────────────

if not has_results:
    st.markdown("""
    <h1 style='font-family:"IBM Plex Mono",monospace;font-size:26px;margin-bottom:4px'>
    🛡️ TawasolPay AI Cyber Risk System
    </h1>
    <p style='color:#8b949e;font-size:14px;margin-bottom:32px'>
    AI-powered risk prioritisation · NIST SP 800-53 Rev.5 · MDR Intelligence
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📁 Step 1 — Upload files**\n\nUpload the 4 data CSVs using the sidebar. "
                    "NIST catalog only needed if building FAISS for first time.")
    with c2:
        st.markdown("**🔑 Step 2 — Enter API key**\n\nGet a free Gemini key at "
                    "[aistudio.google.com](https://aistudio.google.com). Never stored.")
    with c3:
        st.markdown("**🚀 Step 3 — Run**\n\nFirst run builds the FAISS vector database (~5 min). "
                    "Subsequent runs load from cache in <1 second.")

    st.markdown("---")

    # Show results folder status
    if RESULTS_DIR.exists():
        files = list(RESULTS_DIR.glob("*.csv")) + list(RESULTS_DIR.glob("*.json"))
        if files:
            st.markdown("### 📂 Previous Results Found")
            st.markdown(f"Results folder: `{RESULTS_DIR}`")
            for f in sorted(files):
                sz = f.stat().st_size // 1024
                st.markdown(f"  `{f.name}` — {sz} KB")
            st.markdown("Upload files and click Run to generate new results.")

    st.markdown("---")
    st.markdown("""
    **What the system does:**
    1. Joins 5 CSVs with live CISA KEV data
    2. Applies MDR advisory intelligence overrides (if provided)
    3. Scores vulnerabilities across 5 weighted dimensions
    4. Retrieves NIST SP 800-53 guidance via FAISS + BGE reranker
    5. Generates grounded remediation guidance via Gemini 2.0 Flash
    6. Saves all results to `AI Cyber Risk System results/`
    """)
    st.stop()

# ─────────────────────────────────────────────
# LOAD SESSION DATA
# ─────────────────────────────────────────────

R          = st.session_state.results
top5       = R["top5"]
all_v      = R["all_v"]
rag_data   = R.get("rag", {})
mdr_data   = R.get("mdr")
mdr_changes= R.get("changes", {})

# ─────────────────────────────────────────────
# PAGE: TOP 5 RISKS
# ─────────────────────────────────────────────

if page == "Top 5 Risks":

    st.markdown("""
    <h1 style='font-family:"IBM Plex Mono",monospace;font-size:24px;margin-bottom:4px'>
    Top 5 Risks — TawasolPay
    </h1>
    <p style='color:#8b949e;margin-bottom:24px'>
    Ranked by AI risk score · Evidence-grounded · NIST SP 800-53 aligned
    </p>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Top Score",    f"{top5['final_score'].max():.1f}/100")
    c2.metric("P0 Risks",     int((top5['effective_priority']=='P0').sum()))
    c3.metric("KEV Matched",  int(top5['on_kev'].sum()))
    c4.metric("Ransomware",   int(top5['ransomware_any'].sum()))
    c5.metric("Avg Days Open",f"{top5['days_open'].mean():.0f}")
    st.markdown("---")

    for _, row in top5.iterrows():
        rank     = int(row['rank'])
        priority = safe(row.get('effective_priority','P2'))
        score    = float(row.get('final_score',0))
        p_class  = priority_class(priority)

        st.markdown(f"""
        <div class='risk-card {priority.lower()}'>
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:12px'>
            <span style='font-family:"IBM Plex Mono",monospace;font-size:28px;font-weight:700;color:#8b949e'>#{rank}</span>
            <div>
                <div style='font-size:18px;font-weight:700;color:#e6edf3'>{safe(row.get('asset_name'))}</div>
                <div style='font-size:13px;color:#8b949e;font-family:"IBM Plex Mono",monospace'>
                {safe(row.get('cve'))} · {safe(row.get('vulnerability_name'))}</div>
            </div>
            <div style='margin-left:auto;display:flex;align-items:center;gap:8px'>
                <span class='score-badge'>{score}</span>
                <span class='priority-badge {p_class}'>{priority}</span>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([3,2])
        with cl:
            tags = ""
            if str(row.get('on_kev','')).lower()        in ('true','1'): tags += "<span class='tag tag-kev'>🔴 CISA KEV</span>"
            if str(row.get('ransomware_any','')).lower() in ('true','1'): tags += "<span class='tag tag-ransomware'>☣️ Ransomware</span>"
            if safe(row.get('all_actors'))  != 'N/A':                    tags += f"<span class='tag tag-actor'>👤 {safe(row.get('all_actors'))}</span>"
            if str(row.get('region_specific','')).lower() in ('true','1'):tags += "<span class='tag tag-region'>🌍 Middle East</span>"
            if safe(row.get('compliance_scope')) != 'N/A':               tags += f"<span class='tag tag-normal'>{safe(row.get('compliance_scope'))}</span>"
            st.markdown(tags, unsafe_allow_html=True)

            st.markdown(f"""<div class='metric-row'>
            <div class='metric-box'>CVSS<span>{safe(row.get('cvss'))}</span></div>
            <div class='metric-box'>Severity<span>{safe(row.get('severity'))}</span></div>
            <div class='metric-box'>Days Open<span>{safe(row.get('days_open'))}</span></div>
            <div class='metric-box'>Environment<span>{safe(row.get('environment'))}</span></div>
            <div class='metric-box'>EDR<span>{'✅' if safe(row.get('edr_installed'))=='Yes' else '❌'}</span></div>
            <div class='metric-box'>Maturity<span style='font-size:11px'>{safe(row.get('max_maturity'))}</span></div>
            </div>""", unsafe_allow_html=True)

            expl = safe(row.get('explanation',''))
            base_expl = expl.split("MDR IOCs:")[0].strip().rstrip(";.")
            st.markdown(f"<div class='section-header'>Evidence for Ranking</div>"
                        f"<div style='font-size:13px;color:#c9d1d9;line-height:1.7'>{base_expl}</div>",
                        unsafe_allow_html=True)

            mdr_ov = safe(row.get('mdr_overrides',''))
            if mdr_ov not in ('N/A','no changes','MDR not loaded'):
                st.markdown(f"<div class='section-header'>MDR Intelligence Update</div>"
                            f"<div class='mdr-pill'>🔄 {mdr_ov[:150]}</div>", unsafe_allow_html=True)

            iocs = extract_iocs(expl)
            if iocs:
                st.markdown(f"<div class='section-header'>Active IOCs (MDR Advisory)</div>"
                            f"<div class='ioc-box'>{chr(10).join('▸ '+i for i in iocs)}</div>",
                            unsafe_allow_html=True)

        with cr:
            st.markdown("<div class='section-header'>Score Breakdown</div>", unsafe_allow_html=True)
            bars = (dim_bar("Technical", row.get('dim_technical',0), DIM_COLORS["Technical"]) +
                    dim_bar("Exposure",  row.get('dim_exposure',0),  DIM_COLORS["Exposure"])  +
                    dim_bar("Threat",    row.get('dim_threat',0),    DIM_COLORS["Threat"])    +
                    dim_bar("Business",  row.get('dim_business',0),  DIM_COLORS["Business"])  +
                    dim_bar("Control",   row.get('dim_control',0),   DIM_COLORS["Control"]))
            st.markdown(bars, unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:right;margin-top:8px;font-family:\"IBM Plex Mono\",monospace;"
                        f"font-size:11px;color:#8b949e'>"
                        f"{safe(row.get('business_service'))} · RTO {safe(row.get('rto_hours'))}h</div>",
                        unsafe_allow_html=True)

        primary  = safe(row.get('nist_primary_control',''))
        guidance = safe(row.get('nist_guidance',''))
        evidence = safe(row.get('nist_evidence',''))
        top_ctrl = safe(row.get('nist_top_controls',''))

        with st.expander(f"📋 NIST SP 800-53 Guidance — Primary: {primary}", expanded=False):
            st.markdown(f"""
            <div class='section-header'>Retrieved Controls (FAISS + BGE Reranker)</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:12px;color:#8b949e;margin-bottom:12px'>{top_ctrl}</div>
            <div class='section-header'>Remediation Guidance</div>
            <div class='nist-box'>{guidance}</div>
            <div class='section-header' style='margin-top:16px'>Evidence to Collect (NIST SP 800-53A)</div>
            <div class='evidence-box'>{evidence}</div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ALL VULNERABILITIES
# ─────────────────────────────────────────────

elif page == "All Vulnerabilities":
    st.markdown("## 📊 All Vulnerabilities — Scored & Ranked")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    with c1: kf = st.selectbox("KEV",["All","KEV Only","Non-KEV"])
    with c2: rf = st.selectbox("Ransomware",["All","Yes","No"])
    with c3: ms = st.slider("Min Score",0,100,0)
    with c4: ef = st.selectbox("Environment",["All"]+sorted(all_v['environment'].dropna().unique().tolist()))

    flt = all_v.copy()
    if kf=="KEV Only":  flt = flt[flt['on_kev']==True]
    elif kf=="Non-KEV": flt = flt[flt['on_kev']!=True]
    if rf!="All": flt = flt[flt['ransomware_any'].map(str).str.lower().isin(
        ['true','yes'] if rf=="Yes" else ['false','no'])]
    flt = flt[flt['final_score']>=ms]
    if ef!="All": flt = flt[flt['environment']==ef]

    st.markdown(f"**{len(flt)}** vulnerabilities")
    cols = ['asset_name','cve','vulnerability_name','cvss','severity','environment',
            'final_score','on_kev','ransomware_any','all_actors','max_maturity','days_open']
    st.dataframe(flt[[c for c in cols if c in flt.columns]].sort_values(
        'final_score',ascending=False).reset_index(drop=True),
        use_container_width=True, height=500)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Score Distribution**")
        bins = pd.cut(all_v['final_score'],[0,20,40,60,80,100],labels=['0-20','21-40','41-60','61-80','81-100'])
        st.bar_chart(bins.value_counts().sort_index())
    with c2:
        st.markdown("**Exploit Maturity**")
        if 'max_maturity' in all_v.columns:
            st.bar_chart(all_v['max_maturity'].value_counts())

# ─────────────────────────────────────────────
# PAGE: THREAT INTELLIGENCE
# ─────────────────────────────────────────────

elif page == "Threat Intelligence":
    st.markdown("## 🕵️ RAG Retrieval Detail")
    st.markdown("Exact query sent to bge-large-en-v1.5, FAISS similarity rank, BGE reranker rank, and what was sent to Gemini — per risk.")
    st.markdown("---")

    if rag_data and rag_data.get("risks"):
        for risk in rag_data["risks"]:
            with st.expander(
                f"Risk #{risk.get('risk_rank')} — {risk.get('vulnerability_name','N/A')} "
                f"({risk.get('cve','')})  |  Primary: {risk.get('nist_primary_control','')}",
                expanded=False
            ):
                q = risk.get('embedding_query','').replace(
                    "Represent this sentence for searching relevant passages: ","")
                st.markdown("**Embedding Query sent to bge-large-en-v1.5:**")
                st.code(q, language=None)

                controls = risk.get("retrieved_controls",[])
                if controls:
                    st.markdown(f"**{len(controls)} Retrieved Controls — FAISS rank vs Reranker rank:**")
                    for ctrl in controls:
                        sr = ctrl.get('similarity_rank','')
                        rr = ctrl.get('reranker_rank','')
                        ss = ctrl.get('similarity_score','')
                        rs = ctrl.get('reranker_score','')
                        sent = "🟢 sent to LLM" if ctrl.get('sent_to_llm') else "⬜ not sent"
                        direction = ""
                        try:
                            if int(sr) != int(rr):
                                direction = f" ({'⬆️' if int(rr)<int(sr) else '⬇️'} reranked #{sr}→#{rr})"
                        except: pass
                        st.markdown(
                            f"**{ctrl.get('id')}** — {ctrl.get('name')}  \n"
                            f"FAISS similarity: `{ss}` (rank #{sr}) · "
                            f"Reranker score: `{rs}` (rank #{rr}){direction} · {sent}"
                        )
    else:
        st.info("RAG detail not available — run the pipeline first.")

# ─────────────────────────────────────────────
# PAGE: MDR ADVISORY
# ─────────────────────────────────────────────

elif page == "MDR Advisory":
    st.markdown("## 📡 MDR Advisory Intelligence")
    st.markdown("---")

    if not mdr_data:
        st.info("No MDR report was uploaded for this run.")
        st.stop()

    c1,c2,c3 = st.columns(3)
    c1.metric("Report Date", mdr_data.get("report_date","N/A"))
    c2.metric("Risk Level",  mdr_data.get("risk_level","N/A"))
    c3.metric("Overrides",   mdr_changes.get("metadata",{}).get("total_overrides",0) if mdr_changes else 0)
    st.markdown("---")
    st.markdown("### Active Threat Actors")

    for actor in mdr_data.get("actors",[]):
        name   = actor.get("name",""); ransom = actor.get("ransomware","No")
        conf   = actor.get("confidence",""); maturity = actor.get("maturity","")
        iocs   = actor.get("iocs",[]); cves = actor.get("cves",[])
        with st.expander(
            f"{'☣️' if ransom=='Yes' else '🔍'} {name} — {actor.get('campaign','')}  |  {conf}  |  {maturity}",
            expanded=False
        ):
            ca,cb = st.columns([2,1])
            with ca:
                st.markdown(f"**Summary:** {actor.get('summary','')}")
                st.markdown(f"**Region:** {actor.get('target_region','')} · "
                            f"**Confirmed victims:** {'✅' if actor.get('confirmed_victims') else '❌'}")
                if actor.get('dwell_time_days'):
                    st.markdown(f"**Dwell time:** {actor['dwell_time_days']} days")
                if cves:
                    st.markdown(f"**CVEs:** `{'`, `'.join(cves)}`")
                if actor.get('ransomware_family'):
                    st.markdown(f"**Ransomware:** {actor['ransomware_family']}")
            with cb:
                cc = "#f85149" if conf=="High" else "#e3b341"
                rc = "#f85149" if ransom=="Yes" else "#3fb950"
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:11px;color:#8b949e'>Confidence</div>"
                    f"<div style='font-size:20px;font-weight:700;color:{cc}'>{conf}</div>"
                    f"<div style='font-size:11px;color:#8b949e;margin-top:8px'>Ransomware</div>"
                    f"<div style='font-size:20px;font-weight:700;color:{rc}'>{ransom}</div>"
                    f"</div>", unsafe_allow_html=True)
            if iocs:
                st.markdown("**IOCs to Hunt:**")
                st.markdown(
                    f"<div class='ioc-box'>{chr(10).join('▸ '+i for i in iocs)}</div>",
                    unsafe_allow_html=True)

    if mdr_changes and mdr_changes.get("changes"):
        st.markdown("---")
        st.markdown("### Intelligence Overrides Applied to Scoring")
        st.dataframe(
            pd.DataFrame(mdr_changes["changes"])[
                ["actor","cve","field","old_value","new_value","reason","score_impact"]
            ], use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────

elif page == "About":
    st.markdown("## ℹ️ System Architecture")
    st.markdown("---")

    # Show results folder contents
    if RESULTS_DIR.exists():
        st.markdown(f"### 📂 Results Folder: `{RESULTS_DIR}`")
        all_files = sorted(RESULTS_DIR.rglob("*"))
        for f in all_files:
            if f.is_file():
                rel = f.relative_to(RESULTS_DIR)
                sz  = f.stat().st_size
                unit = "KB" if sz > 1024 else "B"
                sz_disp = f"{sz//1024} KB" if sz > 1024 else f"{sz} B"
                st.markdown(f"  `{rel}` — {sz_disp}")
        st.markdown("---")

    st.markdown("""
### Pipeline
```
Upload CSVs + API key
        ↓
Check FAISS cache → ask user: use existing? (Yes/No)
        ↓
MDR Parser (Gemini)        extract actors/IOCs, apply overrides
        ↓
CISA KEV Live Fetch        cross-reference confirmed exploited CVEs
        ↓
Scoring (5 dimensions)     Technical · Exposure · Threat · Business · Control
        ↓
Top 5 Selection
        ↓
FAISS Retrieval            bge-large-en-v1.5 · IndexFlatIP cosine · 1,009 controls
        ↓
BGE Reranker               bge-reranker-base cross-encoder · re-ranks 15 candidates
        ↓
Gemini 2.0 Flash           explains code-selected primary control
        ↓
Save to AI Cyber Risk System results/
```

### Data Split

| | What | Why |
|---|---|---|
| **Embedded** | NIST SP 800-53 Rev.5 (1,009 active controls) | Relevance needs semantic similarity — cannot filter |
| **Structured** | All 5 CSVs, CISA KEV, MDR report | Known schemas — joined, filtered, aggregated directly |

### Scoring Weights

| Dimension | Weight | Measures |
|---|---|---|
| Technical | 25% | CVSS + severity + exploit availability |
| Exposure  | 25% | Internet exposure + environment + asset type |
| Threat    | 30% | Maturity + KEV + ransomware + confidence + region |
| Business  | 15% | Criticality + revenue + compliance scope + RTO |
| Control   | 5%  | Missing EDR, missing auth, patch age |

### Known Limitations
1. **SI-2 retrieval gap** — 512-token limit on bge-large-en-v1.5 may truncate SI-2's key vocabulary
2. **KEV miss** — CVEs not yet in CISA KEV won't be flagged as actively exploited
3. **MDR parsing** — section-by-section parser uses `### N. ActorName` as demarcation
    """)

    st.markdown("---")
    st.markdown("### Run Locally")
    st.code("""
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
streamlit run app.py
    """, language="bash")
