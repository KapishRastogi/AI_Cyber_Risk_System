"""
mdr_processor.py — MDR Report Ingestion and Threat Intelligence Override

Parses the MDR advisory (markdown), extracts structured threat actor data,
compares against threat_intelligence.csv, applies overrides where the report
contains new or updated intelligence, and stores a detailed change log.

ON/OFF:
    Controlled by MDR_REPORT_PATH in main.py.
    If None — this module is never called, system uses CSV data only.
    If set  — full MDR pipeline runs before scoring.

Three public functions:
    parse_mdr_report(path, gemini_client)
        Extracts structured data from the markdown using Gemini.
        Saves: {OUT_DIR}/mdr_parsed.json

    apply_mdr_overrides(threats_df, mdr_data)
        Compares MDR vs CSV for each actor.
        Returns updated threats_df + change_log list.
        Saves: {OUT_DIR}/mdr_changes.json

    build_ioc_map(mdr_data)
        Returns dict: actor_name -> [ioc strings]
        Used by explanation builder to add IOC section per risk.
"""

import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
from google import genai
from google.genai import types


# ─────────────────────────────────────────────
# MATURITY AND CONFIDENCE RANK MAPS
# (must match values used in main.py scoring)
# ─────────────────────────────────────────────

MATURITY_RANK = {
    "Active Exploitation": 6,
    "Weaponized":          5,
    "Commodity Exploit":   4,
    "Proof of Concept":    3,
    "Social Engineering":  2,
    "Not Applicable":      1,
}

CONFIDENCE_RANK = {
    "High":   3,
    "Medium": 2,
    "Low":    1,
}


# ─────────────────────────────────────────────
# PARSE MDR REPORT
# ─────────────────────────────────────────────

def _extract_header_fields(report_text: str) -> dict:
    """
    Extracts top-level report metadata using simple regex.
    These fields have consistent format across MDR reports.
    """
    date_match  = re.search(r"##\s+(\w+ \d{4})", report_text)
    risk_match  = re.search(r"\*\*Risk level:\s*(\w+)\.\*\*", report_text, re.IGNORECASE)
    return {
        "report_date": date_match.group(1)  if date_match  else None,
        "risk_level":  risk_match.group(1)  if risk_match  else None,
    }


def _split_actor_sections(report_text: str) -> list:
    """
    Splits the MDR markdown into one section per threat actor.

    Demarcation: each actor section starts with a ### header:
        ### 1. CrimsonJackal — "Gateway Breaker"
        ### 2. RedMantis — "Collaboration Breach"

    Each section ends where the next ### begins or the document ends.
    Returns list of (actor_name, section_text) tuples.

    This pre-splitting means:
        - Gemini parses ONE actor at a time — no risk of cross-contamination
        - If a section is missing a field, it affects only that actor
        - Works even if actor sections are in different order
    """
    # Split on ### headers, keeping the delimiter
    sections   = re.split(r"(?=^### )", report_text, flags=re.MULTILINE)
    actor_sections = []

    for section in sections:
        # Match ### N. ActorName — "CampaignName" pattern
        header = re.match(r"###\s+\d+\.\s+(.+?)(?:\s+—|\s+-)\s+", section)
        if header:
            actor_name = header.group(1).strip()
            actor_sections.append((actor_name, section.strip()))

    return actor_sections


def _parse_single_actor(
    actor_name: str,
    section_text: str,
    gemini_client,
    gemini_model: str,
) -> dict:
    """
    Parses one actor section using Gemini.
    Sending one actor at a time prevents cross-contamination between actors
    and makes it easier to detect and handle parsing failures per actor.
    """
    prompt = f"""You are a threat intelligence analyst. Extract structured data
from this ONE threat actor section of an MDR advisory.

Return ONLY valid JSON for this single actor. No markdown, no preamble.

Fields to extract:
- name: exact actor name (it is: {actor_name})
- campaign: campaign name from the header
- cves: list of real CVE IDs mentioned (format "CVE-YYYY-NNNNN", exclude synthetic ones)
- ransomware: "Yes" or "No"
- ransomware_family: ransomware name if mentioned, else null
- confidence: "High", "Medium", or "Low"
  Use "High" if confirmed victims are mentioned.
  Use "Medium" if suspected or likely.
  Use "Low" if theoretical or no evidence.
- maturity: exactly one of:
  "Active Exploitation" — confirmed victims or active exploitation confirmed
  "Weaponized"         — exploit chain described, no confirmed victims
  "Proof of Concept"   — PoC exists but not weaponized
  "Not Applicable"     — no exploit details
- target_region: "Middle East", "Global", "MENA", or as stated
- confirmed_victims: true if report explicitly says confirmed victims, else false
- victim_location: city/country if mentioned, else null
- dwell_time_days: average dwell time string if mentioned (e.g. "4-6"), else null
- iocs: list of IOC strings EXACTLY as written under the IOCs section.
  Each IOC is a separate list item. Do not merge multiple IOCs into one string.
- summary: one sentence describing the threat

ACTOR SECTION:
{section_text}

Return JSON for this actor only:
{{ "name": "...", "campaign": "...", ... }}"""

    response = gemini_client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
        ),
    )

    raw = response.text.strip()
    raw = raw.strip('`').strip()
    if raw.startswith('json'): raw = raw[4:].strip()
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[MDR]   Parse failed for {actor_name}: {e}")
        return {"name": actor_name, "parse_error": str(e)}


def parse_mdr_report(
    report_path: str,
    gemini_client,
    gemini_model: str,
    out_dir: str,
) -> dict:
    """
    Reads the MDR markdown advisory and extracts structured per-actor data.

    Approach — section-by-section parsing:
        1. Extract top-level metadata (date, risk level) via regex
        2. Split report into one section per actor using ### headers
           Each ### N. ActorName header is a clear demarcation
        3. Parse each actor section independently via Gemini
           One actor per call — no cross-contamination between actors
           If one actor fails to parse, others are unaffected

    Why section-by-section:
        Sending the whole report at once risks Gemini mixing fields between
        actors (e.g. IronVeil's IOCs appearing under CrimsonJackal).
        Splitting first ensures each actor is parsed in isolation.

    Saves: {out_dir}/mdr_parsed.json
    """
    print("\n[MDR] Parsing threat report...")

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Step 1: Extract header metadata via regex (fast, no LLM needed)
    header = _extract_header_fields(report_text)
    print(f"[MDR] Report date: {header['report_date']}  |  "
          f"Risk level: {header['risk_level']}")

    # Step 2: Split into per-actor sections
    actor_sections = _split_actor_sections(report_text)
    print(f"[MDR] Found {len(actor_sections)} actor sections: "
          f"{[name for name, _ in actor_sections]}")

    # Step 3: Parse each actor independently
    actors = []
    for actor_name, section_text in actor_sections:
        print(f"[MDR]   Parsing: {actor_name}...")
        actor_data = _parse_single_actor(
            actor_name, section_text, gemini_client, gemini_model
        )
        actors.append(actor_data)

    mdr_data = {
        "report_date":  header["report_date"],
        "risk_level":   header["risk_level"],
        "actors":       actors,
        "parsed_at":    datetime.now().isoformat(),
        "source_file":  str(report_path),
    }

    # Save parsed output
    out_path = Path(out_dir) / "mdr_parsed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mdr_data, f, indent=2, ensure_ascii=False)

    n_iocs = sum(len(a.get("iocs", [])) for a in actors)
    print(f"[MDR] Parsed {len(actors)} actors  |  {n_iocs} total IOCs")
    print(f"[MDR] Saved: {out_path}")

    return mdr_data


# ─────────────────────────────────────────────
# APPLY MDR OVERRIDES
# ─────────────────────────────────────────────

def apply_mdr_overrides(
    threats_df: pd.DataFrame,
    mdr_data: dict,
    out_dir: str,
) -> tuple:
    """
    Compares MDR intelligence against threat_intelligence.csv rows.
    Overrides fields where MDR contains newer/stronger intelligence.

    Override rules (MDR wins when):
        confidence : MDR rank > CSV rank  (e.g. Medium → High)
        maturity   : MDR rank > CSV rank  (e.g. Weaponized → Active Exploitation)
        ransomware : MDR says Yes, CSV says No

    Returns:
        (updated_threats_df, change_log)

    change_log is a list of dicts, one per override applied:
        {
            "intel_id":       "TI-3001",
            "actor":          "CrimsonJackal",
            "cve":            "CVE-2024-21762",
            "field":          "confidence",
            "old_value":      "Medium",
            "new_value":      "High",
            "reason":         "MDR reports confirmed victim in Dubai this week",
            "score_impact":   "dim_threat will increase — confidence multiplier raised"
        }

    Saved to: {out_dir}/mdr_changes.json
    """
    print("\n[MDR] Applying intelligence overrides...")

    df          = threats_df.copy()
    actors_map  = {a["name"]: a for a in mdr_data.get("actors", [])}
    change_log  = []

    for idx, row in df.iterrows():
        actor = row.get("threat_actor", "")
        if actor not in actors_map:
            continue

        mdr_actor = actors_map[actor]
        overrides = {}

        # ── Confidence override ──
        csv_conf = str(row.get("confidence", "Low"))
        mdr_conf = str(mdr_actor.get("confidence", "Low"))
        if CONFIDENCE_RANK.get(mdr_conf, 0) > CONFIDENCE_RANK.get(csv_conf, 0):
            overrides["confidence"] = (csv_conf, mdr_conf,
                "MDR reports confirmed victims — confidence upgraded")

        # ── Maturity override ──
        csv_mat = str(row.get("exploit_maturity", "Not Applicable"))
        mdr_mat = str(mdr_actor.get("maturity", "Not Applicable"))
        if MATURITY_RANK.get(mdr_mat, 0) > MATURITY_RANK.get(csv_mat, 0):
            overrides["exploit_maturity"] = (csv_mat, mdr_mat,
                f"MDR reports active exploitation evidence — maturity upgraded")

        # ── Ransomware override ──
        csv_ransom = str(row.get("ransomware_association", "No"))
        mdr_ransom = str(mdr_actor.get("ransomware", "No"))
        if mdr_ransom == "Yes" and csv_ransom == "No":
            overrides["ransomware_association"] = ("No", "Yes",
                f"MDR confirms ransomware: {mdr_actor.get('ransomware_family','unknown')}")

        # ── Apply overrides and log ──
        for field, (old_val, new_val, reason) in overrides.items():
            df.at[idx, field] = new_val

            # Describe score impact
            if field == "confidence":
                score_impact = (
                    f"dim_threat multiplier: "
                    f"{old_val}({_conf_mult(old_val):.1f}x) → "
                    f"{new_val}({_conf_mult(new_val):.1f}x)"
                )
            elif field == "exploit_maturity":
                score_impact = (
                    f"dim_threat maturity component changes"
                )
            else:
                score_impact = "dim_threat ransomware bonus added (+15 pts)"

            change_log.append({
                "intel_id":     row.get("intel_id", ""),
                "actor":        actor,
                "cve":          row.get("matched_cve_or_control", ""),
                "campaign":     row.get("campaign_name", ""),
                "field":        field,
                "old_value":    old_val,
                "new_value":    new_val,
                "reason":       reason,
                "score_impact": score_impact,
            })

            print(f"[MDR]   OVERRIDE: {actor} | {row.get('matched_cve_or_control','')} | "
                  f"{field}: {old_val} → {new_val}")

    # ── Save change log ──
    output = {
        "metadata": {
            "mdr_report_date":  mdr_data.get("report_date"),
            "mdr_risk_level":   mdr_data.get("risk_level"),
            "total_overrides":  len(change_log),
            "actors_matched":   len(set(c["actor"] for c in change_log)),
            "applied_at":       datetime.now().isoformat(),
        },
        "changes": change_log,
    }

    out_path = Path(out_dir) / "mdr_changes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if change_log:
        print(f"[MDR] {len(change_log)} overrides applied  |  saved: {out_path}")
    else:
        print(f"[MDR] No overrides needed — CSV already matches MDR intelligence")

    return df, change_log


def _conf_mult(conf: str) -> float:
    """Confidence multiplier — must match CONFIDENCE_MULT in main.py."""
    return {"High": 1.5, "Medium": 1.3, "Low": 1.1}.get(conf, 1.0)


# ─────────────────────────────────────────────
# BUILD IOC MAP
# ─────────────────────────────────────────────

def build_ioc_map(mdr_data: dict) -> dict:
    """
    Returns dict: actor_name -> list of IOC strings.
    Used by explanation builder to add IOC section per matched risk.

    Example:
    {
        "CrimsonJackal": [
            "Inbound connections from 185.220.x.x/24",
            "VPN log entries with empty User-Agent strings",
            "Scheduled tasks named SystemUpdate or WinDefend"
        ],
        "IronVeil": [ ... ]
    }
    """
    ioc_map = {}
    for actor in mdr_data.get("actors", []):
        name = actor.get("name", "")
        iocs = actor.get("iocs", [])
        if name and iocs:
            ioc_map[name] = iocs
    return ioc_map
