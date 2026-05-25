# TawasolPay — AI Cyber Risk Assistant

An AI-powered system that ingests TawasolPay's asset inventory, vulnerability data, threat intelligence, business service context, and MDR advisory, then produces a prioritised top-5 risk list with evidence, NIST SP 800-53 remediation guidance, and audit evidence — automatically.
---

## What the system produces

For each of the top-5 risks, the system outputs:

- **The asset and vulnerability** — CVE, CVSS, days open, environment
- **Matched threat intelligence** — actor, campaign, ransomware flag, maturity, confidence
- **Business service at risk** — criticality, revenue impact, compliance scope, RTO
- **Evidence for the ranking** — a plain-English explanation of every factor that drove the score
- **NIST SP 800-53 remediation guidance** — retrieved from the actual catalog, not hardcoded
- **Audit evidence to collect** — from NIST SP 800-53A assessment procedures
- **Active IOCs** — from the MDR advisory, matched per threat actor
---

## Run locally

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourname/tawasol-cyber-risk
cd tawasol-cyber-risk
```

---

### Step 2 — Create a virtual environment

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Get a Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Click **Get API key** → **Create API key**
4. Copy the key

The free tier is sufficient. No billing setup required.

---

### Step 5 — Run the Streamlit app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

### Step 6 — Upload your files

In the sidebar on the left:

**Enter your Gemini API key** in the password field at the top.

**Upload the required data files:**

| File | Where to get it |
|------|-----------------|
| `assets.csv` | TawasolPay data pack |
| `vulnerabilities.csv` | TawasolPay data pack |
| `business_services.csv` | TawasolPay data pack |
| `threat_intelligence.csv` | TawasolPay data pack |
| `NIST_SP-800-53_rev5_catalog_load.csv` | Required on first run — not needed after FAISS is built |

**Upload optional files (recommended):**

| File | What it adds |
|------|-------------|
| `sp800-53ar5-assessment-procedures.csv` | Audit evidence section per risk |
| `synthetic_threat_report.md` | MDR intelligence overrides + IOCs |

---

### Step 7 — Choose FAISS option

If this is your **first run**, the sidebar shows:

```
ℹ️ No FAISS index found — will be built on first run. NIST catalog required.
```

Upload the NIST catalog and continue.

If you have **run before**, the sidebar shows:

```
⚡ FAISS index exists
   Model: BAAI/bge-large-en-v1.5
   Vectors: 1009 controls

Use existing FAISS?
● Yes — use existing (catalog not needed)
○ No  — rebuild embeddings (catalog required)
```

Select **Yes** to skip re-embedding and save ~5 minutes. Select **No** only if you have a new catalog file.

---

### Step 8 — Run the pipeline

Click **Run Pipeline**.

---

### Step 9 — View results

After the pipeline completes, the dashboard shows five pages:

- **Top 5 Risks** — ranked risk cards with scores, evidence, IOCs, NIST guidance
- **All Vulnerabilities** — filterable table of all 114 scored vulnerabilities
- **Threat Intelligence** — RAG retrieval detail with FAISS and reranker ranks
- **MDR Advisory** — parsed threat actors, IOCs, overrides applied

---

### Step 10 — Download results

After the pipeline runs, download buttons appear in the sidebar:

```
⬇️ top5_risks.csv
⬇️ all_vulnerabilities.csv
⬇️ rag_detail.json
```

All files are also saved automatically to:
```
AI Cyber Risk System results/
├── top5_risks_with_nist.csv
├── all_vulnerabilities_scored.csv
├── rag_queries_and_retrieved.json
├── mdr_parsed.json
├── mdr_changes.json
├── uploaded_data/          ← your uploaded CSVs saved here
└── rag_cache/              ← FAISS index saved here
```

---
### Required files

| File | Required | Notes |
|------|----------|-------|
| `assets.csv` | Yes | Asset inventory |
| `vulnerabilities.csv` | Yes | Open vulnerabilities |
| `business_services.csv` | Yes | Business service context |
| `threat_intelligence.csv` | Yes | Threat actor campaigns |
| `NIST_SP-800-53_rev5_catalog_load.csv` | First run only | NIST control catalog — reused from cache after first run |
| `sp800-53ar5-assessment-procedures.csv` | Optional | Enables audit evidence section |
| `synthetic_threat_report.md` | Optional | Enables MDR intelligence overrides and IOC extraction |

---

## Supporting Question 1 — The data split

**What was embedded:** The NIST SP 800-53 Rev. 5 control catalog — 1,009 active controls after excluding 180 withdrawn controls. Each control is embedded as a text chunk containing the control ID, name, full control text, and compressed discussion (first 6 sentences, cross-references stripped). These were embedded because relevance to a specific risk cannot be determined by filtering or joining — it requires semantic similarity search across the full policy text of 1,009 controls.

**What was queried as structured records:** All five CSVs (assets, vulnerabilities, business services, threat intelligence, remediation guidance), the CISA KEV catalog fetched live, and the MDR advisory. These were queried directly because they have known schemas, defined relationships (asset_id → business_service_id), and can be joined, filtered, and aggregated without semantic understanding. The MDR advisory was parsed by Gemini into structured JSON then treated as structured data for override comparison against threat_intelligence.csv.

The assessment procedures CSV (NIST SP 800-53A) was also queried as structured records — looked up by exact control ID after retrieval identifies the primary control, not embedded, because it contains audit procedures rather than policy descriptions and exact-match lookup is more reliable than similarity search for this use case.

---

## Supporting Question 2 — Where it goes wrong

**1. CISA KEV miss on novel CVEs**

If a CVE is real and actively exploited but has not yet been added to the CISA KEV catalog, the system will not set `on_kev = True` for that vulnerability. This silently reduces its `dim_threat` score — the KEV bonus of 25 points is not applied — so the vulnerability may rank below others that are on the KEV list even if it is equally or more dangerous. Detection: the system logs how many KEV entries it fetched and prints a warning if the fetch fails entirely. Mitigation: supplement KEV with NVD CVSS enrichment and EPSS (Exploit Prediction Scoring System) scores as additional exploitation likelihood signals.

**2. Retrieval enhancement controls over base controls**

The BGE reranker sometimes promotes enhancement controls (e.g. SI-2(2), IA-2(8)) over their base controls (SI-2, IA-2). Enhancement controls often have shorter, more focused control text that matches the natural-language reranker query more precisely. The consequence is that the evidence lookup fails — the NIST SP 800-53A assessment procedures CSV has entries for base controls but not always for specific enhancements — producing an empty evidence block. Detection: visible in the output as "No assessment procedures found for SI-2(2)". Mitigation: when evidence lookup returns empty for an enhancement, automatically fall back to the parent base control ID (strip the parenthetical suffix) for evidence retrieval.

**3. MDR parsing breaks on non-standard report formats**

The MDR parser uses `### N. ActorName` markdown headers as section demarcation. If an MDR advisory arrives with different formatting — numbered differently, using `##` headers, or plain prose — the section splitter will either fail to find any actors or merge multiple actors into one section, causing Gemini to extract incorrect or mixed fields. Detection: the parsed `mdr_parsed.json` is saved and should be reviewed before relying on MDR overrides — the number of actors found is printed at runtime. Mitigation: add a full-report fallback parse path that sends the entire document to Gemini if section splitting finds zero actors, and add a validation step that checks parsed actor names against threat_intelligence.csv actor names.

---

## Supporting Question 3 — One thing I would change

The single most important improvement would be upgrading the embedding model from `BAAI/bge-large-en-v1.5` (512-token limit) to `BAAI/bge-m3` (8192-token limit). The current model silently truncates 33 of 1,009 NIST control chunks at the 512-token boundary, including RA-5 (Vulnerability Monitoring) and AC-2 (Account Management), which are among the most relevant controls for this scenario. The truncation removes the most distinctive vocabulary from those controls — the specific policy language that would match the risk queries — reducing their similarity scores and causing the reranker to work with incomplete representations. With bge-m3, all 1,009 controls would be fully embedded, SI-2 (Flaw Remediation) would likely surface as the primary control for the unpatched Fortinet VPN vulnerabilities instead of RA-5, and the overall retrieval quality would improve across all risks. The reason it was not used here is RAM: bge-m3 requires approximately 4 GB of memory to run on CPU, which exceeds the available RAM on the development machine. On any machine with 8 GB or more RAM it would be a direct drop-in replacement.

---

## File structure

```
tawasol-cyber-risk/
├── app.py                    Streamlit dashboard (upload files, run pipeline, view results)
├── main.py                   Pipeline orchestrator (scoring, KEV, MDR integration)
├── rag_nist.py               NIST RAG module (FAISS + reranker + Gemini guidance)
├── mdr_processor.py          MDR advisory parser and threat intel override engine
├── requirements.txt          Python dependencies
├── README.md                 This file
└── AI Cyber Risk System results/   Created on first run
    ├── uploaded_data/              Saved copies of uploaded CSV files
    ├── rag_cache/                  FAISS index (persists across runs)
    │   ├── nist_faiss.index
    │   ├── nist_faiss_fingerprint.json
    │   ├── nist_embeddings_cache.npz
    │   └── nist_chunks_cache.json
    ├── top5_risks_with_nist.csv
    ├── all_vulnerabilities_scored.csv
    ├── rag_queries_and_retrieved.json
    ├── mdr_parsed.json
    └── mdr_changes.json
```

---

## Key design decisions

**Scoring is multi-dimensional, not CVSS-only.** A CVSS 10 on an internal dev server ranks below a CVSS 8 on an internet-exposed payment gateway with an active ransomware campaign. The threat dimension (30% weight) captures KEV status, actor maturity, ransomware association, and regional targeting — none of which are in CVSS.

**The primary NIST control is selected by code, not by the LLM.** The reranker scores all 15 FAISS candidates and the top-ranked control becomes the primary. Gemini only explains the pre-selected control using the retrieved control text. This ensures guidance comes from the actual NIST document and not from the LLM's training data.

**MDR intelligence overrides structured data.** When the MDR advisory contains stronger intelligence than the threat_intelligence.csv (e.g. maturity upgraded from Weaponized to Active Exploitation based on confirmed victims), the system applies the override, logs the change, and recalculates scores. All changes are saved to `mdr_changes.json` with old value, new value, reason, and score impact.

**FAISS cache persists across runs.** The 1,009-control embedding takes 3-5 minutes on first run. After that, the FAISS index loads from disk in under 1 second. The Streamlit app asks whether to reuse the existing index or rebuild — rebuilding is only needed when the NIST catalog changes.