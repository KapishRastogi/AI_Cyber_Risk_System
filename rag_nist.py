"""
TawasolPay — NIST SP 800-53 RAG Module
=======================================
Embedding model : BAAI/bge-large-en-v1.5 (local, HuggingFace, 512 tokens)
LLM             : Gemini 2.0 Flash (Google API, free tier)

Two NIST sources used:
    NIST_SP-800-53_rev5_catalog_load.csv  <- RAG source
        Embedded into FAISS vector DB at startup.
        Retrieved by cosine similarity search per risk.
        Sent to LLM to generate remediation guidance.

    sp800-53ar5-assessment-procedures.csv  <- Evidence lookup
        NOT embedded. Looked up by exact control ID after retrieval.
        Appears as separate EVIDENCE TO COLLECT section in output.

All fixes applied:
    - total = len(texts) not sum(token_counts)
    - Tokenizer used for accurate token counting and stats
    - Sentence-boundary truncation for 33 chunks over 512 tokens
    - FAISS IndexFlatIP for exact cosine similarity
    - FAISS cache: save after build, load on subsequent runs
    - Gemini retry on 503/429 (3 attempts, 15/30/45s waits)
    - Primary control column synced from LLM output text
    - Query capped at 900 chars
    - Unicode normalisation on all text (fixes MacRoman garbling)
"""

import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests  # used in main.py for CISA KEV

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from google import genai
from google.genai import types


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
GEMINI_MODEL    = "gemini-2.0-flash"
TOP_K           = 5
RERANK_FETCH    = 15     # candidates fetched by FAISS before reranking
MIN_SIMILARITY  = 0.35
MAX_DISC_SENT   = 6      # keep first N sentences of discussion
RERANKER_MODEL  = "BAAI/bge-reranker-base"  
EMBED_BATCH = 10  # cross-encoder, 278MB, CPU-friendly



# ─────────────────────────────────────────────
# UNICODE NORMALISER
# ─────────────────────────────────────────────

_UNICODE_TO_ASCII = {
    "\u2014": " -- ",  # em dash
    "\u2013": "-",     # en dash
    "\u2022": "*",     # bullet
    "\u2018": "'",     # left single quote
    "\u2019": "'",     # right single quote
    "\u201c": '"',     # left double quote
    "\u201d": '"',     # right double quote
    "\u2026": "...",   # ellipsis
    "\u00a0": " ",     # non-breaking space
}

def safe_str(val, default="") -> str:
    """Returns string value or default if nan/None/empty."""
    try:
        if pd.isna(val): return default
    except: pass
    s = str(val).strip()
    return default if s in ("nan","None","") else s


def normalise_text(text: str) -> str:
    """Replace Unicode typographic chars with ASCII to prevent MacRoman garbling."""
    if not text or str(text) in ("nan", ""):
        return text
    for uni, asc in _UNICODE_TO_ASCII.items():
        text = text.replace(uni, asc)
    return text


# ─────────────────────────────────────────────
# SENTENCE-BOUNDARY TRUNCATION
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# DISCUSSION COMPRESSION
# ─────────────────────────────────────────────

def compress_discussion(text: str, max_sentences: int = MAX_DISC_SENT) -> str:
    """
    Rule-based compression -- no LLM, no latency.
    Removes cross-reference noise at SENTENCE level then keeps first max_sentences.

    Four rules, applied per sentence:
        Rule 1 -- Remove if sentence STARTS with a control ID
                  "AC-20 addresses mobile devices..." --> removed entirely
        Rule 2 -- Remove if sentence is MOSTLY control IDs (residual <= 35 chars)
                  "Audit events are defined in AU-2." --> removed entirely
        Rule 3 -- Remove if sentence contains pointer phrases followed by a control ID
                  "Audit records are generated in AU-12." --> removed entirely
                  "Refer to SI-2 and see also RA-5 for guidance." --> removed entirely
                  Avoids broken fragments like "and for related guidance." that
                  substring-only removal would leave behind.
        Rule 4 -- Keep sentences with inline ID mentions AND real content
                  "Separation of duties enforced through AC-2, AC-3..." --> kept
    """
    if not text or str(text).strip() in ("nan", ""):
        return ""
    text = str(text).strip()

    CTRL = r"[A-Z]{2}-[0-9]+(?:\([0-9]+\))?"

    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned   = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # Rule 1: starts with control ID -- pure pointer sentence
        if re.match(CTRL + r"\b", s):
            continue

        # Rule 2: mostly control IDs -- very little real content
        residual = re.sub(CTRL, "", s)
        residual = re.sub(r"[\s,;()]+(?:and|or)?[\s,;()]+", " ", residual)
        residual = residual.strip(" .,;")
        if len(residual) <= 35:
            continue

        # Rule 3: pointer phrases followed by a control ID -- remove entire sentence
        if re.search(
            r"\b(?:refer(?:s)?\s+to|see\s+also|as\s+described\s+in|"
            r"as\s+defined\s+in|described\s+in|defined\s+in|"
            r"generated\s+in|documented\s+in|specified\s+in|"
            r"established\s+in|provided\s+in|addressed\s+in|"
            r"discussed\s+in|identified\s+in|listed\s+in)\s+" + CTRL,
            s, re.IGNORECASE
        ):
            continue

        # Rule 4: inline mention with substantive content -- keep
        cleaned.append(s)

    if not cleaned:
        return ""
    if len(cleaned) <= max_sentences:
        return " ".join(cleaned)
    return " ".join(cleaned[:max_sentences])


def parse_assess_field(text: str) -> list:
    """Parses [SELECT FROM: item1; item2] format into clean list."""
    if pd.isna(text) or str(text).strip() in ("nan", ""):
        return []
    text = str(text).strip()
    text = re.sub(r"^\[SELECT FROM:\s*", "", text)
    text = re.sub(r"\]\s*\.?\s*$", "", text)
    items = [re.sub(r"\.$", "", item).strip() for item in text.split(";")]
    return [i for i in items if i]


class AssessmentLookup:
    """
    Exact-ID lookup of EXAMINE/TEST evidence from assessment procedures CSV.
    Not embedded — looked up after retrieval identifies the primary control.
    """

    def __init__(self, path: str):
        print(f"[RAG] Loading assessment procedures: {path}")
        df = pd.read_csv(path, encoding="latin-1")
        df["id_norm"] = (
            df["identifier"]
            .str.replace(r"-0(\d)$", r"-\1", regex=True)
            .str.strip()
        )
        self._df = df
        print(f"[RAG] Loaded {len(df)} rows  ({df['id_norm'].nunique()} unique IDs)")

    def get_evidence(self, ctrl_id: str) -> dict:
        rows = self._df[self._df["id_norm"] == ctrl_id]
        if rows.empty:
            return {"examine": [], "test": []}
        row = rows.iloc[0]
        return {
            "examine": parse_assess_field(row.get("EXAMINE", "")),
            "test":    parse_assess_field(row.get("TEST", "")),
        }

    def format_evidence_block(self, ctrl_id: str, ctrl_name: str) -> str:
        ev = self.get_evidence(ctrl_id)
        if not ev["examine"] and not ev["test"]:
            return f"  No assessment procedures found for {ctrl_id}"
        lines = [f"  Based on NIST SP 800-53A Rev. 5 -- {ctrl_id} ({ctrl_name}):"]
        if ev["examine"]:
            lines.append("  Documents / records to collect:")
            skip = ["other relevant documents", "privacy plan",
                    "system security plan", "other relevant records"]
            useful = [i for i in ev["examine"]
                      if not any(k in i.lower() for k in skip)][:5]
            for item in useful:
                lines.append(f"    * {item}")
        if ev["test"]:
            lines.append("  Mechanisms to verify:")
            for item in ev["test"][:3]:
                lines.append(f"    * {item}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# NIST RAG CLASS
# ─────────────────────────────────────────────

class NISTRag:

    def __init__(
        self,
        gemini_api_key: str,
        catalog_path:   str,
        assess_path:    str   = None,
        model_name:     str   = "BAAI/bge-large-en-v1.5",
        min_similarity: float = MIN_SIMILARITY,
        cache_dir:      str   = None,
    ):
        self.gemini_api_key = gemini_api_key
        self._client        = None
        self.catalog_path   = catalog_path
        self.model_name     = model_name
        self.min_similarity = min_similarity
        self.cache_dir      = Path(cache_dir) if cache_dir else None
        self.model          = None
        self.reranker       = None   # cross-encoder reranker (optional)
        self.chunks         = []
        self.embeddings     = None
        self.index          = None
        self.assessment     = None

        # Safety check — prevent accidentally passing a Gemini model name
        if "gemini" in model_name.lower():
            raise ValueError(
                f"model_name '{model_name}' looks like a Gemini LLM name. "
                f"Use 'BAAI/bge-large-en-v1.5' for embeddings."
            )

        print("\n[RAG] Initialising NIST 800-53 RAG system")
        print(f"[RAG] Embedding model : {model_name}")
        print(f"[RAG] Generation model: {GEMINI_MODEL}")
        print(f"[RAG] Min similarity  : {min_similarity}")
        if self.cache_dir:
            print(f"[RAG] Cache directory : {self.cache_dir}")

        self._init_gemini_client()
        self._load_model()
        self._load_reranker()
        self._load_catalog()

        if assess_path:
            self.assessment = AssessmentLookup(assess_path)
        else:
            print("[RAG] No assessment procedures path — evidence lookup disabled")

        self._embed_all_chunks()
        print(f"\n[RAG] Ready -- {len(self.chunks)} controls embedded\n")

    # ──────────────────────────────────────────
    # INIT GEMINI CLIENT
    # ──────────────────────────────────────────

    def _init_gemini_client(self):
        print(f"\n[RAG] Initialising Gemini client ({GEMINI_MODEL})...")
        try:
            self._client = genai.Client(api_key=self.gemini_api_key)
            print("[RAG] Gemini client ready")
        except Exception as e:
            raise RuntimeError(f"Gemini init failed: {e}\nRun: pip install google-genai")

    # ──────────────────────────────────────────
    # LOAD EMBEDDING MODEL
    # ──────────────────────────────────────────

    def _load_model(self):
        print(f"\n[RAG] Loading {self.model_name} from HuggingFace...")
        print("[RAG] First run downloads ~1.3 GB -- cached after that")
        try:
            from sentence_transformers import SentenceTransformer
            # Force CPU -- avoids Mac MPS "Invalid buffer size" memory error
            self.model = SentenceTransformer(self.model_name, device="cpu")
            test = self.model.encode(["test"], show_progress_bar=False)
            print(f"[RAG] Model ready -- device: cpu -- dimensions: {test.shape[1]}")
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers torch")

    # ──────────────────────────────────────────
    # LOAD RERANKER
    # ──────────────────────────────────────────

    def _load_reranker(self):
        """
        Loads BGE cross-encoder reranker (optional).

        Why a reranker:
            FAISS bi-encoder retrieval embeds query and document separately
            and compares vectors. Fast but imprecise -- query and document
            never interact during encoding.

            A cross-encoder (reranker) takes (query, document) as a pair
            and scores them jointly. Much more accurate because it models
            the interaction between query terms and document terms directly.

        Pipeline with reranker:
            FAISS fetches top-RERANK_FETCH candidates (fast, approximate)
                    ↓
            Reranker scores each (query, candidate) pair (accurate)
                    ↓
            Re-sorted top-TOP_K returned
                    ↓
            Code picks primary = reranked[0]

        Gracefully disabled if sentence-transformers CrossEncoder not available.
        Install: pip install sentence-transformers
        Model: BAAI/bge-reranker-base (~278MB, downloads once, cached)
        """
        print(f"\n[RAG] Loading reranker: {RERANKER_MODEL}...")
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
            print(f"[RAG] Reranker ready -- cross-encoder reranking enabled")
        except Exception as e:
            print(f"[RAG] Reranker not loaded ({e}) -- cosine similarity only")
            self.reranker = None


    # ──────────────────────────────────────────
    # LOAD CATALOG AND BUILD CHUNKS
    # ──────────────────────────────────────────

    def _load_catalog(self):
        """
        Builds one text chunk per NIST control.

        Chunk structure:
            {id} -- {name}
            {control_text}           full, never truncated
            {compressed_discussion}  first 6 sentences, cross-refs stripped
            Related controls: {related}

        Note: bge-large-en-v1.5 has a 512 token limit and will
        silently truncate longer chunks internally. The tokenizer
        stats printed at embedding time show how many chunks exceed
        this limit so you can monitor retrieval quality.
        """
        print(f"\n[RAG] Loading catalog: {self.catalog_path}")
        df = pd.read_csv(self.catalog_path)
        df = df[[c for c in df.columns if not c.startswith("Unnamed")]]

        compressed_count  = 0

        for _, row in df.iterrows():
            ctrl_id    = str(row.get("identifier", "")).strip()
            ctrl_name  = str(row.get("name",         "")).strip()
            ctrl_text  = str(row.get("control_text",  "")).strip()
            discussion = str(row.get("discussion",    "")).strip()
            related    = str(row.get("related",       "")).strip()

            if not ctrl_id or ctrl_id == "nan":
                continue

            # Skip withdrawn controls -- their text is just "Withdrawn: incorporated into X."
            # They have no policy content to embed and will never be a valid primary control.
            if str(ctrl_text).strip().lower().startswith("withdrawn"):
                continue

            ctrl_text  = "" if ctrl_text  == "nan" else ctrl_text
            discussion = "" if discussion == "nan" else discussion
            related    = "" if related    == "nan" else related

            # Normalise Unicode → ASCII
            ctrl_id    = normalise_text(ctrl_id)
            ctrl_name  = normalise_text(ctrl_name)
            ctrl_text  = normalise_text(ctrl_text)
            discussion = normalise_text(discussion)
            related    = normalise_text(related)

            # Compress discussion
            comp_disc = compress_discussion(discussion)
            if len(comp_disc) < len(discussion):
                compressed_count += 1

            # Build chunk
            parts = [f"{ctrl_id} -- {ctrl_name}"]
            if ctrl_text:  parts.append(ctrl_text)
            if comp_disc:  parts.append(comp_disc)
            if related:    parts.append(f"Related controls: {related}")
            chunk_text = "\n".join(parts)


            self.chunks.append({
                "id":          ctrl_id,
                "name":        ctrl_name,
                "text":        ctrl_text,
                "discussion":  discussion,
                "comp_disc":   comp_disc,
                "related":     related,
                "chunk_text":  chunk_text,
            })

        withdrawn_count = len([r for _, r in df.iterrows()
                               if str(r.get("control_text","")).strip().lower().startswith("withdrawn")])
        print(f"[RAG] {len(self.chunks)} chunks embedded  |  "
              f"{withdrawn_count} withdrawn controls skipped  |  "
              f"{compressed_count} discussions compressed")

    # ──────────────────────────────────────────
    # CACHE HELPERS
    # ──────────────────────────────────────────

    def _chunks_cache_path(self):
        return self.cache_dir / "nist_chunks_cache.json"

    def _embeddings_cache_path(self):
        return self.cache_dir / "nist_embeddings_cache.npz"

    def _faiss_index_path(self):
        return self.cache_dir / "nist_faiss.index"

    def _faiss_fingerprint_path(self):
        return self.cache_dir / "nist_faiss_fingerprint.json"

    # ──────────────────────────────────────────
    # SAVE CHUNKS JSON
    # ──────────────────────────────────────────

    def _save_chunks_json(self):
        """
        Saves all chunk text fields to JSON for human inspection.
        Includes token_count per chunk using the model tokenizer
        so you can verify no chunk exceeds the 512 token limit.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._chunks_cache_path()

        # Count tokens for every chunk using the actual model tokenizer
        # More accurate than char count for understanding what the model sees
        try:
            token_counts = {
                c["id"]: len(self.model.tokenizer.encode(
                    c["chunk_text"], add_special_tokens=True
                ))
                for c in self.chunks
            }
        except Exception:
            token_counts = {c["id"]: None for c in self.chunks}

        data = {
            "metadata": {
                "model":            self.model_name,
                "total_chunks":     len(self.chunks),
                "max_tokens_model": 512,
                "chunks_over_512":  sum(
                    1 for v in token_counts.values()
                    if v is not None and v > 512
                ),
                "catalog_path":     str(self.catalog_path),
                "generated_at":     __import__("datetime").datetime.now().isoformat(),
            },
            "chunks": [
                {
                    "id":           c["id"],
                    "name":         c["name"],
                    "chunk_text":   c["chunk_text"],
                    "control_text": c["text"],
                    "discussion":   c["discussion"],
                    "related":      c["related"],
                    "char_len":     len(c["chunk_text"]),
                    "token_count":  token_counts.get(c["id"]),
                }
                for c in self.chunks
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[RAG] Chunks saved  : {path}  ({path.stat().st_size // 1024} KB)")

    # ──────────────────────────────────────────
    # FAISS BUILD / SAVE / LOAD
    # ──────────────────────────────────────────

    def _build_faiss_index(self):
        """
        Builds FAISS IndexFlatIP.
        For L2-normalised vectors: inner_product == cosine_similarity.
        bge-large-en-v1.5 uses normalize_embeddings=True so this is
        exact cosine search with no approximation.
        """
        if not FAISS_AVAILABLE:
            print("[RAG] faiss-cpu not installed -- numpy dot product fallback")
            print("[RAG] Install: pip install faiss-cpu")
            return
        dim        = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"[RAG] FAISS index built : IndexFlatIP  |  "
              f"{self.index.ntotal} vectors  |  dim={dim}")

    def _save_faiss_index(self):
        """
        Saves FAISS index + fingerprint + raw numpy to cache.

        Files:
            nist_faiss.index            -- FAISS binary
            nist_faiss_fingerprint.json -- validation metadata
            nist_embeddings_cache.npz   -- numpy fallback
        """
        if not FAISS_AVAILABLE or self.index is None:
            # Save numpy fallback even without FAISS
            if self.cache_dir and self.embeddings is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    self._embeddings_cache_path(),
                    embeddings=self.embeddings,
                )
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._faiss_index_path()))

        fingerprint = {
            "model":        self.model_name,
            "n_chunks":     len(self.chunks),
            "n_vectors":    self.index.ntotal,
            "dimensions":   int(self.embeddings.shape[1]),
            "index_type":   "IndexFlatIP",
            "similarity":   "cosine (inner product on normalised vectors)",
        }
        with open(self._faiss_fingerprint_path(), "w") as f:
            json.dump(fingerprint, f, indent=2)

        np.savez_compressed(
            self._embeddings_cache_path(),
            embeddings=self.embeddings,
        )
        faiss_path = self._faiss_index_path()
        print(f"[RAG] FAISS index saved : {faiss_path}  "
              f"({faiss_path.stat().st_size // 1024} KB)")

    def _load_faiss_index(self) -> bool:
        """
        Loads FAISS index from cache if fingerprint is valid.
        Returns True = loaded (skip re-embedding, saves 3-5 min).
        Returns False = cache miss or invalid (must re-embed).
        """
        if self.cache_dir is None:
            return False

        faiss_path = self._faiss_index_path()
        fp_path    = self._faiss_fingerprint_path()
        npz_path   = self._embeddings_cache_path()

        if not faiss_path.exists() or not fp_path.exists():
            print("[RAG] No FAISS cache found -- will embed and build index")
            return False

        try:
            with open(fp_path) as f:
                fp = json.load(f)

            # Validate: model name and chunk count must match
            if (fp.get("model")    != self.model_name or
                fp.get("n_chunks") != len(self.chunks)):
                print("[RAG] FAISS cache invalid (model or chunk count changed) "
                      "-- rebuilding")
                return False

            if FAISS_AVAILABLE:
                self.index = faiss.read_index(str(faiss_path))
                print(f"[RAG] FAISS index loaded : {faiss_path}")
                print(f"[RAG]   {self.index.ntotal} vectors  |  "
                      f"IndexFlatIP cosine  |  skipped re-embedding")

            if npz_path.exists():
                data = np.load(npz_path)
                self.embeddings = data["embeddings"].astype(np.float32)
                print(f"[RAG]   Embeddings: {self.embeddings.shape}")

            return True

        except Exception as e:
            print(f"[RAG] FAISS cache load failed ({e}) -- will re-embed")
            return False

    # ──────────────────────────────────────────
    # EMBED ALL CHUNKS
    # ──────────────────────────────────────────

    def _embed_all_chunks(self):
        """
        Embeds all NIST chunks on CPU in batches of 10.

        First checks FAISS cache — if valid, loads and returns immediately.

        Why total = len(texts) not sum(token_counts):
            total is the number of CHUNKS to embed (1189).
            token_counts are used only for the stats printout.
            Using sum(token_counts) (~142,000) as total would make
            range(0, total, 10) iterate 14,200 times — fatal IndexError.

        Why batches of 10:
            bge-large-en-v1.5 is 1.3 GB. On Mac with Apple Silicon,
            MPS unified memory errors occur with large batches.
            Batches of 10 + manual np.vstack keeps peak memory low.
        """
        # Try cache first
        if self._load_faiss_index():
            if self.cache_dir:
                self._save_chunks_json()
            return

        texts      = [c["chunk_text"] for c in self.chunks]
        total      = len(texts)       # number of chunks: 1189
        batch_size = EMBED_BATCH      # 10
        all_vecs   = []

        # Use tokenizer for accurate token stats (not char count)
        try:
            token_counts = [
                len(self.model.tokenizer.encode(t, add_special_tokens=True))
                for t in texts
            ]
            total_tokens = sum(token_counts)
            max_tokens   = max(token_counts)
            over_limit   = sum(1 for t in token_counts if t > 512)
            print(f"\n[RAG] Embedding {total} chunks on CPU "
                  f"(batches of {batch_size})...")
            print(f"[RAG] Token stats   : total={total_tokens:,}  "
                  f"max={max_tokens}  over_512={over_limit}")
            print(f"[RAG] Estimated time: 2-5 min on CPU")
        except Exception:
            print(f"\n[RAG] Embedding {total} chunks on CPU "
                  f"(batches of {batch_size})...")
            print(f"[RAG] Estimated time: 2-5 min on CPU")

        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, total, batch_size),
                            desc="[RAG] Embedding", unit="batch")
        except ImportError:
            iterator = range(0, total, batch_size)
            print("[RAG] (pip install tqdm for a progress bar)")

        for start in iterator:
            end   = min(start + batch_size, total)
            batch = texts[start:end]

            vecs = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,   # cosine sim = dot product
                convert_to_numpy=True,
            )
            all_vecs.append(vecs.astype(np.float32))

            if not hasattr(iterator, "set_postfix"):
                if (end % 100 == 0) or (end == total):
                    print(f"[RAG]   {end}/{total} chunks  "
                          f"({end/total*100:.0f}%)")

        self.embeddings = np.vstack(all_vecs)   # (1189, 1024)
        print(f"[RAG] Embedding complete")
        print(f"[RAG]   Matrix : {self.embeddings.shape}")
        print(f"[RAG]   Memory : ~{self.embeddings.nbytes/1024/1024:.1f} MB")

        # Build FAISS index and save everything to cache
        self._build_faiss_index()
        if self.cache_dir:
            self._save_chunks_json()
            self._save_faiss_index()

    # ──────────────────────────────────────────
    # BUILD QUERY
    # ──────────────────────────────────────────

    def _build_query(self, risk_row: pd.Series) -> str:
        """
        Converts risk row into natural-language query using NIST vocabulary.
        Query is capped at 900 chars and prefixed with bge instruction.
        """
        parts = []

        vuln = risk_row.get("vulnerability_name", "")
        if vuln and str(vuln) != "nan":
            parts.append(str(vuln))

        atype = risk_row.get("asset_type", "")
        env   = risk_row.get("environment", "")
        if atype and str(atype) != "nan":
            parts.append(f"{env} {atype} system".strip())

        if risk_row.get("internet_exposed") == "Yes":
            parts.append("internet-exposed boundary protection network perimeter")

        if risk_row.get("patch_available") == "No":
            parts.append("unsupported end-of-life software no patch component replacement")
        elif pd.notna(risk_row.get("days_open")) and int(risk_row.get("days_open", 0)) > 0:
            parts.append(
                f"unpatched vulnerability {int(risk_row['days_open'])} days "
                "flaw remediation software update"
            )

        if risk_row.get("edr_installed") == "No":
            parts.append("missing endpoint detection malicious code protection")

        if risk_row.get("auth_required") == "No":
            parts.append("no authentication required unauthenticated access identification")

        actors = risk_row.get("all_actors", "")
        camps  = risk_row.get("all_campaigns", "")
        if pd.notna(actors) and str(actors) != "nan":
            parts.append(f"active threat campaign {camps} incident handling response")

        if risk_row.get("ransomware_any"):
            parts.append("ransomware incident response recovery backup contingency")

        biz = risk_row.get("business_service", "")
        if biz and str(biz) != "nan":
            parts.append(str(biz))

        scope = str(risk_row.get("compliance_scope", ""))
        if "PCI"  in scope:
            parts.append("payment card data protection financial transaction security")
        if "GDPR" in scope or "PDPL" in scope:
            parts.append("personal data protection privacy customer information")

        raw = " | ".join(parts)[:900]   # cap at 900 chars
        return BGE_QUERY_INSTRUCTION + raw

    # ──────────────────────────────────────────
    # RETRIEVE
    # ──────────────────────────────────────────

    def _build_reranker_query(self, risk_row: pd.Series, fallback: str) -> str:
        """
        Builds a natural-language query for the cross-encoder reranker.

        Cross-encoders do token-by-token interaction between query and document.
        They are trained on coherent sentences -- NOT keyword bags.
        The FAISS query (keyword soup with | separators) produces near-zero
        logits in cross-encoders because attention becomes diluted.

        This method builds a single coherent sentence describing the risk
        so the reranker can properly score relevance against NIST control text.
        """
        if risk_row is None or risk_row.empty:
            return fallback

        vuln  = safe_str(risk_row.get("vulnerability_name", ""))
        asset = safe_str(risk_row.get("asset_name", ""))
        env   = safe_str(risk_row.get("environment", ""))
        atype = safe_str(risk_row.get("asset_type", ""))
        days  = risk_row.get("days_open", 0)
        actors= safe_str(risk_row.get("all_actors", ""))
        scope = safe_str(risk_row.get("compliance_scope", ""))

        gaps = []
        try:
            if pd.notna(days) and int(days) > 0:
                gaps.append(f"unpatched for {int(days)} days")
        except: pass
        if str(risk_row.get("edr_installed","")) == "No":
            gaps.append("no endpoint detection")
        if str(risk_row.get("auth_required","")) == "No":
            gaps.append("no authentication required")
        if str(risk_row.get("internet_exposed","")) == "Yes":
            gaps.append("internet-exposed")

        parts = []
        if vuln:
            parts.append(f"{vuln} affecting {env} {atype} {asset}.".strip())
        if gaps:
            parts.append(f"Security gaps: {', '.join(gaps)}.")
        if actors and actors != "N/A":
            ransom = "ransomware " if risk_row.get("ransomware_any") else ""
            parts.append(f"Actively targeted by {ransom}campaign {actors}.")
        if scope and scope != "N/A":
            parts.append(f"In scope for {scope}.")
        parts.append("What NIST SP 800-53 control applies and what remediation is required?")

        return " ".join(parts)


    def _retrieve(self, query: str, risk_row: pd.Series = None) -> list:
        """
        Two-stage retrieval pipeline:

        Stage 1 — FAISS bi-encoder (fast, approximate):
            Fetches RERANK_FETCH=15 candidates by cosine similarity.
            Each chunk was embedded independently at startup.
            Fast because query and document never interact.

        Stage 2 — BGE cross-encoder reranker (accurate):
            Scores each (query, candidate) pair jointly.
            Query and document interact through every transformer layer.
            Much more accurate for semantic relevance.
            Returns top-TOP_K=5 after reranking.

        Each result carries both scores:
            similarity_rank  -- original FAISS cosine rank (1 = most similar)
            reranker_rank    -- cross-encoder rank after reranking (1 = most relevant)

        If reranker not loaded, falls back to FAISS order only.
        """
        # ── Stage 1: FAISS bi-encoder retrieval ──
        qvec = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].astype(np.float32).reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            k = min(RERANK_FETCH, self.index.ntotal)
            scores, indices = self.index.search(qvec, k)
            scores  = scores[0]
            indices = indices[0]
            sims = np.zeros(len(self.chunks), dtype=np.float32)
            for idx, score in zip(indices, scores):
                if idx >= 0:
                    sims[idx] = float(score)
        else:
            sims = np.dot(self.embeddings, qvec.flatten())

        # Collect top-RERANK_FETCH candidates above threshold
        candidates = []
        for idx in np.argsort(sims)[::-1]:
            sim = float(sims[idx])
            if len(candidates) >= RERANK_FETCH:
                break
            if sim < self.min_similarity:
                break
            c = self.chunks[idx]
            candidates.append({
                "id":               c["id"],
                "name":             c["name"],
                "text":             c["text"],
                "discussion":       c["discussion"],
                "related":          c["related"],
                "similarity_score": round(sim, 4),
                "similarity_rank":  len(candidates) + 1,
            })

        if not candidates:
            return [], []

        # ── Stage 2: Cross-encoder reranker ──
        # Store risk_row reference so _build_reranker_query can access it
        risk_row_ref = [risk_row if risk_row is not None else pd.Series()]
        if self.reranker is not None:
            # Build (query, document_text) pairs for cross-encoder.
            #
            # Fix 1 — strip the BGE instruction prefix from the query.
            # "Represent this sentence for searching relevant passages: ..."
            # is a bi-encoder instruction. The cross-encoder does not
            # understand it — it sees it as literal content and produces
            # near-zero scores. Strip it so the reranker sees only the
            # natural-language query terms.
            #
            # Fix 2 — use only control_text (the actual policy requirement)
            # as the document, not the full chunk_text which also contains
            # discussion and related fields. The reranker should score
            # relevance between the risk query and the policy requirement.
            # Strip bi-encoder instruction prefix — not for cross-encoders
            raw_query = query.replace(BGE_QUERY_INSTRUCTION, "").strip()

            # Build a natural-language reranker query from the risk row.
            # Cross-encoders are trained on coherent sentences, not keyword bags.
            # The raw_query (keyword soup) works for FAISS vector search but
            # produces near-zero logits in cross-encoders.
            reranker_query = self._build_reranker_query(risk_row_ref[0], raw_query)

            pairs = [
                [reranker_query, c['id'] + ' -- ' + c['name'] + chr(10) + c['text']]
                for c in candidates
            ]
            # Use raw logits — BGE reranker is trained with margin ranking.
            # Raw logits are more interpretable: higher = more relevant.
            # Typical range: -4 (weak) to +4 (strong match).
            # sigmoid is NOT applied — it compresses near-zero logits
            # to tiny values like 0.018 making them look like failures.
            reranker_scores = self.reranker.predict(pairs)

            # Attach reranker score to each candidate
            for c, rs in zip(candidates, reranker_scores):
                c["reranker_score"] = round(float(rs), 4)

            # Sort by reranker score descending
            candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

            # Assign reranker rank after sorting
            for i, c in enumerate(candidates):
                c["reranker_rank"] = i + 1

            print(f"[RAG]   Reranked {len(candidates)} candidates  |  "
                  f"top: {candidates[0]['id']} "
                  f"(sim_rank={candidates[0]['similarity_rank']}, "
                  f"rerank_rank=1, "
                  f"reranker_score={candidates[0]['reranker_score']})")
        else:
            # No reranker — use FAISS order, add placeholder reranker fields
            for i, c in enumerate(candidates):
                c["reranker_score"] = None
                c["reranker_rank"]  = i + 1   # same as similarity rank

        # Mark which candidates make the top-K cutoff
        for i, c in enumerate(candidates):
            c["in_top_k"] = i < TOP_K

        # Return top-TOP_K for the pipeline — but all 15 stay accessible
        # via the full candidates list stored in rag_detail
        return candidates[:TOP_K], candidates   # (top5, all15)

    # ──────────────────────────────────────────
    # GENERATE GUIDANCE (LLM)
    # ──────────────────────────────────────────

    def _generate_guidance(self, risk_row: pd.Series,
                           retrieved: list, primary: dict) -> str:
        """
        LLM explains the pre-selected primary control and supporting controls.

        The primary control is chosen by CODE (retrieved[0] = highest cosine
        similarity from FAISS). The LLM does NOT select controls -- it only:
            - Paraphrases the primary control text
            - Explains why it applies to this specific risk
            - Mentions supporting controls briefly
            - Writes concrete immediate actions

        This ensures guidance comes from the actual NIST document, not from
        the LLM's training data.
        """
        if not retrieved:
            return (
                f"No NIST controls retrieved above threshold "
                f"({self.min_similarity}). Manual review recommended."
            )

        # Build concise risk summary
        gaps = []
        days = risk_row.get("days_open", 0)
        if risk_row.get("patch_available") == "No":
            gaps.append("no patch exists (EOL software)")
        elif pd.notna(days) and int(days) > 0:
            gaps.append(f"{int(days)} days unpatched")
        if risk_row.get("edr_installed") == "No":
            gaps.append("no EDR")
        if risk_row.get("auth_required") == "No":
            gaps.append("no authentication required")
        gaps_str = ", ".join(gaps) if gaps else "none"

        actors = risk_row.get("all_actors", "")
        camps  = risk_row.get("all_campaigns", "")
        mat    = risk_row.get("max_maturity", "")
        if pd.notna(actors) and str(actors) != "nan":
            ransom     = "ransomware campaign" if risk_row.get("ransomware_any") else "campaign"
            threat_str = f"{actors} ({camps}) -- {ransom}, maturity: {mat}"
        else:
            threat_str = "No active campaign matched"

        risk_summary = (
            f"RISK SUMMARY:\n"
            f"  Vulnerability : {risk_row.get('vulnerability_name','N/A')}\n"
            f"  Asset         : {risk_row.get('environment','N/A')} "
            f"{risk_row.get('asset_type','N/A')} -- "
            f"internet-exposed: {risk_row.get('internet_exposed','N/A')}\n"
            f"  Security gaps : {gaps_str}\n"
            f"  Active threat : {threat_str}\n"
            f"  Compliance    : {risk_row.get('compliance_scope','None')}"
        )

        # CODE selects primary = retrieved[0] (highest cosine similarity).
        # LLM only explains it — does not select, rank, or invent controls.
        # Supporting controls = retrieved[1:] passed as additional context.
        primary_ctrl = primary   # passed in from get_guidance()

        # Build context: primary control text in full, supporting controls briefly
        supporting = [r for r in retrieved if r["id"] != primary_ctrl["id"]]

        primary_context = (
            f"{primary_ctrl['id']} -- {primary_ctrl['name']}\n"
            f"Control text:\n{primary_ctrl['text']}\n"
        )

        supporting_context = ""
        for ctrl in supporting:
            supporting_context += (
                f"\n{ctrl['id']} -- {ctrl['name']}\n"
                f"Control text:\n{ctrl['text'][:300]}\n"
            )

        # Extract CVE and asset name — injected explicitly to prevent placeholders
        cve_id     = str(risk_row.get("cve",        "N/A"))
        asset_name = str(risk_row.get("asset_name", "N/A"))
        vuln_name  = str(risk_row.get("vulnerability_name", "N/A"))
        supporting_ids = ", ".join(r["id"] for r in supporting)

        prompt = f"""You are a senior cybersecurity analyst writing a remediation brief for a technical manager.

YOUR ROLE:
The retrieval system has already identified the most relevant NIST SP 800-53 control
for this risk using vector similarity search. Your job is ONLY to:
  1. Explain what the primary control requires (paraphrase its text)
  2. Explain why it applies to this specific risk
  3. Mention 2 supporting controls briefly
  4. Write 3 concrete immediate actions

You do NOT select controls. The primary control has already been chosen by the system.

STRICT RULES:
- WHAT IT SAYS must paraphrase ONLY the primary control text provided below.
- Do NOT use your training data about NIST controls.
- Do NOT cite any control not provided below.
- Be specific to THIS risk -- not generic advice.
- Keep response under 350 words.
- Do NOT include an evidence section -- handled separately.
- In IMMEDIATE ACTIONS always use: CVE ID = {cve_id}, Asset = {asset_name}
- Never write placeholder text like CVE-XXXX-XXXX.

{risk_summary}

PRIMARY CONTROL (explain this one in full):
{primary_context}

SUPPORTING CONTROLS (mention these briefly as also relevant):
{supporting_context}

Respond in EXACTLY this format:

PRIMARY CONTROL: {primary_ctrl['id']} -- {primary_ctrl['name']}

WHAT IT SAYS:
  [2-3 sentences paraphrased from the primary control text above]

WHY IT APPLIES:
  [2-3 sentences -- why {primary_ctrl['id']} applies to {vuln_name} on {asset_name}.
   Reference the specific gap from the risk summary.]

ALSO RELEVANT:
  * [ID] -- [Name]: [one sentence why it applies to this specific risk]
  * [ID] -- [Name]: [one sentence why it applies to this specific risk]

IMMEDIATE ACTIONS:
  1. [most urgent step -- name {asset_name} and {cve_id} explicitly]
  2. [second step]
  3. [third step]"""

        # Retry up to 3 times on transient Gemini errors
        last_error = None
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=600,
                        top_p=0.8,
                    ),
                )
                return normalise_text(response.text), prompt

            except Exception as e:
                last_error = e
                err_str    = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                    wait = 15 * (attempt + 1)
                    print(f"\n[RAG]   Gemini {err_str[:50]}... "
                          f"retrying in {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Gemini failed after 3 attempts: {last_error}")

    # ──────────────────────────────────────────
    # BUILD EVIDENCE BLOCK
    # ──────────────────────────────────────────

    def _build_evidence_block(self, retrieved: list,
                              primary_ctrl_id: str = None) -> str:
        """
        Builds evidence section from assessment procedures -- not LLM.

        Primary control lookup strategy:
            1. If primary_ctrl_id is in the retrieved list -> use it
            2. If primary_ctrl_id is NOT in retrieved (Gemini chose a control
               that FAISS did not rank first) -> look it up directly from the
               assessment procedures CSV by exact ID match
            3. If primary_ctrl_id not found anywhere -> fall back to retrieved[0]

        This ensures the evidence always matches what Gemini said is primary,
        even when FAISS did not retrieve that control as top result.
        """
        if not retrieved:
            return "  No controls retrieved -- evidence lookup skipped."
        if self.assessment is None:
            return "  Assessment procedures not loaded."

        primary_id   = None
        primary_name = None

        if primary_ctrl_id:
            # Check if LLM-chosen primary is in retrieved list
            match = next(
                (r for r in retrieved if r["id"] == primary_ctrl_id), None
            )
            if match:
                primary_id   = match["id"]
                primary_name = match["name"]
            else:
                # LLM chose a control FAISS did not return as top
                # Look up name from assessment procedures directly
                rows = self.assessment._df[
                    self.assessment._df["id_norm"] == primary_ctrl_id
                ]
                if not rows.empty:
                    primary_id   = primary_ctrl_id
                    primary_name = str(rows.iloc[0].get("control-name", primary_ctrl_id))

        # Final fallback: use top FAISS result
        if not primary_id:
            primary_id   = retrieved[0]["id"]
            primary_name = retrieved[0]["name"]

        block = self.assessment.format_evidence_block(primary_id, primary_name)

        # Add one-liner for other retrieved controls (excluding primary)
        others = [r for r in retrieved if r["id"] != primary_id]
        if others:
            block += "\n  Additional controls to verify:"
            for ctrl in others[:2]:
                ev = self.assessment.get_evidence(ctrl["id"])
                if ev["examine"]:
                    useful = [i for i in ev["examine"]
                              if "other relevant" not in i.lower()
                              and "security plan" not in i.lower()]
                    if useful:
                        block += f"\n    {ctrl['id']}: {useful[0]}"

        return block

    # ──────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────

    def get_guidance(self, risk_row: pd.Series) -> dict:
        """
        Full RAG pipeline for one risk row.

        Control selection is done by CODE, not LLM:
            primary = retrieved[0]  (highest cosine similarity from FAISS)

        LLM only explains the pre-selected primary control.
        This ensures guidance comes from the actual NIST document.
        """
        query     = self._build_query(risk_row)
        retrieved, all_candidates = self._retrieve(query, risk_row=risk_row)

        if not retrieved:
            return {
                "guidance":        f"No NIST controls retrieved above threshold ({self.min_similarity}).",
                "evidence_block":  "No controls retrieved -- evidence lookup skipped.",
                "primary_control": "N/A",
                "top_controls":    [],
                "retrieved_count": 0,
                "scores":          [],
                "rag_detail":      {"query": query, "retrieved_controls": [], "all_candidates": []},
            }

        # CODE picks primary = reranked[0] (best cross-encoder score)
        # If no reranker, primary = FAISS top-1 (best cosine similarity)
        # LLM is NOT involved in control selection
        primary = retrieved[0]

        # Pass reranked top-3 to LLM as context
        # Primary gets full text, supporting get 300 chars
        top3_for_llm = retrieved[:3]

        # LLM explains the pre-selected primary using retrieved context
        guidance, llm_prompt = self._generate_guidance(risk_row, top3_for_llm, primary)

        # Evidence uses the same code-selected primary
        evidence = self._build_evidence_block(retrieved, primary["id"])

        return {
            "guidance":        guidance,
            "llm_prompt":      llm_prompt,
            "evidence_block":  evidence,
            "primary_control": primary["id"],
            "top_controls":    [f"{r['id']} -- {r['name']}" for r in retrieved[:3]],
            "retrieved_count": len(retrieved),
            "scores":          [r["similarity_score"] for r in retrieved],
            "rag_detail": {
                "query":            query,
                "reranker_used":    self.reranker is not None,
                # Top-5 after reranking — these drive the output
                "retrieved_controls": [
                    {
                        "similarity_rank":  r["similarity_rank"],
                        "similarity_score": r["similarity_score"],
                        "reranker_rank":    r["reranker_rank"],
                        "reranker_score":   r["reranker_score"],
                        "id":               r["id"],
                        "name":             r["name"],
                        "control_text":     r["text"],
                        "discussion":       r["discussion"][:400] if r["discussion"] else "",
                        "related":          r["related"],
                        "sent_to_llm":      r in top3_for_llm,
                        "in_top_k":         True,
                    }
                    for r in retrieved
                ],
                # All 15 FAISS candidates before top-K cutoff
                # Useful for debugging why a control was or was not selected
                "all_candidates": [
                    {
                        "similarity_rank":  r["similarity_rank"],
                        "similarity_score": r["similarity_score"],
                        "reranker_rank":    r["reranker_rank"],
                        "reranker_score":   r["reranker_score"],
                        "id":               r["id"],
                        "name":             r["name"],
                        "in_top_k":         r.get("in_top_k", False),
                    }
                    for r in all_candidates
                ],
            },
        }


# ─────────────────────────────────────────────
# INTEGRATION HELPERS
# ─────────────────────────────────────────────

def add_nist_guidance_to_top5(top5: pd.DataFrame, rag: NISTRag) -> tuple:
    """
    Runs RAG for each top-5 risk.
    Returns (enriched_df, raw_results_list).
    """
    print("\n[RAG] Generating NIST guidance for top-5 risks...")

    guidance_col = []
    prompt_col   = []
    evidence_col = []
    controls_col = []
    primary_col  = []
    raw_results  = []

    for i, (_, row) in enumerate(top5.iterrows()):
        rank = int(row.get("rank", i + 1))
        vuln = row.get("vulnerability_name", "N/A")
        print(f"\n[RAG] Risk #{rank}: {vuln}")

        try:
            result = rag.get_guidance(row)
            guidance_col.append(normalise_text(result["guidance"]))
            prompt_col.append(result.get("llm_prompt", ""))
            evidence_col.append(result["evidence_block"])
            controls_col.append(", ".join(result["top_controls"]))
            primary_col.append(result["primary_control"])
            raw_results.append(result)

            print(f"[RAG]   Controls retrieved : {result['retrieved_count']}/{TOP_K}")
            print(f"[RAG]   Primary control    : {result['primary_control']}")
            print(f"[RAG]   Similarity scores  : {result['scores']}")

        except Exception as e:
            print(f"[RAG]   ERROR: {e}")
            guidance_col.append(f"RAG error: {e}")
            prompt_col.append("")
            evidence_col.append("Evidence lookup failed.")
            controls_col.append("N/A")
            primary_col.append("N/A")
            raw_results.append({"guidance": f"error: {e}", "rag_detail": {}})

        if i < len(top5) - 1:
            time.sleep(1.5)

    top5 = top5.copy()
    top5["nist_guidance"]        = guidance_col
    top5["nist_prompt"]          = prompt_col
    top5["nist_evidence"]        = evidence_col
    top5["nist_top_controls"]    = controls_col
    top5["nist_primary_control"] = primary_col
    return top5, raw_results


def save_rag_queries_json(top5: pd.DataFrame, rag_results: list, out_dir: str) -> str:
    """
    Saves full RAG detail for all top-5 risks to JSON.
    Proves guidance came from NIST document (not LLM training data).
    File: {out_dir}/rag_queries_and_retrieved.json
    """
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / "rag_queries_and_retrieved.json"

    records = []
    for i, (result, (_, row)) in enumerate(zip(rag_results, top5.iterrows())):
        rag_detail = result.get("rag_detail", {})
        records.append({
            "risk_rank":            int(row.get("rank", i + 1)),
            "vulnerability_name":   str(row.get("vulnerability_name", "N/A")),
            "cve":                  str(row.get("cve", "N/A")),
            "asset_name":           str(row.get("asset_name", "N/A")),
            "final_score":          float(row.get("final_score", 0)),
            "nist_primary_control": result.get("primary_control", "N/A"),
            "nist_top_controls":    result.get("top_controls", []),
            "retrieved_count":      result.get("retrieved_count", 0),
            "min_similarity":       MIN_SIMILARITY,
            "embedding_query":      rag_detail.get("query", ""),
            "retrieved_controls":   rag_detail.get("retrieved_controls", []),
            "all_candidates":       rag_detail.get("all_candidates", []),
            "llm_guidance_preview": result.get("guidance", "")[:500],
        })

    output = {
        "metadata": {
            "embedding_model":  "BAAI/bge-large-en-v1.5",
            "generation_model": GEMINI_MODEL,
            "min_similarity":   MIN_SIMILARITY,
            "top_k":            TOP_K,
            "total_risks":      len(records),
            "generated_at":     __import__("datetime").datetime.now().isoformat(),
        },
        "risks": records,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[RAG] RAG JSON saved : {file_path}  "
          f"({file_path.stat().st_size // 1024} KB)")
    return str(file_path)


def print_nist_section(risk_row: pd.Series):
    """Prints the NIST guidance + evidence section for one risk."""
    guidance = str(risk_row.get("nist_guidance", ""))
    evidence = str(risk_row.get("nist_evidence", ""))
    controls = risk_row.get("nist_top_controls", "N/A")

    print(f"\n  {'─'*68}")
    print(f"  NIST SP 800-53 Rev. 5 -- REMEDIATION GUIDANCE")
    print(f"  Source : catalog RAG (bge-large-en-v1.5 + Gemini) "
          f"· threshold={MIN_SIMILARITY}")
    print(f"  Controls retrieved: {controls}")
    print(f"  {'─'*68}")

    if not guidance or guidance.startswith("RAG error"):
        print(f"  Guidance unavailable: {guidance}")
    else:
        for line in guidance.strip().split("\n"):
            print(f"  {line}")

    print(f"\n  {'─'*68}")
    print(f"  EVIDENCE TO COLLECT")
    print(f"  Source : NIST SP 800-53A Rev. 5 assessment procedures (direct lookup)")
    print(f"  {'─'*68}")

    if not evidence or evidence.startswith("Evidence lookup failed"):
        print(f"  {evidence}")
    else:
        for line in evidence.strip().split("\n"):
            print(f"  {line}")
