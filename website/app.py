import os
import re
import sys
import json
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# ── Resolve project paths so existing code can be imported without copying ──
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "graph-construction"))
sys.path.insert(0, str(ROOT / "src" / "KGX-Graph-Similarity"))
# KRPO Multi lives inside the kganalytica library
sys.path.insert(0, str(ROOT / "src" / "library-kganalytica" / "src"))

import extractor as _extractor                                       # noqa: E402
from llms.llm_factory import LLMFactorySelector                      # noqa: E402
from snea_sbert_similarity import snea_sbert_similarity              # noqa: E402
from kganalytica.kg_construction.krpo_multi import krpo_extract      # noqa: E402
from kganalytica.kg_construction.llms.llm_factory import (           # noqa: E402
    LLMFactorySelector as KRPOFactory,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate-kgs", methods=["POST"])
def generate_kgs():
    body = request.get_json(force=True, silent=True) or {}
    texts = body.get("texts", [])
    model = (body.get("model") or "llama-3.1-8b-instant").strip()
    api_key = (body.get("api_key") or "").strip()

    if len(texts) != 3 or not all(isinstance(t, str) and t.strip() for t in texts):
        return jsonify({"error": "Provide exactly 3 non-empty paragraphs."}), 400
    if not api_key:
        return jsonify({"error": "Groq API key is required. Open ⚙ Settings."}), 400

    os.environ["GROQ_API_KEY"] = api_key
    LLMFactorySelector.clear_cache()
    KRPOFactory.clear_cache()

    try:
        result = None

        # ── Primary: KRPO Multi (iterative NLI-grounded extraction) ──────────
        try:
            krpo_llm = KRPOFactory.get_factory(model, api_key=api_key)
            raw_krpo = krpo_extract(krpo_llm, texts, max_rounds=3)
            result = {}
            for i in range(1, len(texts) + 1):
                triples = raw_krpo.get(f"graph_{i}", [])
                validated = [
                    [str(e).replace("_", " ").lower() for e in t]
                    for t in triples
                    if isinstance(t, (list, tuple)) and len(t) == 3
                    and all(isinstance(e, str) for e in t)
                ]
                result[i] = validated
            if not any(result.values()):
                log.warning("KRPO Multi returned no triplets — falling back")
                result = None
            else:
                log.info("KRPO Multi succeeded: " +
                         " | ".join(f"graph_{i}: {len(result[i])} triplets" for i in range(1, 4)))
        except Exception as krpo_exc:
            log.warning(f"KRPO Multi failed ({krpo_exc}) — falling back")
            result = None

        # ── Fallback: single-prompt few-shot joint extraction ─────────────────
        if result is None:
            llm = LLMFactorySelector.get_factory(model)
            result = _extractor.extract(texts, llm)

        return jsonify({"kgs": {str(k): v for k, v in result.items()}})
    except Exception as exc:
        log.exception("KG generation failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/s3kg", methods=["POST"])
def s3kg():
    body = request.get_json(force=True, silent=True) or {}
    kg1 = body.get("kg1", [])
    kg2 = body.get("kg2", [])
    kg3 = body.get("kg3", [])
    alpha = max(0.0, min(1.0, float(body.get("alpha", 0.5))))

    if not (kg1 and kg2 and kg3):
        return jsonify({"error": "All three KGs must be non-empty."}), 400

    try:
        gold_res = snea_sbert_similarity(kg1, kg2, alpha=alpha, return_details=True)
        ctx_res  = snea_sbert_similarity(kg1, kg3, alpha=alpha, return_details=True)
        gold_sim = gold_res["score"]
        ctx_sim  = ctx_res["score"]
        denom = gold_sim + ctx_sim
        cus = round((2 * gold_sim * ctx_sim / denom) if denom > 0 else 0.0, 4)
        return jsonify({
            "gold_sim": gold_sim,
            "ctx_sim":  ctx_sim,
            "cus":      cus,
            "alpha":    alpha,
            "analysis": {"gold": gold_res, "ctx": ctx_res},
        })
    except Exception as exc:
        log.exception("S3KG scoring failed")
        return jsonify({"error": str(exc)}), 500


# ── Helpers for KGA ──────────────────────────────────────────────────────────

def _categorize_triplets(kg_ref, analysis):
    """Bucket kg1 triplets (from matched_triples) into 5 quality categories.

    Returns a dict with keys: aligned, entity_wrong, relation_wrong, extra, missing.
    """
    matched = analysis.get("matched_triples", [])
    aligned, entity_wrong, relation_wrong, extra = [], [], [], []
    matched_kg2_set = set()

    for m in matched:
        k1, k2, sim = m["kg1"], m["kg2"], m["sim"]
        matched_kg2_set.add(tuple(k2))

        if sim >= 0.70:
            aligned.append({"kg1": k1, "kg2": k2, "sim": sim})
        elif sim < 0.45:
            extra.append({"triplet": k1, "best_match": k2, "sim": sim})
        else:
            # Mid-range: determine if entity or relation mismatch dominates
            k1_ents = set((k1[0] + " " + k1[2]).lower().split())
            k2_ents = set((k2[0] + " " + k2[2]).lower().split())
            k1_rel  = set(k1[1].lower().split())
            k2_rel  = set(k2[1].lower().split())
            ent_j = len(k1_ents & k2_ents) / (len(k1_ents | k2_ents) + 1e-8)
            rel_j = len(k1_rel  & k2_rel)  / (len(k1_rel  | k2_rel)  + 1e-8)
            if rel_j < ent_j:
                relation_wrong.append({"kg1": k1, "kg2": k2, "sim": sim})
            else:
                entity_wrong.append({"kg1": k1, "kg2": k2, "sim": sim})

    missing = [list(t) for t in kg_ref if tuple(t) not in matched_kg2_set]
    return {
        "aligned":        aligned,
        "entity_wrong":   entity_wrong,
        "relation_wrong": relation_wrong,
        "extra":          extra,
        "missing":        missing,
    }


def _build_kga_prompt(cats, ref_label):
    def fmt_pairs(lst):
        if not lst:
            return "  (none)"
        return "\n".join(
            f"  {i+1}. LLM: {t['kg1']}  <->  {ref_label}: {t['kg2']}  (sim={t['sim']})"
            for i, t in enumerate(lst)
        )

    def fmt_singles(lst):
        if not lst:
            return "  (none)"
        rows = []
        for i, t in enumerate(lst):
            tr = t["triplet"] if isinstance(t, dict) else t
            rows.append(f"  {i+1}. {tr}")
        return "\n".join(rows)

    return f"""You are a Knowledge Graph Analytica expert. \
An LLM-generated knowledge graph (KG_LLM) has been compared against {ref_label}. \
Triplets are [Subject, Relation, Object].

ALIGNED (high similarity -- correctly captured facts):
{fmt_pairs(cats['aligned'])}

ENTITY WRONG (relation matches but entity names differ):
{fmt_pairs(cats['entity_wrong'])}

RELATION WRONG (entities match but relation predicate differs):
{fmt_pairs(cats['relation_wrong'])}

EXTRA TRIPLETS (in KG_LLM but absent from {ref_label} -- may be hallucinated or extra):
{fmt_singles(cats['extra'])}

MISSING TRIPLETS (in {ref_label} but not captured by KG_LLM):
{fmt_singles(cats['missing'])}

Respond ONLY with a valid JSON object (no markdown fences, no extra text) with exactly these keys:
{{
  "aligned_desc": "2-3 sentence analysis of what the LLM correctly captured",
  "entity_wrong_desc": "2-3 sentence analysis of entity-level errors",
  "relation_wrong_desc": "2-3 sentence analysis of relation predicate errors",
  "extra_desc": "2-3 sentence analysis of extra or possibly hallucinated facts",
  "missing_desc": "2-3 sentence analysis of what important facts are missing",
  "overall": "2-3 sentence overall quality assessment of the LLM response"
}}"""


def _safe_parse_json(raw):
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[^\n]*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw.rstrip())
    try:
        return json.loads(raw)
    except Exception:
        return {
            "aligned_desc": raw,
            "entity_wrong_desc": "",
            "relation_wrong_desc": "",
            "extra_desc": "",
            "missing_desc": "",
            "overall": "",
        }


@app.route("/api/kga", methods=["POST"])
def kga():
    body          = request.get_json(force=True, silent=True) or {}
    kg1           = body.get("kg1", [])
    kg2           = body.get("kg2", [])          # gold
    kg3           = body.get("kg3", [])          # ctx
    gold_analysis = body.get("gold_analysis", {})
    ctx_analysis  = body.get("ctx_analysis",  {})
    api_key       = (body.get("api_key") or "").strip()
    model         = (body.get("model") or "llama-3.1-8b-instant").strip()

    if not api_key:
        return jsonify({"error": "Groq API key is required."}), 400
    if not (kg1 and kg2 and kg3 and gold_analysis and ctx_analysis):
        return jsonify({"error": "Missing kg1/kg2/kg3 or analysis data."}), 400

    os.environ["GROQ_API_KEY"] = api_key
    LLMFactorySelector.clear_cache()

    try:
        llm = LLMFactorySelector.get_factory(model)
        gold_cats = _categorize_triplets(kg2, gold_analysis)
        ctx_cats  = _categorize_triplets(kg3, ctx_analysis)

        gold_raw = llm.get_answer(_build_kga_prompt(gold_cats, "the Gold Answer"))
        ctx_raw  = llm.get_answer(_build_kga_prompt(ctx_cats,  "the Supporting Context"))

        return jsonify({
            "gold": {**gold_cats, "descriptions": _safe_parse_json(gold_raw)},
            "ctx":  {**ctx_cats,  "descriptions": _safe_parse_json(ctx_raw)},
        })
    except Exception as exc:
        log.exception("KGA analysis failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
