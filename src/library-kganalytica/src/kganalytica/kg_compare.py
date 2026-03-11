from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import difflib
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# Type definitions
# =========================================================

Triplet = Tuple[str, str, str]


# =========================================================
# Normalization utilities
# =========================================================

def _norm(text: str) -> str:
    """
    Normalize string by lowercasing, stripping, and collapsing spaces.
    """
    return " ".join(str(text).strip().lower().split())


def normalize_triplet(triplet: Triplet) -> Triplet:
    """
    Normalize a triplet (head, relation, tail).
    """
    if len(triplet) != 3:
        raise ValueError(f"Triplet must have exactly 3 items. Got: {triplet}")
    h, r, t = triplet
    return (_norm(h), _norm(r), _norm(t))


def normalize_triplets(triplets: List[Triplet]) -> List[Triplet]:
    """
    Normalize a list of triplets.
    """
    return [normalize_triplet(tri) for tri in triplets]


def triplet_to_text(tri: Triplet) -> str:
    """
    Convert triplet into a single comparable string.
    """
    h, r, t = tri
    return f"{_norm(h)} | {_norm(r)} | {_norm(t)}"


# =========================================================
# Similarity helpers
# =========================================================

def string_similarity(a: str, b: str) -> float:
    """
    Sequence similarity between two strings.
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity for normalized vectors.
    """
    return float(np.dot(a, b))


# =========================================================
# Config
# =========================================================

@dataclass
class CompareConfig:
    """
    Configuration for KG triplet comparison.
    """
    aligned_threshold: float = 0.76
    entity_cos_th: float = 0.80
    relation_cos_th: float = 0.70
    model_name: str = "paraphrase-MiniLM-L6-v2"


# =========================================================
# Embedding model cache
# =========================================================

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache the embedding model.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


# =========================================================
# Core comparison logic
# =========================================================

def compare_kgs_three_class(
    gold: List[Triplet],
    llm: List[Triplet],
    cfg: Optional[CompareConfig] = None,
) -> Dict[str, Any]:
    """
    Compare two KGs at triplet level.

    Output classes:
      1. aligned
      2. relation_different
      3. entity_different

    Decision logic:
      - Find the best matching LLM triplet for each gold triplet
      - If whole-triplet similarity >= aligned_threshold -> aligned
      - Else compare head, relation, tail using embeddings
      - If entities are similar but relation differs -> relation_different
      - If relation and one entity are similar but another entity differs -> entity_different
      - Else -> unclassified
    """
    if cfg is None:
        cfg = CompareConfig()

    gold = normalize_triplets(gold)
    llm = normalize_triplets(llm)

    result: Dict[str, Any] = {
        "aligned": [],
        "relation_different": [],
        "entity_different": [],
        "unclassified": [],
        "matches_scored": [],
    }

    if not gold:
        return result

    if not llm:
        for gt in gold:
            result["unclassified"].append(
                {
                    "gold": gt,
                    "llm": None,
                    "whole_triplet_similarity": 0.0,
                    "head_cos": None,
                    "rel_cos": None,
                    "tail_cos": None,
                    "decision": "no_llm_triplets",
                }
            )
        return result

    model = get_embedding_model(cfg.model_name)

    llm_texts = [triplet_to_text(x) for x in llm]
    llm_heads = [_norm(h) for (h, _, _) in llm]
    llm_rels = [_norm(r) for (_, r, _) in llm]
    llm_tails = [_norm(t) for (_, _, t) in llm]

    llm_h_emb = model.encode(llm_heads, convert_to_numpy=True, normalize_embeddings=True)
    llm_r_emb = model.encode(llm_rels, convert_to_numpy=True, normalize_embeddings=True)
    llm_t_emb = model.encode(llm_tails, convert_to_numpy=True, normalize_embeddings=True)

    for gt in gold:
        gt_text = triplet_to_text(gt)

        sims = [string_similarity(gt_text, lt) for lt in llm_texts]
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_llm = llm[best_i]

        if best_sim >= cfg.aligned_threshold:
            item = {
                "gold": gt,
                "llm": best_llm,
                "whole_triplet_similarity": round(best_sim, 4),
                "decision": "aligned",
            }
            result["aligned"].append(item)
            result["matches_scored"].append(item)
            continue

        gh, gr, gt_tail = map(_norm, gt)
        gt_h_emb = model.encode([gh], convert_to_numpy=True, normalize_embeddings=True)[0]
        gt_r_emb = model.encode([gr], convert_to_numpy=True, normalize_embeddings=True)[0]
        gt_t_emb = model.encode([gt_tail], convert_to_numpy=True, normalize_embeddings=True)[0]

        h_cos = cosine_similarity(gt_h_emb, llm_h_emb[best_i])
        r_cos = cosine_similarity(gt_r_emb, llm_r_emb[best_i])
        t_cos = cosine_similarity(gt_t_emb, llm_t_emb[best_i])

        entities_ok = (h_cos >= cfg.entity_cos_th) and (t_cos >= cfg.entity_cos_th)
        relation_ok = r_cos >= cfg.relation_cos_th
        head_relation_ok = (h_cos >= cfg.entity_cos_th) and (r_cos >= cfg.relation_cos_th)
        tail_relation_ok = (t_cos >= cfg.entity_cos_th) and (r_cos >= cfg.relation_cos_th)

        scored_item = {
            "gold": gt,
            "llm": best_llm,
            "whole_triplet_similarity": round(best_sim, 4),
            "head_cos": round(h_cos, 4),
            "rel_cos": round(r_cos, 4),
            "tail_cos": round(t_cos, 4),
            "decision": None,
        }

        if entities_ok and (not relation_ok):
            scored_item["decision"] = "relation_different"
            result["relation_different"].append(scored_item)
        elif head_relation_ok or tail_relation_ok:
            scored_item["decision"] = "entity_different"
            result["entity_different"].append(scored_item)
        else:
            scored_item["decision"] = "unclassified"
            result["unclassified"].append(scored_item)

        result["matches_scored"].append(scored_item)

    return result


# =========================================================
# Public library functions
# =========================================================

def get_aligned_triplets(
    gold: List[Triplet],
    llm: List[Triplet],
    cfg: Optional[CompareConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Return aligned triplets between gold KG and LLM KG.
    """
    return compare_kgs_three_class(gold, llm, cfg)["aligned"]


def get_entity_wrong_triplets(
    gold: List[Triplet],
    llm: List[Triplet],
    cfg: Optional[CompareConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Return triplets where relation is close enough but entity differs.
    """
    return compare_kgs_three_class(gold, llm, cfg)["entity_different"]


def get_relation_wrong_triplets(
    gold: List[Triplet],
    llm: List[Triplet],
    cfg: Optional[CompareConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Return triplets where entities are close enough but relation differs.
    """
    return compare_kgs_three_class(gold, llm, cfg)["relation_different"]


# =========================================================
# Summary function
# =========================================================

def compare_two_kgs_full(
    gold: List[Triplet],
    llm: List[Triplet],
    cfg: Optional[CompareConfig] = None,
    use_soft_missing_extra: bool = False,
) -> Dict[str, Any]:
    """
    Full comparison summary between two KGs.
    """
    if cfg is None:
        cfg = CompareConfig()

    classified = compare_kgs_three_class(gold, llm, cfg)


    return {
        "aligned_triplets": classified["aligned"],
        "entity_wrong_triplets": classified["entity_different"],
        "relation_wrong_triplets": classified["relation_different"],
        "unclassified_triplets": classified["unclassified"],
        "matches_scored": classified["matches_scored"],
        "counts": {
            "aligned": len(classified["aligned"]),
            "entity_wrong": len(classified["entity_different"]),
            "relation_wrong": len(classified["relation_different"]),
        }
    }


