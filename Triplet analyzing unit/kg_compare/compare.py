from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import difflib
import numpy as np

from sentence_transformers import SentenceTransformer

from .parse_utils import Triplet, normalize_triplet


@dataclass
class CompareConfig:
    # Stage 1: whole-triplet similarity for "aligned"
    aligned_threshold: float = 0.76

    # Stage 2: cosine similarity thresholds (only used when NOT aligned)
    entity_cos_th: float = 0.8
    relation_cos_th: float = 0.7

    # Embedding model
    model_name: str = "paraphrase-MiniLM-L6-v2"


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _triplet_text(tri: Triplet) -> str:
    h, r, t = tri
    return f"{_norm(h)} | {_norm(r)} | {_norm(t)}"


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are normalized embeddings (unit vectors)
    return float(np.dot(a, b))


def compare_kgs_three_class(
    gold: List[Triplet],
    llm: List[Triplet],
    cfg: CompareConfig,
) -> Dict[str, Any]:
    """
    3-class comparison:
      1) aligned (whole-triplet string similarity >= cfg.aligned_threshold)
      2) relation_different (entities similar, relation not)
      3) entity_different (relation similar, entity not; also used as fallback)

    How "entity vs relation different" is decided:
      - Compute cosine sim of head, relation, tail using SentenceTransformer embeddings.
      - Entities are "ok" if head_cos>=entity_th AND tail_cos>=entity_th
      - Relation is "ok" if rel_cos>=relation_th
      - If entities ok and relation not -> relation_different
      - Else -> entity_different
    """

    gold = [normalize_triplet(x) for x in gold]
    llm = [normalize_triplet(x) for x in llm]

    result: Dict[str, Any] = {
        "aligned": [],
        "relation_different": [],
        "entity_different": [],
        "unclassified": [],
        "matches_scored": [],
    }

    if not gold or not llm:
        return result

    # ---------- Stage 1 prep ----------
    llm_texts = [_triplet_text(x) for x in llm]

    # ---------- Stage 2 prep (embeddings) ----------
    model = SentenceTransformer(cfg.model_name)

    # Embed all LLM parts once (faster)
    llm_heads = [_norm(h) for (h, _, _) in llm]
    llm_rels  = [_norm(r) for (_, r, _) in llm]
    llm_tails = [_norm(t) for (_, _, t) in llm]

    llm_h_emb = model.encode(llm_heads, convert_to_numpy=True, normalize_embeddings=True)
    llm_r_emb = model.encode(llm_rels,  convert_to_numpy=True, normalize_embeddings=True)
    llm_t_emb = model.encode(llm_tails, convert_to_numpy=True, normalize_embeddings=True)

    for gt in gold:
        gt_text = _triplet_text(gt)

        # ----- Stage 1: aligned by whole-triplet similarity -----
        sims = [_similarity(gt_text, lt) for lt in llm_texts]
        best_i = int(max(range(len(sims)), key=lambda i: sims[i]))
        best_sim = float(sims[best_i])
        best_llm = llm[best_i]

        if best_sim >= cfg.aligned_threshold:
            result["aligned"].append((gt, best_llm, round(best_sim, 4)))
            result["matches_scored"].append({
                "gold": gt,
                "llm": best_llm,
                "stage": "aligned_by_whole_triplet",
                "whole_triplet_similarity": round(best_sim, 4),
            })
            continue

        # ----- Stage 2: cosine similarity on parts -----
        gh, gr, gt_tail = map(_norm, gt)
        gt_h_emb = model.encode([gh], convert_to_numpy=True, normalize_embeddings=True)[0]
        gt_r_emb = model.encode([gr], convert_to_numpy=True, normalize_embeddings=True)[0]
        gt_t_emb = model.encode([gt_tail], convert_to_numpy=True, normalize_embeddings=True)[0]

        h_cos = _cos(gt_h_emb, llm_h_emb[best_i])
        r_cos = _cos(gt_r_emb, llm_r_emb[best_i])
        t_cos = _cos(gt_t_emb, llm_t_emb[best_i])

        entities_ok = (h_cos >= cfg.entity_cos_th) and (t_cos >= cfg.entity_cos_th)
        relation_ok = (r_cos >= cfg.relation_cos_th)
        head_relation_ok = (h_cos >= cfg.entity_cos_th) and (r_cos >= cfg.relation_cos_th)
        tail_relation_ok = (t_cos >= cfg.entity_cos_th) and (r_cos >= cfg.relation_cos_th)

        # Class decision for the "not aligned" cases
        decision = "unclassified"

        if entities_ok and (not relation_ok):
            result["relation_different"].append((gt, best_llm, round(best_sim, 4), round(h_cos, 4), round(r_cos, 4), round(t_cos, 4)))
            decision = "relation_different"
        elif head_relation_ok or tail_relation_ok:
            result["entity_different"].append((gt, best_llm, round(best_sim, 4), round(h_cos, 4), round(r_cos, 4), round(t_cos, 4)))
            decision = "entity_different"
        else:
            result["unclassified"].append((gt, best_llm, round(best_sim, 4), round(h_cos, 4), round(r_cos, 4), round(t_cos, 4)))

        result["matches_scored"].append({
            "gold": gt,
            "llm": best_llm,
            "stage": "cosine_on_parts",
            "whole_triplet_similarity": round(best_sim, 4),
            "head_cos": round(h_cos, 4),
            "rel_cos": round(r_cos, 4),
            "tail_cos": round(t_cos, 4),
            "decision": decision,
        })

    return result
