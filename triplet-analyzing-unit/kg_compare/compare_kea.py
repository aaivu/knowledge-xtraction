from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set
import numpy as np

from sentence_transformers import SentenceTransformer

from .parse_utils import Triplet, normalize_triplet


@dataclass
class KEAConfig:
    # Single threshold for head/relation/tail
    cos_th: float = 0.76

    # If you want to mark GT triplets as "missing" when nothing is even remotely close
    # set this lower than cos_th. If you don't want missing/extra, keep it equal to cos_th.
    match_floor: float = 0.50

    model_name: str = "paraphrase-MiniLM-L6-v2"


def normalize_text(s: str) -> str:
    """Normalize text: lowercase, strip, collapse whitespace."""
    return " ".join(str(s).strip().lower().split())


def cosine_similarity_embeddings(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between normalized embedding vectors."""
    return float(np.dot(a, b))


def score_triplet_components(
    gh: np.ndarray, gr: np.ndarray, gt: np.ndarray,
    lh: np.ndarray, lr: np.ndarray, lt: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Score triplet by averaging cosine similarity of head, relation, tail.
    Returns (overall_score, head_cos, rel_cos, tail_cos)."""
    head_cos = cosine_similarity_embeddings(gh, lh)
    rel_cos = cosine_similarity_embeddings(gr, lr)
    tail_cos = cosine_similarity_embeddings(gt, lt)
    overall = (head_cos + rel_cos + tail_cos) / 3.0
    return overall, head_cos, rel_cos, tail_cos


def compare_kgs_kea(
    gold: List[Triplet],
    llm: List[Triplet],
    cfg: KEAConfig,
) -> Dict[str, Any]:
    """
    KEA-style part-wise cosine comparison.

    For each GT triplet:
      1) Find best LLM triplet by overall average cosine(head,rel,tail).
      2) If head>=th and rel>=th and tail>=th -> aligned
      3) Else if head>=th and tail>=th and rel<th -> relation_substitution
      4) Else if rel>=th and (head<th or tail<th) -> entity_substitution
      5) Else -> missing (only if best_overall < match_floor, otherwise it stays unclassified)

    For extra:
      LLM triplets that never got used in aligned/relation/entity matches.
    """

    gold = [normalize_triplet(x) for x in gold]
    llm = [normalize_triplet(x) for x in llm]

    out: Dict[str, Any] = {
        "aligned": [],
        "relation_substitution": [],
        "entity_substitution": [],
        "missing": [],
        "extra": [],
        "matches_scored": [],
    }

    if not gold and llm:
        out["extra"] = llm[:]  # all extra
        return out
    if not llm and gold:
        out["missing"] = gold[:]  # all missing
        return out
    if not gold or not llm:
        return out

    model = SentenceTransformer(cfg.model_name)

    # Pre-encode all triplet components for fast cosine matching
    g_heads = [normalize_text(h) for (h, _, _) in gold]
    g_rels  = [normalize_text(r) for (_, r, _) in gold]
    g_tails = [normalize_text(t) for (_, _, t) in gold]

    l_heads = [normalize_text(h) for (h, _, _) in llm]
    l_rels  = [normalize_text(r) for (_, r, _) in llm]
    l_tails = [normalize_text(t) for (_, _, t) in llm]

    gH = model.encode(g_heads, convert_to_numpy=True, normalize_embeddings=True)
    gR = model.encode(g_rels,  convert_to_numpy=True, normalize_embeddings=True)
    gT = model.encode(g_tails, convert_to_numpy=True, normalize_embeddings=True)

    lH = model.encode(l_heads, convert_to_numpy=True, normalize_embeddings=True)
    lR = model.encode(l_rels,  convert_to_numpy=True, normalize_embeddings=True)
    lT = model.encode(l_tails, convert_to_numpy=True, normalize_embeddings=True)

    used_llm: Set[int] = set()
    th = cfg.cos_th

    for i, gt_tri in enumerate(gold):
        best_j = -1
        best_overall = -1.0
        best_parts = (0.0, 0.0, 0.0)

        for j in range(len(llm)):
            overall, hc, rc, tc = score_triplet_components(gH[i], gR[i], gT[i], lH[j], lR[j], lT[j])
            if overall > best_overall:
                best_overall = overall
                best_j = j
                best_parts = (hc, rc, tc)

        best_llm = llm[best_j]
        hc, rc, tc = best_parts

        decision = "unclassified"
        if hc >= th and rc >= th and tc >= th:
            out["aligned"].append((gt_tri, best_llm, round(best_overall, 4), round(hc, 4), round(rc, 4), round(tc, 4)))
            used_llm.add(best_j)
            decision = "aligned"
        elif hc >= th and tc >= th and rc < th:
            out["relation_substitution"].append((gt_tri, best_llm, round(best_overall, 4), round(hc, 4), round(rc, 4), round(tc, 4)))
            used_llm.add(best_j)
            decision = "relation_substitution"
        elif rc >= th and (hc < th or tc < th):
            out["entity_substitution"].append((gt_tri, best_llm, round(best_overall, 4), round(hc, 4), round(rc, 4), round(tc, 4)))
            used_llm.add(best_j)
            decision = "entity_substitution"
        else:
            # Mark as missing only if nothing is close enough overall
            if best_overall < cfg.match_floor:
                out["missing"].append((gt_tri, best_llm, round(best_overall, 4), round(hc, 4), round(rc, 4), round(tc, 4)))
                decision = "missing"

        out["matches_scored"].append({
            "gold": gt_tri,
            "best_llm": best_llm,
            "overall": round(best_overall, 4),
            "head_cos": round(hc, 4),
            "rel_cos": round(rc, 4),
            "tail_cos": round(tc, 4),
            "cos_th": th,
            "decision": decision,
        })

    # Extra = LLM triplets not used by any match
    for j, ltri in enumerate(llm):
        if j not in used_llm:
            out["extra"].append(ltri)

    return out
