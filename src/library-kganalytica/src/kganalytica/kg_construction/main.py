"""Knowledge graph construction API with KRPOMultiAlign support."""

import logging
from .krpo_multi import krpo_extract
from .llms.llm_factory import LLMFactorySelector

log = logging.getLogger(__name__)


def construct_graph(
    paragraph: str,
    model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
    max_rounds: int = 3,
) -> list:
    llm = LLMFactorySelector.get_factory(model, api_key=api_key)
    result = krpo_extract(llm, [paragraph], max_rounds=max_rounds)
    return result.get("graph_1", [])


def construct_graphs(
    paragraphs: list,
    model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
    max_rounds: int = 3,
) -> dict:
    if not paragraphs:
        return {}
    llm = LLMFactorySelector.get_factory(model, api_key=api_key)
    return krpo_extract(llm, paragraphs, max_rounds=max_rounds)


def construct_kgs(
    rows: list,
    model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
    max_rounds: int = 3,
) -> list:
    if not rows:
        return []
    llm = LLMFactorySelector.get_factory(model, api_key=api_key)
    results = []
    for row in rows:
        row_id = row.get("id", "")
        paragraphs = []
        i = 1
        while True:
            p = row.get(f"paragraph_{i}")
            if p is None:
                break
            if p and str(p).strip():
                paragraphs.append(str(p).strip())
            i += 1
        if not paragraphs:
            log.warning(f"Row {row_id}: no paragraphs found")
            continue
        log.info(f"Row {row_id}: extracting {len(paragraphs)} graph(s)")
        graphs = krpo_extract(llm, paragraphs, max_rounds=max_rounds, row_id=str(row_id))
        out = {"id": row_id}
        for k, v in graphs.items():
            out[f"kg_{k.split('_')[1]}"] = v
        results.append(out)
    return results
