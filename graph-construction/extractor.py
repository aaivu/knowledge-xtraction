import re
import json
import time
import logging

log = logging.getLogger(__name__)

_PLACEHOLDER_TOKENS = {"entity1", "entity2", "entity3", "entity4", "relation"}

_ROOTS_T1 = (
    'A&E Networks will simulcast the original "Roots" in 2016. The original "Roots" premiered in 1977 '
    "and ran for four seasons. The miniseries followed Kunta Kinte, a free black man in Virginia, as he "
    "was sold into slavery."
)
_ROOTS_T2 = (
    '(CNN)One of the biggest TV events of all time is being reimagined for new audiences. "Roots," the '
    "epic miniseries about an African-American slave and his descendants, had a staggering audience of "
    "over 100 million viewers back in 1977"
)
_ISIS_T1 = (
    "ISIS released more than 200 Yazidis, a minority group, a group says. The Islamist terror group has "
    "been killed in recent summer. ISIS released scores of other Yazidis, mainly children and the elderly. "
    "The Peshmerga commander says the freed Yazidis are released."
)
_ISIS_T2 = (
    "(CNN) ISIS on Wednesday released more than 200 Yazidis, a minority group whose members were killed, "
    "captured and displaced when the Islamist terror organization overtook their towns in northern Iraq "
    "last summer, officials said. Most of those released were women and children; the rest were ill or "
    "elderly, said Rassol Omar, a commander in the Peshmerga force that defends northern Iraq's "
    "semi-autonomous Kurdish region. Omar didn't say what led to the release, other than asserting that "
    "Arab tribal leaders helped to coordinate it. The freed Yazidis were received by Peshmerga, who sent "
    "them to the Kurdish regional capital, Irbil, said Nuri Osman, an official with Iraq's Kurdistan "
    "Regional Government. It wasn't immediately clear what motivated Wednesday's release, Osman said."
)
_ROOTS_T3 = (
    "Roots is a 1977 American television miniseries based on Alex Haley's novel. It depicts the life of "
    "Kunta Kinte, an African who was captured and sold into slavery in America. The series was broadcast "
    "on ABC and became one of the most watched programs in US television history."
)
_ISIS_T3 = (
    "The Yazidis are a Kurdish-speaking minority group native to northern Iraq. They follow a religion "
    "that combines elements of several ancient traditions. In August 2014, ISIS launched a major offensive "
    "against Yazidi communities in the Sinjar region, killing hundreds and enslaving thousands. The attack "
    "prompted international condemnation and led to US airstrikes against ISIS positions in northern Iraq."
)

_PROMPT = (
    "You are an expert at creating knowledge graphs based on text.\n"
    "You will receive multiple pieces of text, and you must perform the following steps on each:\n"
    "1. Entity detection: Extract ALL entities comprehensively. Include all named entities, important concepts, objects, and properties mentioned. Also extract attributes, quantities, dates, locations, and roles. Do NOT skip supporting or background details.\n"
    "2. Coreference resolution: Replace ALL pronouns (he, she, it, they, his, her, its) with the actual entity name. Use the same entity label for the same concept across all texts.\n"
    "3. Relation extraction: Identify semantic relationships as simple, concise phrases. Split compound sentences into as many triplets as needed — one triplet per fact.\n"
    "4. Knowledge Graph refinement: Where the same entity or relation appears across multiple graphs, use the same label consistently. Do not merge distinct facts — preserve each triplet independently.\n\n"
    "Format your response as a JSON object. Do not include any text outside the JSON.\n"
    'Each knowledge graph is a list of triples: [["subject", "relation", "object"], ...].\n\n'
    "EXAMPLE 1:\n"
    f"TEXT1 (reference answer):\n{_ROOTS_T1}\n\nTEXT2 (model-generated response):\n{_ROOTS_T2}\n\nTEXT3 (supporting context):\n{_ROOTS_T3}\n\n"
    "YOUR OUTPUT:\n"
    "{\n"
    '  "knowledge_graph1": [\n'
    '      ["A&E Networks", "will simulcast in 2016", "Roots"],\n'
    '      ["Roots", "premiered in", "1977"],\n'
    '      ["Roots", "ran for", "four seasons"],\n'
    '      ["Roots", "instance of", "miniseries"],\n'
    '      ["Roots", "followed", "Kunta Kinte"],\n'
    '      ["Kunta Kinte", "was sold into", "slavery"],\n'
    '      ["Kunta Kinte", "was a", "free black man"]\n'
    "  ],\n"
    '  "knowledge_graph2": [\n'
    '      ["Roots", "one of the", "biggest TV events of all time"],\n'
    '      ["Roots", "had a staggering audience of", "over 100 million viewers"],\n'
    '      ["Roots", "being", "reimagined for new audiences"],\n'
    '      ["Roots", "was about", "an African-American slave and his descendants"],\n'
    '      ["Roots", "premiered", "1977"]\n'
    "  ],\n"
    '  "knowledge_graph3": [\n'
    '      ["Roots", "is a", "American television miniseries"],\n'
    '      ["Roots", "based on", "Alex Haley\'s novel"],\n'
    '      ["Roots", "depicts life of", "Kunta Kinte"],\n'
    '      ["Kunta Kinte", "was captured and sold into", "slavery"],\n'
    '      ["Roots", "was broadcast on", "ABC"],\n'
    '      ["Roots", "premiered in", "1977"],\n'
    '      ["Roots", "one of the most watched", "US television programs"]\n'
    "  ]\n"
    "}\n\n"
    "EXAMPLE 2:\n"
    f"TEXT1 (reference answer):\n{_ISIS_T1}\n\nTEXT2 (model-generated response):\n{_ISIS_T2}\n\nTEXT3 (supporting context):\n{_ISIS_T3}\n\n"
    "YOUR OUTPUT:\n"
    "{\n"
    '  "knowledge_graph1": [\n'
    '      ["ISIS", "released", "more than 200 Yazidis"],\n'
    '      ["Yazidis", "are", "minority group"],\n'
    '      ["ISIS", "released", "children and elderly Yazidis"],\n'
    '      ["Peshmerga commander", "said", "freed Yazidis are released"]\n'
    "  ],\n"
    '  "knowledge_graph2": [\n'
    '      ["ISIS", "released", "more than 200 Yazidis"],\n'
    '      ["Yazidis", "are", "minority group"],\n'
    '      ["Yazidis", "killed and displaced by", "ISIS"],\n'
    '      ["ISIS", "released", "children and elderly Yazidis"],\n'
    '      ["Peshmerga commander", "said", "freed Yazidis are released"],\n'
    '      ["Peshmerga", "received", "freed Yazidis"],\n'
    '      ["Peshmerga", "sent freed Yazidis to", "Irbil"],\n'
    '      ["Arab tribal leaders", "helped coordinate", "release of Yazidis"]\n'
    "  ],\n"
    '  "knowledge_graph3": [\n'
    '      ["Yazidis", "are", "Kurdish-speaking minority group"],\n'
    '      ["Yazidis", "native to", "northern Iraq"],\n'
    '      ["ISIS", "launched offensive against", "Yazidi communities"],\n'
    '      ["ISIS", "offensive in", "Sinjar region"],\n'
    '      ["ISIS", "killed hundreds of", "Yazidis"],\n'
    '      ["ISIS", "enslaved thousands of", "Yazidis"],\n'
    '      ["attack", "prompted", "international condemnation"],\n'
    '      ["attack", "led to", "US airstrikes against ISIS"]\n'
    "  ]\n"
    "}\n\n"
)



def _validate(triplets: list) -> list:
    # drop placeholder entries and anything that is not a 3-string list
    valid = []
    for t in triplets:
        if isinstance(t, (list, tuple)) and len(t) == 3 and all(isinstance(e, str) for e in t):
            if t[0].lower().strip() not in _PLACEHOLDER_TOKENS and t[2].lower().strip() not in _PLACEHOLDER_TOKENS:
                valid.append(list(t))
    return valid


def _normalize(label: str) -> str:
    s = label.replace("_", " ").lower()
    return re.sub(r" {2,}", " ", s).strip()


def extract(paragraphs: list, llm) -> dict:
    # Single-prompt few-shot extraction. Returns {1: [[s,r,o],...], 2: ...}
    if not paragraphs:
        return {}

    n = len(paragraphs)
    graph_keys = ", ".join(f"knowledge_graph{i + 1}" for i in range(n))
    instruction = (
        f"Now extract knowledge graphs for the following {n} text(s). "
        f"Return a JSON object with exactly {n} key(s): {graph_keys}. "
        "Each value is a list of [subject, relation, object] triples.\n\n"
    )
    text_block = "\n\n".join(f"TEXT{i + 1}:\n{p}" for i, p in enumerate(paragraphs))
    prompt = _PROMPT + instruction + text_block

    data = {}
    for attempt in range(3):
        raw = llm.get_answer(prompt)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            data = {}
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        graphs = {i + 1: _validate(data.get(f"knowledge_graph{i + 1}", [])) for i in range(len(paragraphs))}
        if any(v for v in graphs.values()):
            break
        log.warning(f"Attempt {attempt + 1}/3 — empty result, retrying...")
        time.sleep(1)

    graphs = {i + 1: _validate(data.get(f"knowledge_graph{i + 1}", [])) for i in range(len(paragraphs))}
    for i, triplets in graphs.items():
        log.info(f"graph_{i}: {len(triplets)} triplets")
    return {k: [[_normalize(e) for e in t] for t in v] for k, v in graphs.items()}