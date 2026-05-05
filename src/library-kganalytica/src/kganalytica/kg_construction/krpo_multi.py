"""
KRPOMultiAlign ΓÇö simultaneous N-graph extraction with KRPO feedback loop.

One LLM call extracts all N graphs jointly from N paragraphs with label alignment.
Then iterative KRPO loop: tripletΓåÆtext (NLI grounding) ΓåÆ score each graph against its
own paragraph ΓåÆ combined = avg(scores) ΓåÆ backward_eval ΓåÆ backward_pred ΓåÆ optimizer ΓåÆ
accept trial prompt if better. Early stop at combined ΓëÑ 0.85.
"""

import ast
import json
import logging
import re
import time

log = logging.getLogger(__name__)

# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# Embedded templates
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

_TRIPLET2TEXT = (
    "You are given a factual triplet in the form (subject, relation, object).\n\n"
    "Your task is to convert the triplet into a natural language sentence, ensuring the following:\n\n"
    "- Do not add any information not present in the triplet.\n"
    "- Retain the semantics of the relation as much as possible.\n"
    "- The sentence should accurately reflect the meaning of the triplet.\n"
    "- Just need to give one sentence.\n\n\n"
    "Input triplet: {input_triplet}\n"
    "Output:"
)

_NLI_EXAMPLES = (
    "### Example 1\n\n"
    "Premise:\n\"\"\"\nAlice submitted her application before the deadline and received an offer "
    "for the software engineering position two weeks later.\n\"\"\"\n\n"
    "Hypothesis:\n\"\"\"\nAlice was offered the job after submitting her application.\n\"\"\"\n\n"
    "Output:\n"
    "{\n"
    '  "label": "entailment",\n'
    '  "confidence": 0.95,\n'
    '  "reasoning": "The premise clearly states that Alice submitted the application and received '
    'an offer afterward, which directly supports the hypothesis."\n'
    "}\n\n\n"
    "### Example 2\n\n"
    "Premise:\n\"\"\"\nThe report stated that the earthquake occurred in 2010, causing massive "
    "destruction across the region.\n\"\"\"\n\n"
    "Hypothesis:\n\"\"\"\nThe earthquake took place in 2015.\n\"\"\"\n\n"
    "Output:\n"
    "{\n"
    '  "label": "contradiction",\n'
    '  "confidence": 0.92,\n'
    '  "reasoning": "The premise explicitly mentions 2010 as the year of the earthquake, which '
    'directly contradicts the hypothesis claiming 2015."\n'
    "}\n\n\n"
    "### Example 3\n\n"
    "Premise:\n\"\"\"\nDr. Kumar has published several papers on machine learning, with a recent "
    "focus on reinforcement learning applications.\n\"\"\"\n\n"
    "Hypothesis:\n\"\"\"\nDr. Kumar prefers teaching over research.\n\"\"\"\n\n"
    "Output:\n"
    "{\n"
    '  "label": "neutral",\n'
    '  "confidence": 0.87,\n'
    '  "reasoning": "The premise discusses Dr. Kumar\'s research focus but does not provide any '
    'information about his teaching preferences."\n'
    "}\n\n"
)

_NLI_TMPL = (
    "You are an expert in Natural Language Inference (NLI). Your task is to determine the logical "
    "relationship between a given *premise* (a long text) and a *hypothesis* (a short text).\n\n"
    "There are three possible labels:\n"
    '1. "entailment" \u2013 The hypothesis logically follows from the premise.\n'
    '2. "contradiction" \u2013 The hypothesis is logically inconsistent with the premise.\n'
    '3. "neutral" \u2013 The hypothesis is neither entailed nor contradicted by the premise.\n\n'
    "Note: All double quotes in the reasoning should be escaped with backslashes.\n\n"
    "Return the result strictly in the following JSON format:\n"
    "{{\n"
    '  "label": "entailment" | "contradiction" | "neutral",\n'
    '  "confidence": float between 0 and 1,\n'
    '  "reasoning": "a brief explanation of your decision"\n'
    "}}\n\n"
    "{few_shot_examples}\n"
    "Premise:\n\"\"\"\n{raw_sentence}\n\"\"\"\n\n"
    "Hypothesis:\n\"\"\"\n{kg_text}\n\"\"\"\n\n"
    "Output:"
)

# Base extraction template ΓÇö {n}, {graph_list}, {text_block} filled at runtime
_MULTI_EXTRACTION_BASE = (
    "Your task is to transform {n} related text(s) into a COMPREHENSIVE set of knowledge graphs "
    "as lists of triplets. Extract all {n} graphs simultaneously so that entity and relation "
    "labels are CONSISTENT ACROSS ALL THE GRAPHS whenever possible.\n\n"
    "IMPORTANT RULES:\n"
    "- Extract EVERY piece of information mentioned in each text as triplets\n"
    "- The SAME real-world entity MUST use the SAME label in ALL graphs "
    "(e.g., always \"Marie_Curie\", never sometimes \"Curie\" or \"M._Curie\")\n"
    "- The SAME relationship type MUST use the SAME label in ALL graphs "
    "(e.g., always \"born_in\", never sometimes \"born_at\")\n"
    "- Keep entities concise using underscores for multi-word entities "
    "(e.g., \"Dr._Emily_Carter\", \"New_York\")\n"
    "- Keep relations simple and reusable (e.g., \"located_in\", \"member_of\", \"citizen_of\")\n"
    "- Do NOT include duplicate triplets within a graph\n"
    "- Do NOT infer or assume information not explicitly stated in the respective text\n"
    "- Replace pronouns with actual entity names when clear from context\n\n"
    "OUTPUT FORMAT (output ONLY these {n} line(s), no extra text):\n"
    "{graph_list}\n\n"
    "EXAMPLE:\n"
    "Text 1: Marie Curie was born in Warsaw on November 7, 1867. She won the Nobel Prize in Physics in 1903.\n"
    "Text 2: Marie Curie, the Polish physicist, conducted research in Paris alongside Pierre Curie.\n"
    "graph_1: [['Marie_Curie', 'born_in', 'Warsaw'], ['Marie_Curie', 'birth_date', 'November_7_1867'], "
    "['Marie_Curie', 'won', 'Nobel_Prize_in_Physics'], ['Nobel_Prize_in_Physics', 'awarded_in', '1903']]\n"
    "graph_2: [['Marie_Curie', 'nationality', 'Polish'], ['Marie_Curie', 'occupation', 'physicist'], "
    "['Marie_Curie', 'conducted_research_in', 'Paris'], ['Marie_Curie', 'collaborated_with', 'Pierre_Curie']]\n\n"
    "{text_block}"
)

_BACKWARD_EVAL_TMPL = (
    "You are an expert evaluator for multi-graph relation triplet extraction.\n\n"
    "Given joint extraction results from {n} paragraph(s):\n\n"
    "{graph_eval_block}\n"
    "Combined NLI score (average across all graphs): {score}\n\n"
    "Provide concise, structured, and actionable feedback focused on:\n"
    "1. Correctness \u2014 which triplets were hallucinated or contradicted in any graph?\n"
    "2. Completeness \u2014 what explicitly stated facts were missed from any paragraph?\n"
    "3. Clarity \u2014 are entity/relation labels consistent and concise?\n"
    "4. Alignment \u2014 are the same real-world entities/relations using the same labels across all graphs?\n\n"
    "Do NOT list triplets again. Identify PATTERNS behind failures only."
)

_BACKWARD_PRED_TMPL = (
    "You are an expert evaluator for Multi-Graph Relational Triplet Extraction.\n\n"
    "Given:\n"
    "- The multi-graph extraction system prompt: {sys_prompt}\n"
    "{para_block}"
    "{triplet_block}"
    "- NLI-based feedback: {feedback}\n\n"
    "Provide concise and actionable feedback on how to improve the system prompt to better guide "
    "simultaneous multi-graph extraction in terms of correctness, completeness, clarity, and "
    "cross-graph label consistency.\n"
    "Focus on structural and instructional improvements only."
)

_OPTIMIZER_EXAMPLE_TMPL = (
    "---\n"
    "<CONVERSATION>\n"
    "<LM_SYSTEM_PROMPT> {sys_prompt} </LM_SYSTEM_PROMPT>\n"
    "<LM_INPUT>\n{text_block}</LM_INPUT>\n"
    "<LM_OUTPUT>\n{triplet_block}</LM_OUTPUT>\n"
    "</CONVERSATION>\n"
    "<FEEDBACK>{feedback}</FEEDBACK>\n"
    "---"
)

_OPTIMIZER_TMPL = (
    "You are improving a structured system prompt for Multi-Graph Relational Triplet Extraction.\n\n"
    "The prompt extracts {n} knowledge graph(s) simultaneously from {n} related paragraph(s) "
    "with label alignment embedded.\n\n"
    "The prompt to improve:\n"
    "<VARIABLE> {sys_prompt} </VARIABLE>\n\n"
    "Contextual feedback from past extractions:\n"
    "<CONTEXT> {examples} </CONTEXT>\n\n"
    "Optimize the prompt: make it clearer, more complete, better grounded in source texts, "
    "and better at consistent entity and relation labels across all graphs.\n"
    "Constraints:\n"
    "- Preserve output format EXACTLY:\n"
    "{graph_format_hint}"
    "- Keep entities concise with underscores for multi-word entities\n"
    "- Relations must be simple and reusable\n"
    "- Do NOT add new examples to the prompt\n"
    "- Do NOT infer information not explicitly stated\n"
    "- Same real-world entities/relations MUST use the same label in all graphs\n\n"
    "Output Format:\n"
    "<IMPROVED_PROMPT>\n"
    "{{The improved prompt only, no preamble.}}\n"
    "</IMPROVED_PROMPT>"
)


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# Prompt builders
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def _build_base_prompt(n: int) -> str:
    """Build the multi-extraction base prompt for N paragraphs."""
    graph_list = "\n".join(
        f"graph_{i}: [['Entity1', 'relation', 'Entity2'], ...]" for i in range(1, n + 1)
    )
    text_block = ""
    return _MULTI_EXTRACTION_BASE.format(n=n, graph_list=graph_list, text_block=text_block)


def _build_extraction_prompt(oie_prompt: str, paragraphs: list) -> str:
    text_lines = "\n".join(f"Text {i+1}: {p}" for i, p in enumerate(paragraphs))
    return oie_prompt + "\n" + text_lines


# Parsing helpers

def _fallback_parse(raw: str) -> list:
    raw = raw.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(raw)
            if isinstance(result, list):
                return result
        except Exception:
            pass
    triplets = re.findall(r"\[(['\"][^]]+?['\"])\]", raw)
    parsed = []
    for t in triplets:
        elems = re.findall(r"""['"]([^'"]+)['"]""", t)
        if len(elems) == 3:
            parsed.append(elems)
    return parsed


def _parse_graph_n_format(response: str, n: int) -> dict:
    """Parse 'graph_1: [...]\ngraph_2: [...]...' format for N graphs."""
    graphs = {}
    for i in range(1, n + 1):
        pattern = rf"graph_{i}\s*:\s*(\[.*?\])(?=\s*graph_|\s*$)"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            try:
                graphs[i] = ast.literal_eval(m.group(1))
            except Exception:
                graphs[i] = _fallback_parse(m.group(1))
        else:
            parts = re.split(r"graph_\d+\s*:", response)
            if i < len(parts):
                raw = parts[i].strip()
                bm = re.search(r"\[.*\]", raw, re.DOTALL)
                if bm:
                    try:
                        graphs[i] = ast.literal_eval(bm.group(0))
                    except Exception:
                        graphs[i] = _fallback_parse(bm.group(0))
                else:
                    graphs[i] = []
            else:
                graphs[i] = []
    return graphs


def _extract_improved_prompt(raw: str):
    m = re.search(r"<IMPROVED_PROMPT>(.*?)</IMPROVED_PROMPT>", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# LLM wrapper
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def _llm(llm_factory, prompt: str, max_tokens: int = 2048) -> str:
    for attempt in range(4):
        try:
            return llm_factory.generate(prompt, temperature=0.0, max_tokens=max_tokens)
        except Exception as e:
            log.warning(f"LLM call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# KRPO scoring
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def _triplet_to_text(llm_factory, triplet: list) -> str:
    prompt = _TRIPLET2TEXT.format(input_triplet=str(triplet))
    return _llm(llm_factory, prompt, max_tokens=256)


def _nli_label(llm_factory, paragraph: str, hypothesis: str) -> str:
    """Returns one of 'entailment', 'neutral', 'contradiction'."""
    prompt = _NLI_TMPL.format(
        few_shot_examples=_NLI_EXAMPLES,
        raw_sentence=paragraph,
        kg_text=hypothesis,
    )
    raw = _llm(llm_factory, prompt, max_tokens=300)
    try:
        jm = re.search(r"```json\s*(.*?)```", raw, re.DOTALL)
        if jm:
            raw = jm.group(1).strip()
        raw = re.sub(r',\s*"reasoning"\s*:\s*".*$', "}", raw, flags=re.DOTALL)
        result = json.loads(raw.replace("\n", ""))
        return result.get("label", "neutral").lower()
    except Exception:
        pass
    rl = raw.lower()
    if "entailment" in rl:
        return "entailment"
    if "contradiction" in rl:
        return "contradiction"
    return "neutral"


def _score_graph(triplets: list, paragraph: str, llm_factory) -> tuple:
    """Returns (score, eval_pairs). eval_pairs = [(triplet, label), ...]"""
    if not triplets:
        return 0.0, []
    score = 0.0
    pairs = []
    for tri in triplets:
        text = _triplet_to_text(llm_factory, tri)
        label = _nli_label(llm_factory, paragraph, text)
        pairs.append((tri, label))
        if label == "entailment":
            score += 1.0
        elif label == "contradiction":
            score -= 0.5
    return score / len(triplets), pairs


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# KRPO backward / optimizer helpers
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def _backward_eval(llm_factory, graphs: list, all_pairs: list, combined_score: float, n: int) -> str:
    graph_eval_block = ""
    for i, (graph, pairs) in enumerate(zip(graphs, all_pairs), 1):
        eval_str = "\n".join(f"  {t}: {l}" for t, l in pairs)
        graph_eval_block += (
            f"Graph {i} triplets: {graph}\n"
            f"Graph {i} NLI evaluations:\n{eval_str}\n\n"
        )
    prompt = _BACKWARD_EVAL_TMPL.format(
        n=n,
        graph_eval_block=graph_eval_block,
        score=f"{combined_score:.4f}",
    )
    return _llm(llm_factory, prompt, max_tokens=512)


def _backward_pred(llm_factory, oie_prompt: str, paragraphs: list, graphs: list, feedback: str) -> str:
    para_block = "".join(f"- Input Paragraph {i+1}: {p}\n" for i, p in enumerate(paragraphs))
    triplet_block = "".join(f"- Graph {i+1} extracted triplets: {g}\n" for i, g in enumerate(graphs))
    prompt = _BACKWARD_PRED_TMPL.format(
        sys_prompt=oie_prompt,
        para_block=para_block,
        triplet_block=triplet_block,
        feedback=feedback,
    )
    return _llm(llm_factory, prompt, max_tokens=512)


def _optimize_prompt(llm_factory, oie_prompt: str, paragraphs: list, graphs: list, feedback: str, n: int):
    text_block = "".join(f"Text {i+1}: {p}\n" for i, p in enumerate(paragraphs))
    triplet_block = "".join(f"graph_{i+1}: {g}\n" for i, g in enumerate(graphs))
    example = _OPTIMIZER_EXAMPLE_TMPL.format(
        sys_prompt=oie_prompt,
        text_block=text_block,
        triplet_block=triplet_block,
        feedback=feedback,
    )
    graph_format_hint = "".join(
        f"    graph_{i}: [['Entity1', 'relation', 'Entity2'], ...]\n" for i in range(1, n + 1)
    )
    prompt = _OPTIMIZER_TMPL.format(
        n=n,
        sys_prompt=oie_prompt,
        examples=example,
        graph_format_hint=graph_format_hint,
    )
    raw = _llm(llm_factory, prompt, max_tokens=1024)
    return _extract_improved_prompt(raw)


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# Public API
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

def krpo_extract(
    llm_factory,
    paragraphs: list,
    max_rounds: int = 3,
    row_id: str = "",
) -> dict:
    """
    KRPO loop for simultaneous N-graph extraction.

    Args:
        llm_factory: LLM factory with .generate(prompt, temperature, max_tokens) method.
        paragraphs:  List of N paragraph strings (N ΓëÑ 1).
        max_rounds:  Maximum KRPO refinement rounds (default 3).
        row_id:      Identifier for logging (optional).

    Returns:
        dict: {"graph_1": [...], "graph_2": [...], ...} ΓÇö one list of triplets per paragraph.
    """
    n = len(paragraphs)
    oie_prompt = _build_base_prompt(n)

    best_graphs: list = [[] for _ in range(n)]
    best_combined_score = -999.0

    for rnd in range(max_rounds):
        full_prompt = _build_extraction_prompt(oie_prompt, paragraphs)
        raw = _llm(llm_factory, full_prompt, max_tokens=2048)
        parsed = _parse_graph_n_format(raw, n)
        graphs = [parsed.get(i + 1, []) for i in range(n)]

        if not any(graphs):
            log.warning(f"  Row {row_id} round {rnd+1}: no triplets extracted ΓÇö skipping round")
            continue

        scores_and_pairs = [_score_graph(g, p, llm_factory) for g, p in zip(graphs, paragraphs)]
        scores = [s for s, _ in scores_and_pairs]
        all_pairs = [pairs for _, pairs in scores_and_pairs]
        combined_score = sum(scores) / n

        score_details = " | ".join(
            f"graph_{i+1}={len(graphs[i])} triplets score={scores[i]:.3f}"
            for i in range(n)
        )
        log.info(f"  Row {row_id} round {rnd+1}: {score_details} | combined={combined_score:.3f}")

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_graphs = graphs

        if combined_score >= 0.85 or rnd == max_rounds - 1:
            break

        feedback1 = _backward_eval(llm_factory, graphs, all_pairs, combined_score, n)
        feedback2 = _backward_pred(llm_factory, oie_prompt, paragraphs, graphs, feedback1)
        new_prompt = _optimize_prompt(llm_factory, oie_prompt, paragraphs, graphs, feedback2, n)

        if new_prompt:
            trial_raw = _llm(llm_factory, _build_extraction_prompt(new_prompt, paragraphs), max_tokens=2048)
            trial_parsed = _parse_graph_n_format(trial_raw, n)
            trial_graphs = [trial_parsed.get(i + 1, []) for i in range(n)]
            if any(trial_graphs):
                trial_scores = [_score_graph(g, p, llm_factory)[0] for g, p in zip(trial_graphs, paragraphs)]
                trial_combined = sum(trial_scores) / n
                if trial_combined > best_combined_score:
                    oie_prompt = new_prompt
                    best_combined_score = trial_combined
                    best_graphs = trial_graphs
                    log.info(f"  Row {row_id}: prompt updated, new combined score={trial_combined:.3f}")

    return {f"graph_{i+1}": best_graphs[i] for i in range(n)}
