"""
create_pubmedqa_ranked_faithfulness_dataset.py

Creates a RANKED FAITHFULNESS benchmark for evaluating whether a similarity
metric can correctly order candidate answers by how faithfully each reflects
the source medical context.

Evaluation thesis
-----------------
Four candidates are generated per question at four discrete faithfulness levels.
The ideal metric should score them in descending order (4 > 3 > 2 > 1).
Ranking quality is measured with Kendall's tau and Spearman correlation.

The CRITICAL discriminative boundary is Level 3 → Level 2 (numerical_distortion
vs. relation_inversion). Embedding-based metrics typically score both above 0.95
because "A inhibits B" and "B inhibits A" are lexically almost identical.  A
KG-based method that captures edge *direction* should clearly distinguish them.

Faithfulness levels
-------------------
  4  faithful_paraphrase   — Active→passive + medical vocab swap + synonym sub.
                             Semantically equivalent but surface-different.
  3  numerical_distortion  — Key statistics shifted ±35–60 %. Factually wrong
                             numbers, but the relational structure is intact.
  2  relation_inversion    — One subject↔object swap (simple, no compound).
                             Factually wrong causal direction.
  1  context_ignoring      — Ground-truth answer from a *different* medical topic.
                             Completely off-context.

Dataset structure
-----------------
  100 questions × 4 candidate rows = 400 rows.
  A question is included ONLY when all four perturbations succeed.

Columns
-------
  pair_id, question_id, question, context,
  gt_answer, candidate_answer,
  faithfulness_level, change_type, dataset_name

REQUIRES Python ≤ 3.13  (spaCy is not yet compatible with Python 3.14)
Run with:
  cd /Users/admin/Desktop/FYP_Benchmarking
  source .venv-spacy/bin/activate
  python3 src/evaluation/dataset_creation/create_pubmedqa_ranked_faithfulness_dataset.py

Output: datasets/pubmedqa_ranked_faithfulness_400.csv
"""

import re
import random
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# NLTK downloads
# ─────────────────────────────────────────────────────────────────────────────

for _pkg in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4'):
    nltk.download(_pkg, quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SEED              = 42
TARGET_QUESTIONS  = 100          # quadruplets; 100 × 4 = 400 rows
MIN_ANSWER_WORDS  = 30
MIN_ANSWER_SENTS  = 2

# Output path relative to THIS script's location (src/evaluation/datasets/)
OUTPUT_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# spaCy
# ─────────────────────────────────────────────────────────────────────────────

def _load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(
            ["python3", "-m", "spacy", "download", "en_core_web_sm"], check=True
        )
        return spacy.load("en_core_web_sm")


nlp = _load_spacy()

# ─────────────────────────────────────────────────────────────────────────────
# Medical entity extraction (regex)
# ─────────────────────────────────────────────────────────────────────────────

_MEDICAL_RE = re.compile(
    r'\b(?:'
    r'[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,4}'   # Title-Case multi-word
    r'|'
    r'[A-Z]{2,10}'                                   # ALL-CAPS acronym
    r')\b'
)
_SKIP_PHRASES = {
    'However', 'Therefore', 'Moreover', 'Furthermore', 'Although',
    'In Conclusion', 'In Summary', 'The Results', 'This Study',
    'These Results', 'The Present', 'Our Results', 'We Found',
    'It Is', 'There Is', 'There Are',
}


def extract_medical_entities(text: str) -> list[str]:
    found = []
    for m in _MEDICAL_RE.finditer(text):
        phrase = m.group(0)
        if phrase not in _SKIP_PHRASES and len(phrase) > 2:
            found.append(phrase)
    return list(dict.fromkeys(found))


# ─────────────────────────────────────────────────────────────────────────────
# Medical vocabulary paraphrases
# ─────────────────────────────────────────────────────────────────────────────

_MED_PARAPHRASES: dict[str, str] = {
    'most': 'the majority of', 'many': 'numerous',
    'few': 'a small number of', 'several': 'a number of', 'some': 'certain',
    'often': 'frequently', 'rarely': 'infrequently',
    'commonly': 'frequently', 'generally': 'typically', 'usually': 'typically',
    'show': 'demonstrate', 'find': 'observe', 'suggest': 'indicate',
    'conclude': 'determine', 'report': 'document', 'compare': 'contrast',
    'measure': 'assess', 'increase': 'elevate', 'decrease': 'reduce',
    'improve': 'enhance', 'affect': 'influence', 'cause': 'induce',
    'patients': 'individuals', 'patient': 'individual',
    'study': 'investigation', 'studies': 'investigations',
    'results': 'findings', 'result': 'finding', 'data': 'evidence',
    'risk': 'probability', 'treatment': 'therapy', 'treatments': 'therapies',
    'disease': 'condition', 'diseases': 'conditions',
    'outcome': 'result', 'outcomes': 'results',
    'significant': 'substantial', 'important': 'crucial',
    'associated': 'linked', 'higher': 'greater', 'lower': 'reduced',
    'similar': 'comparable', 'effective': 'efficacious', 'clinical': 'medical',
}


def apply_medical_vocab_swap(text: str, n: int = 4) -> str:
    candidates = [
        (surface, replacement)
        for surface, replacement in _MED_PARAPHRASES.items()
        if re.search(r'\b' + re.escape(surface) + r'\b', text, re.IGNORECASE)
    ]
    if not candidates:
        return text
    random.shuffle(candidates)
    result = text
    for surface, replacement in candidates[:n]:
        result = re.sub(
            r'\b' + re.escape(surface) + r'\b',
            replacement, result, count=1, flags=re.IGNORECASE
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Past participle lookup (for active → passive)
# ─────────────────────────────────────────────────────────────────────────────

_PAST_PARTICIPLES: dict[str, str] = {
    'show': 'shown', 'find': 'found', 'associate': 'associated',
    'observe': 'observed', 'report': 'reported', 'include': 'included',
    'increase': 'increased', 'reduce': 'reduced', 'improve': 'improved',
    'detect': 'detected', 'measure': 'measured', 'perform': 'performed',
    'conduct': 'conducted', 'evaluate': 'evaluated', 'assess': 'assessed',
    'treat': 'treated', 'cause': 'caused', 'inhibit': 'inhibited',
    'induce': 'induced', 'prevent': 'prevented', 'affect': 'affected',
    'use': 'used', 'apply': 'applied', 'define': 'defined',
    'confirm': 'confirmed', 'suggest': 'suggested', 'indicate': 'indicated',
    'demonstrate': 'demonstrated', 'identify': 'identified',
    'determine': 'determined', 'compare': 'compared', 'analyze': 'analyzed',
    'examine': 'examined', 'correlate': 'correlated', 'control': 'controlled',
    'select': 'selected', 'recruit': 'recruited', 'diagnose': 'diagnosed',
    'prescribe': 'prescribed', 'administer': 'administered',
    'classify': 'classified', 'stratify': 'stratified',
    'randomize': 'randomized', 'match': 'matched', 'exclude': 'excluded',
}


def _past_participle(root) -> str:
    lemma = root.lemma_
    if lemma in _PAST_PARTICIPLES:
        return _PAST_PARTICIPLES[lemma]
    if root.text.lower().endswith(('ed', 'en')):
        return root.text.lower()
    return lemma + ('d' if lemma.endswith('e') else 'ed')


def apply_active_passive(answer: str) -> str | None:
    """Convert first eligible active sentence to passive voice."""
    doc = nlp(answer)
    for sent in doc.sents:
        root = sent.root
        if root.pos_ != 'VERB':
            continue

        nsubj_list = [t for t in root.children if t.dep_ == 'nsubj']
        obj_list   = [t for t in root.children if t.dep_ in ('dobj', 'obj')]
        if not nsubj_list or not obj_list:
            continue

        nsubj_tok = nsubj_list[0]
        obj_tok   = obj_list[0]
        subj_span = doc[nsubj_tok.left_edge.i : nsubj_tok.right_edge.i + 1]
        obj_span  = doc[obj_tok.left_edge.i   : obj_tok.right_edge.i   + 1]

        subj_text = subj_span.text
        obj_text  = obj_span.text
        if len(subj_text) < 2 or len(obj_text) < 2:
            continue

        be_form = 'were' if obj_tok.tag_ in ('NNS', 'NNPS') else 'was'
        verb_pp = _past_participle(root)
        passive = f"{obj_text} {be_form} {verb_pp} by {subj_text}"

        trail    = sent.text.rstrip()[-1] if sent.text.rstrip()[-1] in '.!?' else '.'
        modified = answer.replace(sent.text.strip(), passive + trail, 1)

        if modified != answer and len(modified.split()) >= MIN_ANSWER_WORDS:
            return modified

    return None


# ─────────────────────────────────────────────────────────────────────────────
# WordNet synonym substitution
# ─────────────────────────────────────────────────────────────────────────────

_WN_POS = {
    'NN':  wordnet.NOUN, 'NNS': wordnet.NOUN,
    'JJ':  wordnet.ADJ,  'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
    'RB':  wordnet.ADV,  'RBR': wordnet.ADV, 'RBS': wordnet.ADV,
}
_SKIP_TAGS = {'NNP', 'NNPS'}


def _get_synonym(lemma: str, wn_pos: str) -> str | None:
    for syn in wordnet.synsets(lemma, pos=wn_pos):
        for lem in syn.lemmas():
            cand = lem.name().replace('_', ' ')
            if cand != lemma and len(cand) > 2 and cand.isalpha():
                return cand
    return None


def apply_synonym_paraphrase(answer: str, n_replacements: int = 7) -> str | None:
    doc = nlp(answer)

    ne_chars: set[int] = set()
    for ent in doc.ents:
        ne_chars.update(range(ent.start_char, ent.end_char))

    candidates: list[tuple] = []
    for token in doc:
        if token.tag_ in _SKIP_TAGS:
            continue
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if len(token.text) < 4:
            continue
        if token.text[0].isupper() or token.text.isupper():
            continue
        if any(i in ne_chars for i in range(token.idx, token.idx + len(token.text))):
            continue

        wn_pos = _WN_POS.get(token.tag_)
        if wn_pos is None:
            continue

        synonym = _get_synonym(token.lemma_, wn_pos)
        if synonym:
            candidates.append((token, synonym))

    if len(candidates) < 3:
        return None

    random.shuffle(candidates)
    to_replace = candidates[:n_replacements]

    result = answer
    offset = 0
    for token, synonym in sorted(to_replace, key=lambda x: x[0].idx):
        start  = token.idx + offset
        end    = start + len(token.text)
        result = result[:start] + synonym + result[end:]
        offset += len(synonym) - len(token.text)

    return result if result != answer else None


# ─────────────────────────────────────────────────────────────────────────────
# Sentence reorder paraphrase
# ─────────────────────────────────────────────────────────────────────────────

def apply_sentence_reorder(answer: str) -> str | None:
    sents = sent_tokenize(answer)
    if len(sents) < 4:
        return None
    first, *middle, last = sents
    if len(middle) < 2:
        return None
    shuffled = middle[:]
    random.shuffle(shuffled)
    if shuffled == middle:
        shuffled = middle[::-1]
    result = ' '.join([first] + shuffled + [last])
    return result if result != answer else None


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 4 — Faithful Paraphrase
# Multi-strategy: active→passive + medical vocab, heavy synonym + medical vocab,
# or sentence reorder + vocab + synonym.
# ─────────────────────────────────────────────────────────────────────────────

def make_faithful_paraphrase(answer: str) -> str | None:
    """
    Try three strategies in random order; return the first that produces
    a text clearly different from the original (> 3 changed tokens).
    """
    strategies = ['A', 'B', 'C']
    random.shuffle(strategies)

    for s in strategies:
        if s == 'A':
            step1 = apply_active_passive(answer)
            if step1:
                step2 = apply_medical_vocab_swap(step1, n=4)
                if step2 and step2 != answer:
                    return step2

        elif s == 'B':
            step1 = apply_synonym_paraphrase(answer, n_replacements=7)
            if step1:
                step2 = apply_medical_vocab_swap(step1, n=4)
                if step2 and step2 != answer:
                    return step2
                if step1 != answer:
                    return step1

        elif s == 'C':
            step1 = apply_sentence_reorder(answer) or answer
            step2 = apply_medical_vocab_swap(step1, n=3) or step1
            step3 = apply_synonym_paraphrase(step2, n_replacements=4)
            if step3 and step3 != answer:
                return step3

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3 — Factual Corruption
# Primary: shift statistics by ±35–60 % (skipping years 1900–2024).
# Fallback: replace one entity with a plausible same-domain entity (similar
#           MeSH topic), preserving the relational structure entirely.
#
# Either way the relation direction is intact — the answer is factually
# wrong in one detail but structurally coherent. This is the critical
# contrast with Level 2 (relation inversion).
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r'\b(\d+(?:\.\d+)?)\b')
_YEAR_RE = re.compile(r'\b(19\d{2}|20[01]\d|202[0-4])\b')


def _is_year(num_str: str) -> bool:
    return bool(_YEAR_RE.fullmatch(num_str.strip()))


def _try_numerical_distortion(answer: str, n_changes: int = 3) -> str | None:
    """Shift n non-year numbers by ±35–60 %. Returns None if no eligible numbers."""
    matches = [
        m for m in _NUM_RE.finditer(answer)
        if float(m.group()) > 0 and not _is_year(m.group())
    ]
    if not matches:
        return None

    to_change = random.sample(matches, min(n_changes, len(matches)))
    result = answer
    offset = 0
    for m in sorted(to_change, key=lambda x: x.start()):
        original = float(m.group())
        factor = random.choice([
            random.uniform(0.40, 0.65),
            random.uniform(1.35, 1.65),
        ])
        new_val = original * factor
        new_str = (
            str(int(round(new_val)))
            if original == int(original)
            else f"{new_val:.1f}"
        )
        start = m.start() + offset
        end   = m.end()   + offset
        result = result[:start] + new_str + result[end:]
        offset += len(new_str) - len(m.group())

    return result if result != answer else None


def _try_entity_substitution(
    answer: str,
    current_idx: int,
    records: list[dict],
) -> str | None:
    """
    Replace one medical entity with one from a SIMILAR topic (some MeSH overlap).
    Preserves relation structure — just one node/fact is wrong.
    """
    source_ents = extract_medical_entities(answer)
    if not source_ents:
        return None

    current_meshes = next(
        (r["meshes"] for r in records if r["idx"] == current_idx), set()
    )
    target = random.choice(source_ents)

    # Prefer docs with SOME MeSH overlap (same domain, not wildly off-topic)
    same_domain = [
        ent
        for r in records
        if r["idx"] != current_idx
        and current_meshes and r["meshes"] and not current_meshes.isdisjoint(r["meshes"])
        for ent in extract_medical_entities(r["long_answer"])
        if ent not in source_ents and ent != target
    ]
    if not same_domain:
        # Fallback: any different document's entity
        same_domain = [
            ent
            for r in records
            if r["idx"] != current_idx
            for ent in extract_medical_entities(r["long_answer"])
            if ent not in source_ents and ent != target
        ]
    if not same_domain:
        return None

    replacement = random.choice(same_domain)
    result = answer.replace(target, replacement, 1)
    return result if result != answer else None


def make_factual_corruption(
    answer: str,
    current_idx: int,
    records: list[dict],
) -> tuple[str, str] | tuple[None, None]:
    """
    Returns (corrupted_text, sub_type) where sub_type is
    'numerical_distortion' or 'entity_substitution'.
    Returns (None, None) on failure.
    """
    # Try numerical distortion first
    result = _try_numerical_distortion(answer)
    if result:
        return result, "numerical_distortion"

    # Fallback to entity substitution (same domain)
    result = _try_entity_substitution(answer, current_idx, records)
    if result:
        return result, "entity_substitution"

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — Relation Inversion (SIMPLE — subject ↔ object swap only)
# No compound perturbation here; the diagnostic value is the direction flip.
# This is the CRITICAL level — embeddings score it near-identical to level 3,
# but a KG metric capturing edge direction should score it much lower.
# ─────────────────────────────────────────────────────────────────────────────

def make_relation_inversion(answer: str) -> str | None:
    """
    Find the first sentence with a clear nsubj + dobj/obj and swap them.
    Returns None if no eligible sentence exists.
    """
    doc = nlp(answer)

    for sent in doc.sents:
        root = sent.root
        if root.pos_ != 'VERB':
            continue

        nsubj_list = [t for t in root.children if t.dep_ in ('nsubj', 'nsubjpass')]
        obj_list   = [t for t in root.children if t.dep_ in ('dobj', 'obj', 'attr')]
        if not nsubj_list or not obj_list:
            continue

        nsubj_tok  = nsubj_list[0]
        obj_tok    = obj_list[0]
        subj_span  = doc[nsubj_tok.left_edge.i : nsubj_tok.right_edge.i + 1]
        obj_span   = doc[obj_tok.left_edge.i   : obj_tok.right_edge.i   + 1]

        nsubj_text = subj_span.text
        obj_text   = obj_span.text

        if len(nsubj_text) < 2 or len(obj_text) < 2 or nsubj_text == obj_text:
            continue

        sent_text  = sent.text
        sent_start = answer.find(sent_text)
        if sent_start == -1:
            continue

        # Offsets within the sentence
        ns_start = subj_span.start_char - sent.start_char
        ns_end   = subj_span.end_char   - sent.start_char
        ob_start = obj_span.start_char  - sent.start_char
        ob_end   = obj_span.end_char    - sent.start_char

        # Only swap when subject appears before object
        if ns_end > ob_start:
            continue

        modified_sent = (
            sent_text[:ns_start] + obj_text +
            sent_text[ns_end:ob_start] + nsubj_text +
            sent_text[ob_end:]
        )
        modified_answer = (
            answer[:sent_start] + modified_sent +
            answer[sent_start + len(sent_text):]
        )

        if modified_answer != answer:
            return modified_answer

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — Context Ignoring
# Use the ground-truth answer of a *different* medical topic as the candidate.
# Maximum MeSH distance from current record ensures topic divergence.
# ─────────────────────────────────────────────────────────────────────────────

def make_context_ignoring(current_idx: int, records: list[dict]) -> str | None:
    current_rec = next((r for r in records if r["idx"] == current_idx), None)
    if not current_rec:
        return None

    candidates = [
        r for r in records
        if r["idx"] != current_idx
        and len(r["long_answer"].split()) >= MIN_ANSWER_WORDS
    ]
    if not candidates:
        return None

    # Sort by ascending MeSH overlap (fewest shared terms first = most off-topic)
    candidates.sort(
        key=lambda r: len(current_rec["meshes"] & r["meshes"])
    )
    quartile = max(1, len(candidates) // 4)
    return random.choice(candidates[:quartile])["long_answer"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pubmedqa() -> list[dict]:
    print("\nLoading PubMedQA (pqa_labeled) from HuggingFace …")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception:
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    records: list[dict] = []
    for idx, item in enumerate(ds):
        answer = (item.get("long_answer") or "").strip()
        if not answer:
            continue
        if len(answer.split()) < MIN_ANSWER_WORDS:
            continue
        if len(sent_tokenize(answer)) < MIN_ANSWER_SENTS:
            continue

        ctx_paras   = (item.get("context") or {}).get("contexts", [])
        ctx_snippet = ctx_paras[0].strip() if ctx_paras else ""
        if len(ctx_snippet.split()) > 300:
            ctx_snippet = " ".join(ctx_snippet.split()[:300]) + " …"

        meshes = (item.get("context") or {}).get("meshes", [])
        records.append({
            "idx":             idx,
            "question":        item["question"].strip(),
            "long_answer":     answer,
            "context_snippet": ctx_snippet,
            "meshes":          set(meshes) if meshes else set(),
            "final_decision":  (item.get("final_decision") or "").lower(),
        })

    print(f"✓ {len(records)} QA triples passed quality filter")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Quadruplet assembly
# ─────────────────────────────────────────────────────────────────────────────

def try_build_quadruplet(rec: dict, records: list[dict]) -> list[dict] | None:
    """
    Attempt to generate all four candidates for a single question.
    Returns a list of 4 row dicts (one per faithfulness level) or None if any
    perturbation fails.
    """
    gt = rec["long_answer"]

    # Level 4 — faithful paraphrase
    level4 = make_faithful_paraphrase(gt)
    if not level4 or level4 == gt:
        return None

    # Level 3 — numerical distortion
    level3 = make_numerical_distortion(gt)
    if not level3 or level3 == gt:
        return None

    # Level 2 — relation inversion (simple)
    level2 = make_relation_inversion(gt)
    if not level2 or level2 == gt:
        return None

    # Level 1 — context ignoring
    level1 = make_context_ignoring(rec["idx"], records)
    if not level1 or level1 == gt:
        return None

    base = {
        "question_id":  rec["idx"],
        "question":     rec["question"],
        "context":      rec["context_snippet"],
        "gt_answer":    gt,
        "dataset_name": "pubmedqa_ranked_faithfulness",
    }

    return [
        {**base, "candidate_answer": level4, "faithfulness_level": 4,
         "change_type": "faithful_paraphrase"},
        {**base, "candidate_answer": level3, "faithfulness_level": 3,
         "change_type": "numerical_distortion"},
        {**base, "candidate_answer": level2, "faithfulness_level": 2,
         "change_type": "relation_inversion"},
        {**base, "candidate_answer": level1, "faithfulness_level": 1,
         "change_type": "context_ignoring"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("PUBMEDQA RANKED FAITHFULNESS BENCHMARK — DATASET CREATION")
    print("=" * 70)
    print(f"Target  : {TARGET_QUESTIONS} quadruplets × 4 candidates = "
          f"{TARGET_QUESTIONS * 4} rows")
    print("Levels  : 4=faithful_paraphrase  3=numerical_distortion")
    print("          2=relation_inversion   1=context_ignoring")
    print("Metric  : Kendall's tau / Spearman corr over predicted ranking")
    print("Source  : PubMedQA pqa_labeled (1 000 expert-annotated QA triples)")
    print("NLP     : spaCy en_core_web_sm + WordNet  (no LLM, no KG)")

    records = load_pubmedqa()
    if not records:
        print("✗ No records loaded.")
        return

    shuffled = records[:]
    random.shuffle(shuffled)

    rows: list[dict] = []
    fail_stats: dict[str, int] = {
        "faithful_paraphrase": 0,
        "factual_corruption": 0,
        "relation_inversion": 0,
        "context_ignoring": 0,
    }

    pbar = tqdm(shuffled, desc="Building quadruplets")
    for rec in pbar:
        if len(rows) >= TARGET_QUESTIONS * 4:
            break

        gt = rec["long_answer"]

        # Level 4 — faithful paraphrase
        level4 = make_faithful_paraphrase(gt)
        if not level4 or level4 == gt:
            fail_stats["faithful_paraphrase"] += 1
            continue

        # Level 3 — factual corruption (numeric distortion OR entity substitution)
        level3, l3_type = make_factual_corruption(gt, rec["idx"], records)
        if not level3 or level3 == gt:
            fail_stats["factual_corruption"] += 1
            continue

        # Level 2 — relation inversion (simple SVO swap)
        level2 = make_relation_inversion(gt)
        if not level2 or level2 == gt:
            fail_stats["relation_inversion"] += 1
            continue

        # Level 1 — context ignoring (different-topic answer)
        level1 = make_context_ignoring(rec["idx"], records)
        if not level1 or level1 == gt:
            fail_stats["context_ignoring"] += 1
            continue

        base = {
            "question_id":  rec["idx"],
            "question":     rec["question"],
            "context":      rec["context_snippet"],
            "gt_answer":    gt,
            "dataset_name": "pubmedqa_ranked_faithfulness",
        }

        rows.extend([
            {**base, "candidate_answer": level4, "faithfulness_level": 4,
             "change_type": "faithful_paraphrase"},
            {**base, "candidate_answer": level3, "faithfulness_level": 3,
             "change_type": l3_type},
            {**base, "candidate_answer": level2, "faithfulness_level": 2,
             "change_type": "relation_inversion"},
            {**base, "candidate_answer": level1, "faithfulness_level": 1,
             "change_type": "context_ignoring"},
        ])

        q_done = len(rows) // 4
        pbar.set_postfix({"quadruplets": q_done})

    if not rows:
        print("✗ No quadruplets generated.")
        return

    df = pd.DataFrame(rows)
    df.insert(0, "pair_id", range(len(df)))
    df = df[[
        "pair_id", "question_id", "question", "context",
        "gt_answer", "candidate_answer",
        "faithfulness_level", "change_type", "dataset_name",
    ]]

    out = OUTPUT_DIR / "pubmedqa_ranked_faithfulness_400.csv"
    df.to_csv(out, index=False)

    n_q = len(df) // 4
    print("\n" + "=" * 70)
    print(f"✓ Saved {len(df)} rows ({n_q} quadruplets) → {out}")
    print("\nBreakdown by change_type:")
    print(df["change_type"].value_counts().to_string())
    print("\nBreakdown by faithfulness_level:")
    print(df["faithfulness_level"].value_counts().sort_index().to_string())
    print("\nPerturbation failure counts (records skipped per bottleneck level):")
    for k, v in fail_stats.items():
        print(f"  {k:<26}: {v}")
    print("\nLevel-3 sub-type breakdown:")
    l3 = df[df["faithfulness_level"] == 3]["change_type"].value_counts()
    print(l3.to_string())

    avg_len = df["candidate_answer"].str.split().str.len().mean()
    avg_gt  = df["gt_answer"].str.split().str.len().mean()
    print(f"\nAvg GT answer length        : {avg_gt:.0f} words")
    print(f"Avg candidate answer length : {avg_len:.0f} words")

    print("\nSample quadruplet (first question):")
    q_sample = df[df["question_id"] == df["question_id"].iloc[0]]
    for _, row in q_sample.iterrows():
        print(f"\n  Level {row['faithfulness_level']} [{row['change_type']}]")
        preview = row["candidate_answer"][:200].replace('\n', ' ')
        print(f"  {preview}…")


if __name__ == "__main__":
    main()