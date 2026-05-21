"""
create_pubmedqa_contextual_dataset.py

Creates a QA-grounded benchmark for evaluating whether semantic similarity
methods detect contextual understanding failures in medical QA.

Grounding: PubMedQA (pqa_labeled) — 1000 expert-annotated medical QA triples
           each with a context passage, question, and ground-truth long answer.

Pair structure:
  response1  = Ground truth long answer  (reference)
  response2  = Perturbed or substitute answer
  label      = 0 (contextually wrong) | 1 (contextually aligned)

Perturbation types — label=0 (50 each, compound perturbations):
  entity_hallucination  — Replace 2-3 entities from different topic + distort numbers
  relation_inversion    — Invert SVO relation + replace entity in another sentence
  context_ignoring      — Completely different answer from different medical topic
  fact_omission         — Remove key sentence(s) + replace entity in remainder

Similar pairs — label=1 (200, diverse strategies):
  correct_paraphrase    — One of: active→passive + synonyms | heavy synonym +
                          medical vocab swap | sentence reorder + medical vocab swap

REQUIRES Python ≤ 3.13  (spaCy is not yet compatible with Python 3.14)
Run with:  source .venv-spacy/bin/activate
           python3 dataset_creation/create_pubmedqa_contextual_dataset.py

Output: datasets/pubmedqa_contextual_400.csv
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

SEED             = 42
TARGET_PER_TYPE  = 50
TARGET_SIMILAR   = 200
MIN_ANSWER_WORDS = 30
MIN_ANSWER_SENTS = 2

OUTPUT_DIR = Path("datasets")
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
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"], check=True)
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
# Numerical distortion  (used in entity_hallucination to add factual noise)
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r'\b(\d+(?:\.\d+)?)\b')


def _distort_number(text: str) -> str:
    """Shift one number in text by 25–60% up or down."""
    matches = [m for m in _NUM_RE.finditer(text) if float(m.group()) > 0]
    if not matches:
        return text
    m = random.choice(matches)
    original = float(m.group())
    factor   = random.choice([
        random.uniform(0.40, 0.70),   # shrink
        random.uniform(1.30, 1.70),   # inflate
    ])
    new_val = original * factor
    new_str = str(int(round(new_val))) if original == int(original) else f"{new_val:.1f}"
    return text[:m.start()] + new_str + text[m.end():]


# ─────────────────────────────────────────────────────────────────────────────
# Medical-domain vocabulary paraphrases
# (quantifiers, domain verbs, domain nouns — context-preserving swaps)
# ─────────────────────────────────────────────────────────────────────────────

_MED_PARAPHRASES: dict[str, str] = {
    # Quantifiers / determiners
    'most':        'the majority of',
    'many':        'numerous',
    'few':         'a small number of',
    'several':     'a number of',
    'some':        'certain',
    # Frequency
    'often':       'frequently',
    'rarely':      'infrequently',
    'commonly':    'frequently',
    'generally':   'typically',
    'usually':     'typically',
    # Domain verbs (uninflected — applied after lemma lookup)
    'show':        'demonstrate',
    'find':        'observe',
    'suggest':     'indicate',
    'conclude':    'determine',
    'report':      'document',
    'compare':     'contrast',
    'measure':     'assess',
    'increase':    'elevate',
    'decrease':    'reduce',
    'improve':     'enhance',
    'affect':      'influence',
    'cause':       'induce',
    # Domain nouns (exact surface match)
    'patients':    'individuals',
    'patient':     'individual',
    'study':       'investigation',
    'studies':     'investigations',
    'results':     'findings',
    'result':      'finding',
    'data':        'evidence',
    'risk':        'probability',
    'treatment':   'therapy',
    'treatments':  'therapies',
    'disease':     'condition',
    'diseases':    'conditions',
    'outcome':     'result',
    'outcomes':    'results',
    # Adjectives
    'significant': 'substantial',
    'important':   'crucial',
    'associated':  'linked',
    'higher':      'greater',
    'lower':       'reduced',
    'similar':     'comparable',
    'effective':   'efficacious',
    'clinical':    'medical',
}


def apply_medical_vocab_swap(text: str, n: int = 4) -> str:
    """
    Replace n tokens with medical-domain paraphrases using whole-word regex.
    Targets surface-level tokens (no lemmatisation) for exact matches.
    """
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
# Active → passive voice transformation (structural paraphrase)
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
    'classify': 'classified', 'stratify': 'stratified', 'enrol': 'enrolled',
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
    """
    Convert first eligible active-voice sentence to passive voice.
    e.g. "Aspirin inhibits inflammation." → "Inflammation was inhibited by Aspirin."
    Keeps the rest of the answer unchanged.
    """
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

        be_form  = 'were' if obj_tok.tag_ in ('NNS', 'NNPS') else 'was'
        verb_pp  = _past_participle(root)
        passive  = f"{obj_text} {be_form} {verb_pp} by {subj_text}"

        # Preserve trailing punctuation
        trail    = sent.text.rstrip()[-1] if sent.text.rstrip()[-1] in '.!?' else '.'
        modified = answer.replace(sent.text.strip(), passive + trail, 1)

        if modified != answer and len(modified.split()) >= MIN_ANSWER_WORDS:
            return modified

    return None


# ─────────────────────────────────────────────────────────────────────────────
# WordNet synonym substitution  (POS-aware, heavier replacement)
# ─────────────────────────────────────────────────────────────────────────────

_WN_POS = {
    'NN':  wordnet.NOUN, 'NNS': wordnet.NOUN,
    'JJ':  wordnet.ADJ,  'JJR': wordnet.ADJ,  'JJS': wordnet.ADJ,
    'RB':  wordnet.ADV,  'RBR': wordnet.ADV,  'RBS': wordnet.ADV,
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
    """
    Replace n content words (NOUN/ADJ/ADV) with WordNet synonyms.
    Skips proper nouns, named entities, stop words, short tokens, acronyms.
    Higher default n for more surface variation.
    """
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

        wn_pos  = _WN_POS.get(token.tag_)
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
        start   = token.idx + offset
        end     = start + len(token.text)
        result  = result[:start] + synonym + result[end:]
        offset += len(synonym) - len(token.text)

    return result if result != answer else None


# ─────────────────────────────────────────────────────────────────────────────
# Sentence reorder paraphrase (structural variety for multi-sentence answers)
# ─────────────────────────────────────────────────────────────────────────────

def apply_sentence_reorder(answer: str) -> str | None:
    """Shuffle interior sentences, keeping first and last fixed."""
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
# Composite paraphrase  (label=1) — randomly picks one of three strategies
# so the similar pairs have diverse surface changes, not just word swaps
# ─────────────────────────────────────────────────────────────────────────────

def make_paraphrase(answer: str) -> str | None:
    """
    Try three strategies in random order, return first that succeeds.

    Strategy A — structural:  active→passive  +  medical vocab swap (4 terms)
    Strategy B — lexical:     synonym sub (7) +  medical vocab swap (4 terms)
    Strategy C — discourse:   sentence reorder + medical vocab swap (3 terms)
                               + synonym sub (4)
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
            step1 = apply_sentence_reorder(answer)
            step1 = step1 or answer   # fallback to original order if reorder fails
            step2 = apply_medical_vocab_swap(step1, n=3)
            step2 = step2 or step1
            step3 = apply_synonym_paraphrase(step2, n_replacements=4)
            if step3 and step3 != answer:
                return step3

    return None


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
# Entity pool
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_pool(records: list[dict]) -> dict[int, list[str]]:
    pool: dict[int, list[str]] = {}
    for rec in records:
        ents = extract_medical_entities(rec["long_answer"])
        if ents:
            pool[rec["idx"]] = ents
    print(f"  Entity pool: {sum(len(v) for v in pool.values())} entities "
          f"across {len(pool)} documents")
    return pool


def _get_replacement_entity(
    avoid: set[str],
    current_idx: int,
    current_meshes: set,
    records: list[dict],
    entity_pool: dict[int, list[str]],
) -> str | None:
    """Find one entity from a different-topic document not already used."""
    # Prefer different MeSH topic
    candidates = [
        ent
        for idx, ents in entity_pool.items()
        if idx != current_idx
        for ent in ents
        if ent not in avoid
        and (not current_meshes or
             next((r for r in records if r["idx"] == idx), {}).get("meshes", set()).isdisjoint(current_meshes))
    ]
    if not candidates:
        candidates = [
            ent for idx, ents in entity_pool.items()
            if idx != current_idx
            for ent in ents
            if ent not in avoid
        ]
    return random.choice(candidates) if candidates else None


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 1 — Entity Hallucination  (compound: multi-entity + number)
#
# Replace 2–3 medical entities with entities from different-topic documents.
# Additionally distort one number in the text.
# Simulates: LLM hallucinates multiple entities not grounded in the context.
# ─────────────────────────────────────────────────────────────────────────────

def apply_entity_hallucination(
    answer: str,
    current_idx: int,
    records: list[dict],
    entity_pool: dict[int, list[str]],
    n_entities: int = 3,
) -> str | None:
    source_ents = extract_medical_entities(answer)
    if not source_ents:
        return None

    current_meshes = next((r["meshes"] for r in records if r["idx"] == current_idx), set())
    n = min(n_entities, len(source_ents))
    targets = random.sample(source_ents, n)

    result    = answer
    used_reps = set(source_ents)   # don't reuse any original entity

    for target in targets:
        rep = _get_replacement_entity(used_reps, current_idx, current_meshes, records, entity_pool)
        if rep:
            result = result.replace(target, rep, 1)
            used_reps.add(rep)

    # Additionally distort a number for extra factual noise
    result = _distort_number(result)

    return result if result != answer else None


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 2 — Relation Inversion  (compound: SVO swap + entity replacement)
#
# Step 1: Invert the subject/object of a key verb via spaCy dep parse.
# Step 2: In a DIFFERENT sentence, replace one entity with a wrong one.
# Simulates: LLM inverts a causal relation AND introduces a wrong entity.
# ─────────────────────────────────────────────────────────────────────────────

def apply_relation_inversion(
    answer: str,
    current_idx: int,
    records: list[dict],
    entity_pool: dict[int, list[str]],
) -> str | None:
    doc = nlp(answer)

    inverted_answer = None
    inverted_sent   = None

    for sent in doc.sents:
        root = sent.root
        if root.pos_ != 'VERB':
            continue

        nsubj_list = [t for t in root.children if t.dep_ in ('nsubj', 'nsubjpass')]
        obj_list   = [t for t in root.children if t.dep_ in ('dobj', 'obj', 'attr')]
        if not nsubj_list or not obj_list:
            continue

        nsubj_span = doc[nsubj_list[0].left_edge.i : nsubj_list[0].right_edge.i + 1]
        obj_span   = doc[obj_list[0].left_edge.i   : obj_list[0].right_edge.i   + 1]
        nsubj_text = nsubj_span.text
        obj_text   = obj_span.text

        if len(nsubj_text) < 2 or len(obj_text) < 2 or nsubj_text == obj_text:
            continue

        sent_text  = sent.text
        sent_start = answer.find(sent_text)
        if sent_start == -1:
            continue

        ns_start = nsubj_span.start_char - sent.start_char
        ns_end   = nsubj_span.end_char   - sent.start_char
        ob_start = obj_span.start_char   - sent.start_char
        ob_end   = obj_span.end_char     - sent.start_char

        if ns_end > ob_start:
            continue

        modified_sent = (
            sent_text[:ns_start] + obj_text +
            sent_text[ns_end:ob_start] + nsubj_text +
            sent_text[ob_end:]
        )
        inverted_answer = (
            answer[:sent_start] + modified_sent +
            answer[sent_start + len(sent_text):]
        )
        inverted_sent = sent_text
        break

    if inverted_answer is None or inverted_answer == answer:
        return None

    # Step 2: replace one entity in a sentence OTHER than the inverted one
    current_meshes = next((r["meshes"] for r in records if r["idx"] == current_idx), set())
    other_sents    = [s for s in sent_tokenize(inverted_answer) if s != inverted_sent]

    for other_sent in random.sample(other_sents, min(3, len(other_sents))):
        ents = extract_medical_entities(other_sent)
        if not ents:
            continue
        target = random.choice(ents)
        rep    = _get_replacement_entity(
            set(extract_medical_entities(inverted_answer)),
            current_idx, current_meshes, records, entity_pool
        )
        if rep:
            inverted_answer = inverted_answer.replace(target, rep, 1)
            break

    return inverted_answer if inverted_answer != answer else None


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 3 — Context Ignoring
# (unchanged — substituting a completely different answer is already strong)
# ─────────────────────────────────────────────────────────────────────────────

def apply_context_ignoring(current_idx: int, records: list[dict]) -> str | None:
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

    candidates.sort(key=lambda r: len(current_rec["meshes"] & r["meshes"]))
    quartile = max(1, len(candidates) // 4)
    return random.choice(candidates[:quartile])["long_answer"]


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation 4 — Fact Omission  (compound: remove sentence(s) + entity swap)
#
# Step 1: Remove the 1–2 most informationally dense sentences.
# Step 2: In the remaining text, replace one entity with a wrong one.
# Simulates: LLM omits the key fact AND introduces an unrelated entity.
# ─────────────────────────────────────────────────────────────────────────────

def _sentence_importance(sent_text: str, final_decision: str) -> float:
    doc_s  = nlp(sent_text)
    score  = float(len(doc_s.ents)) * 2.0
    if re.search(r'\d+', sent_text):
        score += 1.5
    if final_decision and final_decision in sent_text.lower():
        score += 3.0
    score += len(sent_text.split()) / 30.0
    return score


def apply_fact_omission(
    answer: str,
    final_decision: str,
    current_idx: int,
    records: list[dict],
    entity_pool: dict[int, list[str]],
) -> str | None:
    sentences = sent_tokenize(answer)
    if len(sentences) < MIN_ANSWER_SENTS:
        return None

    scores = [_sentence_importance(s, final_decision) for s in sentences]

    # Remove top 1 sentence (or top 2 if answer is long enough)
    n_remove = 2 if len(sentences) >= 5 else 1
    remove_indices = set(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_remove]
    )
    kept = [s for i, s in enumerate(sentences) if i not in remove_indices]

    if not kept:
        return None

    result = " ".join(kept)
    if len(result.split()) < MIN_ANSWER_WORDS // 2:
        return None

    # Step 2: replace one entity in the remaining text
    current_meshes = next((r["meshes"] for r in records if r["idx"] == current_idx), set())
    ents           = extract_medical_entities(result)
    if ents:
        target = random.choice(ents)
        rep    = _get_replacement_entity(
            set(ents), current_idx, current_meshes, records, entity_pool
        )
        if rep:
            result = result.replace(target, rep, 1)

    return result if result != answer else None


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_pair(rec, response1, response2, label, perturbation_type) -> dict:
    return {
        "question":          rec["question"],
        "context_snippet":   rec["context_snippet"],
        "response1":         response1,
        "response2":         response2,
        "label":             label,
        "perturbation_type": perturbation_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pair creation
# ─────────────────────────────────────────────────────────────────────────────

def create_entity_hallucination_pairs(records, entity_pool, n=TARGET_PER_TYPE):
    pairs: list[dict] = []
    shuffled = records[:]
    random.shuffle(shuffled)
    for rec in tqdm(shuffled, desc="  entity_hallucination", leave=False):
        if len(pairs) >= n:
            break
        modified = apply_entity_hallucination(
            rec["long_answer"], rec["idx"], records, entity_pool
        )
        if modified and modified != rec["long_answer"]:
            pairs.append(_make_pair(rec, rec["long_answer"], modified, 0, "entity_hallucination"))
    return pairs[:n]


def create_relation_inversion_pairs(records, entity_pool, n=TARGET_PER_TYPE):
    pairs: list[dict] = []
    shuffled = records[:]
    random.shuffle(shuffled)
    for rec in tqdm(shuffled, desc="  relation_inversion", leave=False):
        if len(pairs) >= n:
            break
        modified = apply_relation_inversion(
            rec["long_answer"], rec["idx"], records, entity_pool
        )
        if modified and modified != rec["long_answer"]:
            pairs.append(_make_pair(rec, rec["long_answer"], modified, 0, "relation_inversion"))
    return pairs[:n]


def create_context_ignoring_pairs(records, n=TARGET_PER_TYPE):
    pairs: list[dict] = []
    shuffled = records[:]
    random.shuffle(shuffled)
    for rec in tqdm(shuffled, desc="  context_ignoring", leave=False):
        if len(pairs) >= n:
            break
        wrong = apply_context_ignoring(rec["idx"], records)
        if wrong and wrong != rec["long_answer"]:
            pairs.append(_make_pair(rec, rec["long_answer"], wrong, 0, "context_ignoring"))
    return pairs[:n]


def create_fact_omission_pairs(records, entity_pool, n=TARGET_PER_TYPE):
    pairs: list[dict] = []
    shuffled = records[:]
    random.shuffle(shuffled)
    for rec in tqdm(shuffled, desc="  fact_omission", leave=False):
        if len(pairs) >= n:
            break
        modified = apply_fact_omission(
            rec["long_answer"], rec["final_decision"], rec["idx"], records, entity_pool
        )
        if modified and modified != rec["long_answer"]:
            pairs.append(_make_pair(rec, rec["long_answer"], modified, 0, "fact_omission"))
    return pairs[:n]


def create_paraphrase_pairs(records, n=TARGET_SIMILAR):
    pairs: list[dict] = []
    shuffled = records[:]
    random.shuffle(shuffled)
    pbar = tqdm(shuffled[: n * 4], desc="  correct_paraphrase", leave=False)
    for rec in pbar:
        if len(pairs) >= n:
            break
        modified = make_paraphrase(rec["long_answer"])
        if modified and modified != rec["long_answer"]:
            pairs.append(_make_pair(rec, rec["long_answer"], modified, 1, "correct_paraphrase"))
        pbar.set_postfix({"pairs": len(pairs)})
    return pairs[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("PUBMEDQA CONTEXTUAL UNDERSTANDING BENCHMARK — DATASET CREATION")
    print("=" * 65)
    print(f"Target  : {TARGET_SIMILAR} similar + {TARGET_PER_TYPE * 4} dissimilar "
          f"= {TARGET_SIMILAR + TARGET_PER_TYPE * 4} total pairs")
    print("Source  : PubMedQA pqa_labeled (1 000 expert-annotated QA triples)")
    print("NLP     : spaCy en_core_web_sm + WordNet  (no LLM, no KG)")
    print("Mode    : Compound perturbations for label=0, multi-strategy for label=1")

    records = load_pubmedqa()
    if not records:
        print("✗ No records loaded.")
        return

    print("\nBuilding medical entity pool …")
    entity_pool = build_entity_pool(records)

    print("\nCreating dissimilar pairs (label=0) …")
    eh = create_entity_hallucination_pairs(records, entity_pool)
    ri = create_relation_inversion_pairs(records, entity_pool)
    ci = create_context_ignoring_pairs(records)
    fo = create_fact_omission_pairs(records, entity_pool)

    print(f"\n  entity_hallucination : {len(eh):>3}  (2-3 entity swaps + number distortion)")
    print(f"  relation_inversion   : {len(ri):>3}  (SVO inversion + entity swap)")
    print(f"  context_ignoring     : {len(ci):>3}  (different-topic answer)")
    print(f"  fact_omission        : {len(fo):>3}  (key sentence(s) removed + entity swap)")

    print("\nCreating similar pairs (label=1) …")
    pp = create_paraphrase_pairs(records)
    print(f"  correct_paraphrase   : {len(pp):>3}  (active↔passive / heavy synonym / reorder+vocab)")

    all_pairs = eh + ri + ci + fo + pp
    if not all_pairs:
        print("✗ No pairs created.")
        return

    random.shuffle(all_pairs)

    df = pd.DataFrame(all_pairs)
    df.insert(0, "pair_id", range(len(df)))
    df["dataset_name"] = "pubmedqa_contextual"
    df = df[[
        "pair_id", "question", "context_snippet",
        "response1", "response2",
        "label", "perturbation_type", "dataset_name",
    ]]

    out = OUTPUT_DIR / "pubmedqa_contextual_400.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 65)
    print(f"✓ Saved {len(df)} pairs → {out}")
    print("\nBreakdown by perturbation type:")
    print(df["perturbation_type"].value_counts().to_string())
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    avg = (
        (df["response1"].str.split().str.len() + df["response2"].str.split().str.len()) / 2
    ).mean()
    print(f"\nAvg answer length : {avg:.0f} words")


if __name__ == "__main__":
    main()