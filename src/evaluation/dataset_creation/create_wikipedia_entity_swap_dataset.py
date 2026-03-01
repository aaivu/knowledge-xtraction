"""
create_wikipedia_entity_swap_dataset.py

Creates a Wikipedia-based paragraph dataset with 4 perturbation types,
mirroring Semantic-KG methodology but using pure NLP tools (no KG construction).

NLP approach: regex + simple heuristics only (no spaCy, no NLTK ne_chunk)
→ low memory, Python 3.14 compatible, fast

Perturbation types (50 dissimilar pairs each):
  node_replacement  - Pair paragraphs about different entities of same type
  node_deletion     - Remove sentences about secondary proper-noun phrases
  edge_deletion     - Remove sentences that link two named phrases with a verb
  edge_replacement  - Replace a content verb with its antonym (WordNet)

Similar pairs (200):
  paraphrase via backtranslation (en→de→en) or sentence-reorder fallback

Output: datasets/wikipedia_entity_swap_400.csv
Format: pair_id, response1, response2, label, perturbation_type, dataset_name
"""

import re
import random
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NLTK downloads (lightweight — tokenizer only, no chunker)
# ---------------------------------------------------------------------------
for _p in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4'):
    nltk.download(_p, quiet=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED            = 42
TARGET_PER_TYPE = 50
TARGET_SIMILAR  = 200
PARA_MIN_WORDS  = 80
PARA_MAX_WORDS  = 400
WIKI_SCAN_LIMIT = 5000   # reduced from 8000
COLLECT_TARGET  = 900    # stop early once we have this many paragraphs

OUTPUT_DIR = Path("datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Regex entity detection  (replaces NLTK ne_chunk — no heavy model needed)
# ---------------------------------------------------------------------------

# Matches 1-5 consecutive Title-Case words (proxy for named entities)
_PROPER_RE = re.compile(r'\b[A-Z][a-zA-Z]{1,30}(?:\s+[A-Z][a-zA-Z]{1,30}){0,4}\b')

# Words that are capitalized but NOT entities (sentence starters, common titles)
_STOPWORDS = {
    'The', 'A', 'An', 'In', 'On', 'At', 'By', 'For', 'Of', 'To',
    'And', 'But', 'Or', 'Nor', 'As', 'Is', 'Was', 'Are', 'Were',
    'It', 'This', 'That', 'These', 'Those', 'He', 'She', 'They', 'We',
    'His', 'Her', 'Their', 'Its', 'Our', 'With', 'From', 'Into',
    'During', 'After', 'Before', 'Between', 'Through', 'Over', 'Under',
    'While', 'When', 'Where', 'Although', 'Because', 'Since',
    'However', 'Therefore', 'Thus', 'Hence', 'Also', 'Moreover',
}


def get_proper_phrases(text: str) -> list[str]:
    """
    Fast regex extraction of named-entity-like phrases.
    Skips first word of each sentence (always capitalised) and common stop words.
    """
    entities = []
    for sent in sent_tokenize(text):
        # Skip the very first word of the sentence
        trimmed = re.sub(r'^\S+\s*', '', sent)
        for m in _PROPER_RE.finditer(trimmed):
            phrase = m.group(0)
            if phrase not in _STOPWORDS and len(phrase) > 3:
                entities.append(phrase)
    return list(dict.fromkeys(entities))   # deduplicate, preserve order


# Heuristic keyword sets for classifying paragraph type
_PERSON_KW = {
    'born', 'died', 'married', 'graduated', 'actor', 'actress',
    'politician', 'musician', 'writer', 'athlete', 'player',
    'he', 'she', 'his', 'her', 'singer', 'director', 'author',
}
_ORG_KW = {
    'company', 'corporation', 'founded', 'established', 'university',
    'institute', 'organisation', 'organization', 'firm', 'agency',
    'department', 'ministry', 'headquartered', 'subsidiary',
}
_GPE_KW = {
    'country', 'city', 'capital', 'state', 'province', 'territory',
    'located', 'population', 'nation', 'republic', 'kingdom', 'region',
    'bordered', 'coast', 'river', 'island', 'continent',
}


def classify_type(text: str) -> str:
    words = set(text.lower().split())
    scores = {
        'PERSON': len(words & _PERSON_KW),
        'ORG':    len(words & _ORG_KW),
        'GPE':    len(words & _GPE_KW),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'PERSON'


# ---------------------------------------------------------------------------
# Antonym lookup
# ---------------------------------------------------------------------------

_MANUAL_ANTONYMS = {
    'increase': 'decrease', 'expand':  'reduce',   'support': 'oppose',
    'win':      'lose',     'create':  'destroy',   'join':    'leave',
    'lead':     'follow',   'founded': 'dissolved', 'help':    'hinder',
    'accept':   'reject',   'approve': 'disapprove','gain':    'lose',
    'build':    'demolish', 'grow':    'decline',   'rise':    'fall',
    'improve':  'worsen',   'defend':  'attack',    'enter':   'exit',
    'begin':    'end',      'open':    'close',     'include': 'exclude',
    'allow':    'prevent',  'promote': 'demote',    'elect':   'dismiss',
    'appoint':  'remove',   'produce': 'consume',   'sell':    'buy',
    'send':     'receive',  'give':    'take',      'love':    'hate',
    'born':     'died',     'married': 'divorced',  'hired':   'fired',
    'passed':   'failed',   'won':     'lost',      'gained':  'lost',
    'acquired': 'sold',     'joined':  'left',      'signed':  'rejected',
}

# Verbs we do NOT want to replace (too generic / grammatically risky)
_SKIP_VERBS = {
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    "'s", "'re", "'ve", "'d",
}


def verb_antonym(verb: str) -> str | None:
    lemma = verb.lower().strip(".,;:!?'\"")
    if lemma in _SKIP_VERBS:
        return None
    # WordNet first
    for syn in wordnet.synsets(lemma, pos=wordnet.VERB):
        for lem in syn.lemmas():
            if lem.antonyms():
                candidate = lem.antonyms()[0].name().replace('_', ' ')
                if candidate != lemma:
                    return candidate
    return _MANUAL_ANTONYMS.get(lemma)


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def apply_node_deletion(text: str) -> str | None:
    """
    Node deletion: remove sentences that introduce a secondary proper-noun phrase.
    Simulates removing a secondary entity from the KG.
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return None

    first_phrases = set(get_proper_phrases(sentences[0]))
    secondary = []
    for sent in sentences[1:]:
        for phrase in get_proper_phrases(sent):
            if phrase not in first_phrases:
                secondary.append((phrase, sent))

    if not secondary:
        return None

    # Pick a secondary phrase and remove all sentences mentioning it
    target_phrase, _ = random.choice(secondary)
    kept = [s for s in sentences if target_phrase not in s]

    if len(kept) == len(sentences) or len(kept) < 2:
        return None
    result = ' '.join(kept)
    return result if len(result.split()) >= PARA_MIN_WORDS else None


def apply_edge_deletion(text: str) -> str | None:
    """
    Edge deletion: remove a sentence that links two proper-noun phrases with a verb.
    Simulates removing an edge (relation) from the KG.
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return None

    # Find 'relation sentences': contain ≥2 proper phrases + a content word
    rel_sents = []
    for sent in sentences:
        phrases = get_proper_phrases(sent)
        if len(phrases) < 2:
            continue
        words_lower = sent.lower().split()
        # Must contain at least one non-copula verb-like word
        has_content = any(
            w not in _SKIP_VERBS and len(w) > 3
            for w in words_lower
        )
        if has_content:
            rel_sents.append(sent)

    if not rel_sents:
        return None

    target = random.choice(rel_sents)
    kept   = [s for s in sentences if s.strip() != target.strip()]

    if len(kept) == len(sentences) or len(kept) < 2:
        return None
    result = ' '.join(kept)
    return result if len(result.split()) >= PARA_MIN_WORDS else None


def apply_edge_replacement(text: str) -> str | None:
    """
    Edge replacement: replace a content verb with its antonym.
    Simulates changing the predicate of a KG edge.
    """
    # Look in sentences that have ≥2 proper phrases (relation sentences)
    sentences = sent_tokenize(text)
    random.shuffle(sentences)

    for sent in sentences:
        if len(get_proper_phrases(sent)) < 2:
            continue
        # Scan words for replaceable verbs
        words = sent.split()
        for i, word in enumerate(words):
            antonym = verb_antonym(word)
            if antonym:
                words[i] = antonym
                modified_sent = ' '.join(words)
                modified_text = text.replace(sent, modified_sent, 1)
                if modified_text != text:
                    return modified_text
    return None


# ---------------------------------------------------------------------------
# Paraphrase (similar pairs)
# ---------------------------------------------------------------------------

_en_de_pipe = None
_de_en_pipe = None


def _load_translation_models():
    global _en_de_pipe, _de_en_pipe
    if _en_de_pipe is None:
        from transformers import pipeline as hf_pipeline
        print("  Loading Helsinki-NLP models (~600 MB one-time)...")
        _en_de_pipe = hf_pipeline("translation_en_to_de",
                                  model="Helsinki-NLP/opus-mt-en-de", device=-1)
        _de_en_pipe = hf_pipeline("translation_de_to_en",
                                  model="Helsinki-NLP/opus-mt-de-en", device=-1)
        print("  ✓ Models loaded.")


def backtranslate(text: str) -> str | None:
    try:
        _load_translation_models()
        de = _en_de_pipe(text[:800], max_length=512)[0]['translation_text']
        en = _de_en_pipe(de,         max_length=512)[0]['translation_text']
        return en if en and en != text else None
    except Exception:
        return None


def sentence_reorder_paraphrase(text: str) -> str | None:
    """Fallback: shuffle interior sentences (no model needed)."""
    sentences = sent_tokenize(text)
    if len(sentences) < 4:
        return None
    first, *middle, last = sentences
    if len(middle) < 2:
        return None
    shuffled = middle[:]
    random.shuffle(shuffled)
    if shuffled == middle:   # already identical order — try again
        shuffled = middle[::-1]
    return ' '.join([first] + shuffled + [last])


def make_paraphrase(text: str, use_backtranslation: bool = True) -> str | None:
    if use_backtranslation:
        result = backtranslate(text)
        if result:
            return result
    return sentence_reorder_paraphrase(text)


# ---------------------------------------------------------------------------
# Wikipedia paragraph collection
# ---------------------------------------------------------------------------

_NOISE_RE = re.compile(r'(\[\[|\]\]|==|{\||<ref|thumb\||File:|Image:|\*\s|#\s)', re.I)


def is_clean(text: str) -> bool:
    if _NOISE_RE.search(text):
        return False
    words = text.split()
    return PARA_MIN_WORDS <= len(words) <= PARA_MAX_WORDS and text.count('.') >= 2


def collect_paragraphs():
    """
    Stream wikimedia/wikipedia and collect clean paragraphs.
    Uses regex classification instead of NLTK ne_chunk — memory efficient.
    """
    print(f"\nStreaming Wikipedia (scan ≤{WIKI_SCAN_LIMIT} articles, target {COLLECT_TARGET} paragraphs)...")

    wiki = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train",
        streaming=True,
    )

    by_type: dict[str, list] = defaultdict(list)
    flat:    list[dict]       = []
    scanned = 0

    for article in wiki:
        if scanned >= WIKI_SCAN_LIMIT or len(flat) >= COLLECT_TARGET:
            break

        for para in [p.strip() for p in article['text'].split('\n\n')][:4]:
            if not is_clean(para):
                continue

            phrases = get_proper_phrases(para)
            if len(phrases) < MIN_ENTITIES:
                continue

            etype = classify_type(para)
            entry = {'text': para, 'title': article['title'], 'etype': etype}
            by_type[etype].append(entry)
            flat.append(entry)
            break  # one paragraph per article

        scanned += 1
        if scanned % 500 == 0:
            print(f"  {scanned} articles scanned | {len(flat)} paragraphs | "
                  f"{ {k: len(v) for k, v in by_type.items()} }")

    print(f"✓ {len(flat)} paragraphs collected from {scanned} articles")
    print(f"  Type dist: { {k: len(v) for k, v in by_type.items()} }")
    return dict(by_type), flat


MIN_ENTITIES = 3  # minimum proper-noun phrases required

# ---------------------------------------------------------------------------
# Pair creation
# ---------------------------------------------------------------------------

def create_node_replacement_pairs(by_type: dict, n: int = TARGET_PER_TYPE) -> list[dict]:
    """Pair paragraphs from different articles but the same entity type."""
    pairs = []
    for paragraphs in by_type.values():
        random.shuffle(paragraphs)
        for i in range(0, len(paragraphs) - 1, 2):
            if len(pairs) >= n:
                break
            p1, p2 = paragraphs[i], paragraphs[i + 1]
            if p1['title'] != p2['title']:
                pairs.append({
                    'response1':         p1['text'],
                    'response2':         p2['text'],
                    'label':             0,
                    'perturbation_type': 'node_replacement',
                })
    return pairs[:n]


def create_node_deletion_pairs(flat: list, n: int = TARGET_PER_TYPE) -> list[dict]:
    pairs = []
    random.shuffle(flat)
    for entry in flat:
        if len(pairs) >= n:
            break
        modified = apply_node_deletion(entry['text'])
        if modified and modified != entry['text']:
            pairs.append({
                'response1':         entry['text'],
                'response2':         modified,
                'label':             0,
                'perturbation_type': 'node_deletion',
            })
    return pairs[:n]


def create_edge_deletion_pairs(flat: list, n: int = TARGET_PER_TYPE) -> list[dict]:
    pairs = []
    random.shuffle(flat)
    for entry in flat:
        if len(pairs) >= n:
            break
        modified = apply_edge_deletion(entry['text'])
        if modified and modified != entry['text']:
            pairs.append({
                'response1':         entry['text'],
                'response2':         modified,
                'label':             0,
                'perturbation_type': 'edge_deletion',
            })
    return pairs[:n]


def create_edge_replacement_pairs(flat: list, n: int = TARGET_PER_TYPE) -> list[dict]:
    pairs = []
    random.shuffle(flat)
    for entry in flat:
        if len(pairs) >= n:
            break
        modified = apply_edge_replacement(entry['text'])
        if modified and modified != entry['text']:
            pairs.append({
                'response1':         entry['text'],
                'response2':         modified,
                'label':             0,
                'perturbation_type': 'edge_replacement',
            })
    return pairs[:n]


def create_similar_pairs(flat: list, n: int = TARGET_SIMILAR,
                         use_backtranslation: bool = True) -> list[dict]:
    pairs = []
    random.shuffle(flat)
    pbar = tqdm(flat[:n * 4], desc="Creating paraphrases")
    for entry in pbar:
        if len(pairs) >= n:
            break
        para = make_paraphrase(entry['text'], use_backtranslation=use_backtranslation)
        if para and para != entry['text'] and len(para.split()) >= PARA_MIN_WORDS:
            pairs.append({
                'response1':         entry['text'],
                'response2':         para,
                'label':             1,
                'perturbation_type': 'paraphrase',
            })
        pbar.set_postfix({'pairs': len(pairs)})
    return pairs[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(use_backtranslation: bool = True):
    print("=" * 60)
    print("WIKIPEDIA ENTITY-SWAP DATASET CREATION")
    print("=" * 60)
    print(f"Target  : {TARGET_SIMILAR} similar + {TARGET_PER_TYPE * 4} dissimilar "
          f"= {TARGET_SIMILAR + TARGET_PER_TYPE * 4} pairs")
    print(f"Paraph  : {'backtranslation (en→de→en)' if use_backtranslation else 'sentence reorder (no model)'}")
    print(f"NLP     : regex + WordNet only (memory-efficient, Python 3.14 safe)")

    by_type, flat = collect_paragraphs()
    if not flat:
        print("✗ No paragraphs collected.")
        return

    print("\nCreating dissimilar pairs...")
    print("  [1/4] Node Replacement ...")
    node_repl = create_node_replacement_pairs(by_type)
    print(f"        {len(node_repl)} pairs")

    print("  [2/4] Node Deletion ...")
    node_del  = create_node_deletion_pairs(flat)
    print(f"        {len(node_del)} pairs")

    print("  [3/4] Edge Deletion ...")
    edge_del  = create_edge_deletion_pairs(flat)
    print(f"        {len(edge_del)} pairs")

    print("  [4/4] Edge Replacement ...")
    edge_repl = create_edge_replacement_pairs(flat)
    print(f"        {len(edge_repl)} pairs")

    print("\nCreating similar (paraphrase) pairs...")
    similar = create_similar_pairs(flat, use_backtranslation=use_backtranslation)
    print(f"  {len(similar)} pairs")

    all_pairs = node_repl + node_del + edge_del + edge_repl + similar
    if not all_pairs:
        print("✗ No pairs created.")
        return

    random.shuffle(all_pairs)
    df = pd.DataFrame(all_pairs)
    df['pair_id']      = range(len(df))
    df['dataset_name'] = 'wikipedia_entity_swap'
    df = df[['pair_id', 'response1', 'response2', 'label', 'perturbation_type', 'dataset_name']]

    out = OUTPUT_DIR / 'wikipedia_entity_swap_400.csv'
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(f"✓ Saved {len(df)} pairs → {out}")
    print("\nBreakdown:")
    print(df['perturbation_type'].value_counts().to_string())
    print("\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    avg = ((df['response1'].str.split().str.len() +
            df['response2'].str.split().str.len()) / 2).mean()
    print(f"\nAvg text length: {avg:.0f} words")
    print(f"\nNext: send '{out}' to colleague to run AA-KEA.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-backtranslation", action="store_true",
                        help="Use sentence reorder instead of translation models")
    args = parser.parse_args()
    main(use_backtranslation=not args.no_backtranslation)