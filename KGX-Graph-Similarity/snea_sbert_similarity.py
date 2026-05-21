# pip install sentence-transformers networkx numpy torch
# (grakel is NOT required ΓÇö WL kernel is implemented in pure Python below)

import hashlib
from collections import Counter, defaultdict

import numpy as np
import networkx as nx
import torch
from sentence_transformers import SentenceTransformer


# ΓöÇΓöÇ Pure-Python WeisfeilerΓÇôLehman subtree kernel ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Identical algorithm to GraKel's WeisfeilerLehman(n_iter=5, normalize=True)
# but requires no C++ compiler.

def _wl_kernel_pure(nx_g1: nx.Graph, nx_g2: nx.Graph, n_iter: int = 5) -> float:
    """Normalised WL subtree kernel between two labelled NetworkX graphs."""

    def _adj(G):
        adj = defaultdict(list)
        for u, v in G.edges():
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def _relabel(labels, adj):
        new = {}
        for n, lbl in labels.items():
            nbr_lbls = sorted(labels[nb] for nb in adj[n] if nb in labels)
            raw = lbl + '\x00' + '\x00'.join(nbr_lbls)
            new[n] = hashlib.blake2s(raw.encode(), digest_size=8).hexdigest()
        return new

    def _feature(labels):
        return Counter(labels.values())

    labels1 = {n: d.get('label', str(n)) for n, d in nx_g1.nodes(data=True)}
    labels2 = {n: d.get('label', str(n)) for n, d in nx_g2.nodes(data=True)}
    adj1 = _adj(nx_g1)
    adj2 = _adj(nx_g2)

    feat1: Counter = Counter()
    feat2: Counter = Counter()

    for _ in range(n_iter + 1):
        feat1 += _feature(labels1)
        feat2 += _feature(labels2)
        labels1 = _relabel(labels1, adj1)
        labels2 = _relabel(labels2, adj2)

    all_keys = set(feat1) | set(feat2)
    v1 = np.array([feat1.get(k, 0) for k in all_keys], dtype=float)
    v2 = np.array([feat2.get(k, 0) for k in all_keys], dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

def snea_sbert_similarity(kg1_triples, kg2_triples, alpha=0.5, return_details=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SentenceTransformer('paraphrase-MPNet-base-v2', device=device)

    def embed(texts):
        e = model.encode(texts, convert_to_tensor=True, device=device)
        return e.detach().cpu().numpy()

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    _empty = {'score': 0.0, 'wl_score': 0.0, 'sbert_score': 0.0,
              'matched_triples': [], 'node_alignments': [], 'relation_alignments': []}

    kg1 = [t for t in kg1_triples if len(t) == 3]
    kg2 = [t for t in kg2_triples if len(t) == 3]
    if not kg1 or not kg2:
        return _empty if return_details else 0.0

    # Step 1: semantic triple matching
    e1 = embed([' '.join(map(str, t)) for t in kg1])
    e2 = embed([' '.join(map(str, t)) for t in kg2])
    filtered_kg2 = []
    matched_triples_detail = []
    for i, row in enumerate(e1):
        sims = [cos(row, x) for x in e2]
        best_idx = int(np.argmax(sims))
        best = kg2[best_idx]
        if best not in filtered_kg2:
            filtered_kg2.append(best)
        matched_triples_detail.append({
            'kg1': list(map(str, kg1[i])),
            'kg2': list(map(str, best)),
            'sim': round(float(sims[best_idx]), 4),
        })
    if not filtered_kg2:
        return _empty if return_details else 0.0

    # Step 2: soft label alignment ΓÇö returns (mapping1, mapping2, aligned_pairs)
    def align(labels1, labels2, prefix):
        if not labels1 or not labels2:
            return {}, {}, []
        l1, l2 = list(labels1), list(labels2)
        n1 = embed(l1); n2 = embed(l2)
        n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-8
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-8
        S  = np.dot(n1, n2.T)
        m1 = {}
        pairs = []
        for i, l in enumerate(l1):
            best_j = int(np.argmax(S[i]))
            if S[i, best_j] >= 0.65:
                m1[l] = f"{prefix}_{best_j}"
                pairs.append({'from': l, 'to': l2[best_j], 'sim': round(float(S[i, best_j]), 4)})
            else:
                m1[l] = l
        m2 = {l: f"{prefix}_{j}" for j, l in enumerate(l2)}
        return m1, m2, pairs

    # Step 3: build graphs, relabel, run WL kernel
    def build_nx(triples):
        G = nx.Graph()
        for s, p, o in triples:
            s, p, o = s.lower(), p.lower(), o.lower()
            G.add_edge(s, o, relation=p)
            G.nodes[s]['label'] = s
            G.nodes[o]['label'] = o
        return G

    def relabel(G, mapping):
        H = nx.Graph()
        for n, d in G.nodes(data=True):
            lbl = d.get('label', n)
            H.add_node(n, label=mapping.get(lbl, lbl))
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, relation=d.get('relation', 'rel'))
        return H

    g1 = build_nx(kg1)
    g2 = build_nx(filtered_kg2)

    if not g1.edges() or not g2.edges():
        # fallback: SBERT only
        emb1 = embed([' '.join(map(str, t)) for t in kg1]).mean(axis=0)
        emb2 = embed([' '.join(map(str, t)) for t in filtered_kg2]).mean(axis=0)
        score = float(np.clip(cos(emb1, emb2), 0.0, 1.0))
        if return_details:
            return {**_empty, 'score': round(score, 4), 'sbert_score': round(score, 4),
                    'matched_triples': matched_triples_detail}
        return score

    nm1, nm2, node_pairs = align(set(nx.get_node_attributes(g1, 'label').values()),
                                  set(nx.get_node_attributes(g2, 'label').values()), 'node')
    rm1, rm2, rel_pairs  = align(set(nx.get_edge_attributes(g1, 'relation').values()),
                                  set(nx.get_edge_attributes(g2, 'relation').values()), 'rel')

    rl1 = relabel(g1, {**nm1, **rm1})
    rl2 = relabel(g2, {**nm2, **rm2})

    try:
        wl_score = _wl_kernel_pure(rl1, rl2, n_iter=5)
    except Exception:
        wl_score = 0.0

    # Step 4: SBERT mean-pool cosine
    emb1 = embed([' '.join(map(str, t)) for t in kg1]).mean(axis=0)
    emb2 = embed([' '.join(map(str, t)) for t in filtered_kg2]).mean(axis=0)
    sbert_score = float(np.clip(cos(emb1, emb2), 0.0, 1.0))

    # Step 5: blend
    final_score = float(alpha * wl_score + (1.0 - alpha) * sbert_score)

    if return_details:
        return {
            'score':               round(final_score, 4),
            'wl_score':            round(float(wl_score), 4),
            'sbert_score':         round(float(sbert_score), 4),
            'matched_triples':     matched_triples_detail,
            'node_alignments':     node_pairs,
            'relation_alignments': rel_pairs,
        }
    return final_score
