import numpy as np
import networkx as nx
import torch
from sentence_transformers import SentenceTransformer

try:
    from grakel import Graph
    from grakel.kernels import WeisfeilerLehman
    GRAKEL_AVAILABLE = True
except ImportError:
    Graph = None
    WeisfeilerLehman = None
    GRAKEL_AVAILABLE = False


def kg_similarity(kg1_triples, kg2_triples, alpha=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("paraphrase-MPNet-base-v2", device=device)

    def embed(texts):
        e = model.encode(texts, convert_to_tensor=True, device=device)
        return e.detach().cpu().numpy()

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    kg1 = [t for t in kg1_triples if len(t) == 3]
    kg2 = [t for t in kg2_triples if len(t) == 3]
    if not kg1 or not kg2:
        return 0.0

    e1 = embed([" ".join(map(str, t)) for t in kg1])
    e2 = embed([" ".join(map(str, t)) for t in kg2])

    filtered_kg2 = []
    for row in e1:
        sims = [cos(row, x) for x in e2]
        best = kg2[int(np.argmax(sims))]
        if best not in filtered_kg2:
            filtered_kg2.append(best)

    if not filtered_kg2:
        return 0.0

    def align(labels1, labels2, prefix):
        if not labels1 or not labels2:
            return {}, {}
        l1, l2 = list(labels1), list(labels2)
        n1 = embed(l1)
        n2 = embed(l2)
        n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-8
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-8
        s = np.dot(n1, n2.T)

        m1 = {
            l: (f"{prefix}_{np.argmax(s[i])}" if s[i].max() >= 0.65 else l)
            for i, l in enumerate(l1)
        }
        m2 = {l: f"{prefix}_{j}" for j, l in enumerate(l2)}
        return m1, m2

    def build_nx(triples):
        g = nx.Graph()
        for s, p, o in triples:
            s, p, o = str(s).lower(), str(p).lower(), str(o).lower()
            g.add_edge(s, o, relation=p)
            g.nodes[s]["label"] = s
            g.nodes[o]["label"] = o
        return g

    def relabel(g, mapping):
        h = nx.Graph()
        for n, d in g.nodes(data=True):
            lbl = d.get("label", n)
            h.add_node(n, label=mapping.get(lbl, lbl))
        for u, v, d in g.edges(data=True):
            h.add_edge(u, v, relation=d.get("relation", "rel"))
        return h

    def to_grakel(g):
        if not GRAKEL_AVAILABLE or not g.edges():
            return None
        return Graph(
            {(u, v): 1 for u, v in g.edges()},
            node_labels={n: d.get("label", n) for n, d in g.nodes(data=True)},
            edge_labels={(u, v): d.get("relation", "rel") for u, v, d in g.edges(data=True)},
        )

    g1 = build_nx(kg1)
    g2 = build_nx(filtered_kg2)

    nm1, nm2 = align(
        set(nx.get_node_attributes(g1, "label").values()),
        set(nx.get_node_attributes(g2, "label").values()),
        "node",
    )
    rm1, rm2 = align(
        set(nx.get_edge_attributes(g1, "relation").values()),
        set(nx.get_edge_attributes(g2, "relation").values()),
        "rel",
    )

    gk1 = to_grakel(relabel(g1, {**nm1, **rm1}))
    gk2 = to_grakel(relabel(g2, {**nm2, **rm2}))

    wl_score = 0.0
    if gk1 is not None and gk2 is not None:
        try:
            k = WeisfeilerLehman(n_iter=5, normalize=True).fit_transform([gk1, gk2])
            wl_score = float(k[0, 1])
        except Exception:
            wl_score = 0.0

    emb1 = embed([" ".join(map(str, t)) for t in kg1]).mean(axis=0)
    emb2 = embed([" ".join(map(str, t)) for t in filtered_kg2]).mean(axis=0)
    sbert_score = float(np.clip(cos(emb1, emb2), 0.0, 1.0))

    if not GRAKEL_AVAILABLE:
        return sbert_score

    return float(alpha * wl_score + (1.0 - alpha) * sbert_score)