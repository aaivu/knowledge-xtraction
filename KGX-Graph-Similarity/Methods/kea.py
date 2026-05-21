import networkx as nx
import numpy as np
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, NeighborhoodSubgraphPairwiseDistance
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sbert_model = SentenceTransformer('paraphrase-MPNet-base-v2', device=DEVICE)

def get_sbert_embedding(label):
    embedding = sbert_model.encode(label, convert_to_tensor=True, device=DEVICE)
    return embedding.detach().cpu().numpy()

def choose_representative(cluster):
    return min(cluster, key=len)

def is_number(label):
    try:
        float(label)
        return True
    except ValueError:
        return False

def cluster_data(nodes):
    new_nodes = {}
    
    numeric_labels = {label: label for label in nodes if is_number(label)}
    non_numeric_labels = [label for label in nodes if not is_number(label)]

    embeddings = np.array([get_sbert_embedding(label) for label in non_numeric_labels])
    cluster = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.35)
    labels = cluster.fit_predict(embeddings)

    clusters = {}
    for label, cluster_id in zip(non_numeric_labels, labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(label)

    for cluster_id, cluster in clusters.items():
        representative = choose_representative(cluster)
        for label in cluster:
            new_nodes[label] = representative

    new_nodes.update(numeric_labels)
    
    return new_nodes

def create_networkx_graph(triple_list):
    G = nx.Graph()
    for triple in triple_list:
        subject, predicate, obj = triple
        G.add_edge(subject.lower(), obj.lower(), relation=predicate.lower())
        G.nodes[subject.lower()]['label'] = subject.lower()
        G.nodes[obj.lower()]['label'] = obj.lower()
    return G



def relabel_graph(nx_graph, label_clusters):
    new_graph = nx.Graph()

    node_to_cluster = {}

    for node, data in nx_graph.nodes(data=True):
        original_label = data['label']
        curr_cluster = label_clusters[original_label]
        node_to_cluster[node] = curr_cluster
        new_graph.add_node(curr_cluster)

    for u, v, data in nx_graph.edges(data=True):
        new_u = node_to_cluster[u]
        new_v = node_to_cluster[v]
        
        if new_graph.has_edge(new_u, new_v):
            pass
        else:
            new_graph.add_edge(new_u, new_v, relation=data['relation'])

    return new_graph

def convert_to_grakel_graph(nx_graph):
    node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
    
    edge_labels = {(u, v): data.get('relation', 'default_relation') for u, v, data in nx_graph.edges(data=True)}

    edges = {(u, v): 1 for u, v in nx_graph.edges()}

    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)

def determine_number_of_clusters(labels):
    n_labels = len(labels)
    return max(3, int(np.sqrt(n_labels)))

def normalize_label(label):
    return label.lower()

def cluster_labels(labels):
    normalized_labels = [normalize_label(label) for label in labels]
    embeddings = np.vstack([get_sbert_embedding(label) for label in normalized_labels])
    n_clusters = determine_number_of_clusters(labels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clustered_labels = {label: f'cluster_{kmeans.labels_[i]}' for i, label in enumerate(labels)}
    return clustered_labels

def get_triple_embedding(triple):
    triple_text = ' '.join(map(str, triple))
    embedding = get_sbert_embedding(triple_text)
    return embedding

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def match_and_filter_triples(kg1_triples, kg2_triples):
    picked_triples = []
    for triple in kg1_triples:
        curr_kg1_embedding = get_triple_embedding(triple)
        similarities = []
        for other_triple in kg2_triples:
            curr_kg2_embedding = get_triple_embedding(other_triple)
            curr_similarity = cosine_similarity(curr_kg1_embedding, curr_kg2_embedding)
            similarities.append((curr_similarity, other_triple))

        chosen_triple = max(similarities, key=lambda x: x[0])
        if chosen_triple[1] not in picked_triples:
            picked_triples.append(chosen_triple[1])
    return picked_triples

def calculate_kea_similarity(kg1_triples, kg2_triples):

    kg1_triples = [sublist for sublist in kg1_triples if len(sublist) == 3]
    kg2_triples = [sublist for sublist in kg2_triples if len(sublist) == 3]


    if len(kg2_triples) == 0:
        return 0
    if len(kg1_triples) == 0:
        return 0

    filtered_kg2_triples = match_and_filter_triples(kg1_triples, kg2_triples)

    kg1_graph = create_networkx_graph(kg1_triples)
    kg2_graph = create_networkx_graph(filtered_kg2_triples)

    all_labels = set(nx.get_node_attributes(kg1_graph, 'label').values()) | \
                set(nx.get_node_attributes(kg2_graph, 'label').values()) | \
                set(nx.get_edge_attributes(kg1_graph, 'relation').values()) | \
                set(nx.get_edge_attributes(kg2_graph, 'relation').values())

    label_clusters = cluster_data(all_labels)


    relabelled_kg1 = relabel_graph(kg1_graph, label_clusters)
    relabelled_kg2 = relabel_graph(kg2_graph, label_clusters)

    kg1_grakel = convert_to_grakel_graph(relabelled_kg1)
    kg2_grakel = convert_to_grakel_graph(relabelled_kg2)

    nspd_kernel = WeisfeilerLehman(n_jobs=2, normalize=True)
    kernel_matrix = nspd_kernel.fit_transform([kg1_grakel, kg2_grakel])

    similarity = kernel_matrix[0, 1]
    return similarity, relabelled_kg1.edges(data=True), relabelled_kg2.edges(data=True)


def extract_all_labels(triples):
    """
    Extract all unique labels from triples (subjects, predicates, objects)

    Args:
        triples: List of triples [[s, p, o], ...]

    Returns:
        List of unique lowercase labels
    """
    labels = set()
    for triple in triples:
        if len(triple) == 3:
            labels.add(triple[0].lower())
            labels.add(triple[1].lower())
            labels.add(triple[2].lower())
    return list(labels)


def calculate_gaussian_feature_similarity(kg1_triples, kg2_triples, sigma=1.0):
    """
    Compute Gaussian kernel similarity on SBERT embeddings

    Formula: K(x1, x2) = exp(-||emb1 - emb2||^2 / (2 * sigma^2))

    Args:
        kg1_triples: First set of triples
        kg2_triples: Second set of triples
        sigma: Kernel width parameter (default: 1.0)

    Returns:
        float: Similarity score (0-1)
    """
    kg1_triples = [sublist for sublist in kg1_triples if len(sublist) == 3]
    kg2_triples = [sublist for sublist in kg2_triples if len(sublist) == 3]

    if len(kg1_triples) == 0 or len(kg2_triples) == 0:
        return 0.0

    labels1 = extract_all_labels(kg1_triples)
    labels2 = extract_all_labels(kg2_triples)

    if not labels1 or not labels2:
        return 0.0

    embeddings1 = np.array([get_sbert_embedding(label) for label in labels1])
    embeddings2 = np.array([get_sbert_embedding(label) for label in labels2])

    similarities = []
    for emb1 in embeddings1:
        max_sim = 0
        for emb2 in embeddings2:
            dist = np.linalg.norm(emb1 - emb2)
            sim = np.exp(-dist**2 / (2 * sigma**2))
            max_sim = max(max_sim, sim)
        similarities.append(max_sim)

    return float(np.mean(similarities))


def calculate_kea_composite_similarity(kg1_triples, kg2_triples, alpha=0.6, sigma=1.0):
    """
    Composite kernel: combines structural and semantic similarity

    Formula: composite = alpha * structural + (1-alpha) * semantic

    Args:
        kg1_triples: First set of triples
        kg2_triples: Second set of triples
        alpha: Weight for structural similarity (default: 0.6)
               1.0 = pure structural (original KEA)
               0.0 = pure semantic
        sigma: Gaussian kernel width (default: 1.0)

    Returns:
        dict: {
            'composite': combined score,
            'structural': WL kernel score,
            'semantic': Gaussian kernel score
        }
    """
    structural_sim, _, _ = calculate_kea_similarity(kg1_triples, kg2_triples)

    semantic_sim = calculate_gaussian_feature_similarity(kg1_triples, kg2_triples, sigma)

    composite_sim = alpha * structural_sim + (1 - alpha) * semantic_sim

    return {
        'composite': float(composite_sim),
        'structural': float(structural_sim),
        'semantic': float(semantic_sim)
    }
