from typing import List, Dict
# Nodewise group similarity function
def semantic_wl_kernel_nodewise_similarity(
    kg1_triplets: List[List[str]],
    kg2_triplets: List[List[str]],
    max_iterations: int = 5,
    similarity_threshold: float = 0.90,
    use_weighted_aggregation: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Compute Semantic WL Kernel similarity by comparing node group embeddings pairwise (no pooling).
    For each group, compute cosine similarity between corresponding group embeddings.
    Returns mean of these similarities and the vector.
    """
    # Step 1: Build directed graphs
    G1 = build_directed_graph(kg1_triplets)
    G2 = build_directed_graph(kg2_triplets)

    # Step 2-4: Run Semantic WL iterations
    labels1, embeddings1 = semantic_wl_iterations(G1, max_iterations, similarity_threshold, use_weighted_aggregation, verbose)
    labels2, embeddings2 = semantic_wl_iterations(G2, max_iterations, similarity_threshold, use_weighted_aggregation, verbose)

    # Get node groups for both graphs
    groups1 = compute_node_groups(embeddings1, similarity_threshold)
    groups2 = compute_node_groups(embeddings2, similarity_threshold)

    # Combine unique group IDs
    all_group_ids = sorted(set(groups1.values()) | set(groups2.values()))
    dim = get_embedding_dim()

    # Compute mean embedding for each group in each graph
    def group_mean_embedding(embeddings, groups, group_id):
        nodes = [n for n, gid in groups.items() if gid == group_id]
        if not nodes:
            return np.zeros(dim)
        return np.mean([embeddings[n] for n in nodes], axis=0)

    similarities = []
    for group_id in all_group_ids:
        emb1 = group_mean_embedding(embeddings1, groups1, group_id)
        emb2 = group_mean_embedding(embeddings2, groups2, group_id)
        sim = cosine_sim(emb1, emb2)
        similarities.append(sim)

    mean_similarity = float(np.mean(similarities)) if similarities else 0.0
    return {
        'similarity': mean_similarity,
        'groupwise_similarities': similarities,
        'group_ids': all_group_ids,
        'graph1_labels': labels1,
        'graph2_labels': labels2
    }
"""
Semantic Weisfeiler-Lehman (WL) Kernel for Knowledge Graph Similarity

This module implements a semantic variant of the WL kernel that uses SBERT embeddings
to compute similarity between knowledge graphs represented as triplet lists.

Improvements:
- Embedding caching for performance
- Batch embedding computation
- Weighted graph pooling (centrality-based)
- Edge embedding combination
- Weighted neighbor aggregation
- Cross-graph node alignment
- Configurable SBERT model
"""

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Set, Optional, Union

# Available SBERT models (can be configured)
SBERT_MODELS = {
    'default': 'all-MiniLM-L6-v2',           # 384-dim, fast
    'accurate': 'all-mpnet-base-v2',          # 768-dim, more accurate
    'biomedical': 'pritamdeka/S-PubMedBert-MS-MARCO',  # For biomedical KGs
    #'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2'  # Multilingual support
}

# Model dimensions
MODEL_DIMENSIONS = {
    'all-MiniLM-L6-v2': 384,
    'all-mpnet-base-v2': 768,
    'pritamdeka/S-PubMedBert-MS-MARCO': 768,
    'paraphrase-multilingual-MiniLM-L12-v2': 384
}

# Initialize SBERT model (can be changed via set_sbert_model)
_current_model_name = 'all-MiniLM-L6-v2'
sbert_model = SentenceTransformer(_current_model_name)

# Embedding cache for performance
_embedding_cache: Dict[str, np.ndarray] = {}
_cache_enabled = True


def set_sbert_model(model_name: str = 'default') -> None:
    """
    Set the SBERT model to use for embeddings.
    
    Args:
        model_name: Either a key from SBERT_MODELS ('default', 'accurate', 'biomedical', 'multilingual')
                   or a full model name from HuggingFace
    """
    global sbert_model, _current_model_name, _embedding_cache
    
    if model_name in SBERT_MODELS:
        model_name = SBERT_MODELS[model_name]
    
    _current_model_name = model_name
    sbert_model = SentenceTransformer(model_name)
    # Clear cache when model changes
    _embedding_cache = {}


def get_embedding_dim() -> int:
    """Get the embedding dimension of the current model."""
    return MODEL_DIMENSIONS.get(_current_model_name, 384)


def clear_embedding_cache() -> None:
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable embedding caching."""
    global _cache_enabled
    _cache_enabled = enabled


def get_sbert_embedding(text: str, use_cache: bool = True) -> np.ndarray:
    """
    Compute SBERT embedding for a given text with caching.
    
    Args:
        text: Input text string
        use_cache: Whether to use cache (default: True)
        
    Returns:
        numpy array of embedding
    """
    global _embedding_cache
    
    if use_cache and _cache_enabled and text in _embedding_cache:
        return _embedding_cache[text]
    
    embedding = sbert_model.encode(text, convert_to_tensor=False)
    
    if use_cache and _cache_enabled:
        _embedding_cache[text] = embedding
    
    return embedding


def get_batch_embeddings(texts: List[str], use_cache: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for multiple texts in one batch (much faster).
    
    Args:
        texts: List of input text strings
        use_cache: Whether to use cache (default: True)
        
    Returns:
        Dictionary mapping texts to their embeddings
    """
    global _embedding_cache
    
    result = {}
    texts_to_encode = []
    
    # Check cache first
    for text in texts:
        if use_cache and _cache_enabled and text in _embedding_cache:
            result[text] = _embedding_cache[text]
        else:
            texts_to_encode.append(text)
    
    # Batch encode uncached texts
    if texts_to_encode:
        embeddings = sbert_model.encode(texts_to_encode, convert_to_tensor=False, 
                                         batch_size=32, show_progress_bar=False)
        for text, emb in zip(texts_to_encode, embeddings):
            result[text] = emb
            if use_cache and _cache_enabled:
                _embedding_cache[text] = emb
    
    return result


def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Cosine similarity score
    """
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def build_directed_graph(triplets: List[List[str]]) -> nx.DiGraph:
    """
    Step 1: Graph Construction
    
    Construct a directed graph from triplets where:
    - Entities (subjects and objects) become nodes
    - Relations become labeled directed edges
    
    Args:
        triplets: List of [subject, predicate, object] triplets
        
    Returns:
        NetworkX DiGraph with labeled edges
    """
    G = nx.DiGraph()
    
    for triplet in triplets:
        if len(triplet) != 3:
            continue
        subject, predicate, obj = triplet
        
        # Add nodes
        G.add_node(subject)
        G.add_node(obj)
        
        # Add directed edge with relation label
        G.add_edge(subject, obj, relation=predicate)
    
    return G


def initialize_node_labels(G: nx.DiGraph) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Step 2: Initial Node Labels
    
    Assign each node an initial label equal to its entity name
    and compute SBERT embedding for each node label using batch processing.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Tuple of (node_labels dict, node_embeddings dict)
    """
    node_labels = {node: str(node) for node in G.nodes()}
    
    # Batch compute all node embeddings at once
    texts = list(node_labels.values())
    embeddings_dict = get_batch_embeddings(texts)
    
    node_embeddings = {node: embeddings_dict[label] for node, label in node_labels.items()}
    
    return node_labels, node_embeddings


def get_neighbor_descriptions(G: nx.DiGraph, node: str, node_labels: Dict[str, str]) -> List[str]:
    """
    Collect semantic descriptions for all neighbors of a node.
    
    For outgoing edges: "current_label --relation--> neighbor_label"
    For incoming edges: "neighbor_label --relation--> current_label"
    
    Args:
        G: NetworkX DiGraph
        node: Current node
        node_labels: Dictionary mapping nodes to their current labels
        
    Returns:
        Sorted list of semantic neighbor descriptions
    """
    descriptions = []
    current_label = node_labels[node]
    
    # Outgoing edges: node -> neighbor
    for _, neighbor, data in G.out_edges(node, data=True):
        relation = data.get('relation', 'related_to')
        neighbor_label = node_labels[neighbor]
        desc = f"{current_label} --{relation}--> {neighbor_label}"
        descriptions.append(desc)
    
    # Incoming edges: neighbor -> node
    for neighbor, _, data in G.in_edges(node, data=True):
        relation = data.get('relation', 'related_to')
        neighbor_label = node_labels[neighbor]
        desc = f"{neighbor_label} --{relation}--> {current_label}"
        descriptions.append(desc)
    
    # Sort for deterministic ordering
    return sorted(descriptions)


def wl_relabel_iteration(G: nx.DiGraph, 
                         node_labels: Dict[str, str], 
                         node_embeddings: Dict[str, np.ndarray],
                         use_weighted_aggregation: bool = False) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Step 3: WL Relabeling Iteration (Semantic Relabeling)
    
    Update each node's label by incorporating its neighbors and relation semantics.
    
    Args:
        G: NetworkX DiGraph
        node_labels: Current node labels
        node_embeddings: Current node embeddings
        use_weighted_aggregation: Use embedding-based weighted aggregation (default: False)
        
    Returns:
        Tuple of (new_node_labels, new_node_embeddings)
    """
    new_labels = {}
    
    # First pass: compute all new labels
    for node in G.nodes():
        current_label = node_labels[node]
        neighbor_descs = get_neighbor_descriptions(G, node, node_labels)
        
        if neighbor_descs:
            new_label = current_label + " | " + " | ".join(neighbor_descs)
        else:
            new_label = current_label
        
        new_labels[node] = new_label
    
    # Batch compute all new embeddings at once
    texts = list(new_labels.values())
    embeddings_dict = get_batch_embeddings(texts)
    
    if use_weighted_aggregation:
        # Use weighted aggregation approach
        new_embeddings = wl_weighted_aggregation(G, node_labels, node_embeddings)
    else:
        # Standard approach: embedding of concatenated label
        new_embeddings = {node: embeddings_dict[label] for node, label in new_labels.items()}
    
    return new_labels, new_embeddings


def wl_weighted_aggregation(G: nx.DiGraph,
                            node_labels: Dict[str, str],
                            node_embeddings: Dict[str, np.ndarray],
                            self_weight: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Weighted neighbor aggregation for WL relabeling.
    
    Instead of encoding concatenated strings, aggregate embeddings directly
    with weights based on relations.
    
    Args:
        G: NetworkX DiGraph
        node_labels: Current node labels
        node_embeddings: Current node embeddings
        self_weight: Weight for the node's own embedding (default: 0.5)
        
    Returns:
        Dictionary mapping nodes to new aggregated embeddings
    """
    new_embeddings = {}
    
    # Collect all relations to batch encode
    all_relations = set()
    for _, _, data in G.edges(data=True):
        all_relations.add(data.get('relation', 'related_to'))
    relation_embeddings = get_batch_embeddings(list(all_relations))
    
    for node in G.nodes():
        current_emb = node_embeddings[node]
        neighbor_embs = []
        
        # Outgoing edges
        for _, neighbor, data in G.out_edges(node, data=True):
            relation = data.get('relation', 'related_to')
            relation_emb = relation_embeddings[relation]
            neighbor_emb = node_embeddings[neighbor]
            # Combine relation and neighbor embeddings
            combined = (relation_emb + neighbor_emb) / 2
            neighbor_embs.append(combined)
        
        # Incoming edges
        for neighbor, _, data in G.in_edges(node, data=True):
            relation = data.get('relation', 'related_to')
            relation_emb = relation_embeddings[relation]
            neighbor_emb = node_embeddings[neighbor]
            combined = (relation_emb + neighbor_emb) / 2
            neighbor_embs.append(combined)
        
        if neighbor_embs:
            neighbor_agg = np.mean(neighbor_embs, axis=0)
            new_embeddings[node] = self_weight * current_emb + (1 - self_weight) * neighbor_agg
        else:
            new_embeddings[node] = current_emb
    
    return new_embeddings


def compute_node_groups(node_embeddings: Dict[str, np.ndarray], 
                        similarity_threshold: float = 0.90) -> Dict[str, int]:
    """
    Step 4: Node Grouping
    
    Group nodes based on embedding similarity using hierarchical clustering.
    Two nodes belong to the same group if their cosine similarity >= threshold.
    
    Args:
        node_embeddings: Dictionary mapping nodes to embeddings
        similarity_threshold: Minimum cosine similarity to be in same group
        
    Returns:
        Dictionary mapping nodes to group IDs
    """
    nodes = list(node_embeddings.keys())
    
    if len(nodes) == 0:
        return {}
    
    if len(nodes) == 1:
        return {nodes[0]: 0}
    
    # Stack embeddings into matrix
    embeddings = np.vstack([node_embeddings[node] for node in nodes])
    
    # Convert similarity threshold to distance threshold
    # cosine_distance = 1 - cosine_similarity
    distance_threshold = 1 - similarity_threshold
    
    # Use Agglomerative Clustering with cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )
    
    cluster_labels = clustering.fit_predict(embeddings)
    
    return {node: int(label) for node, label in zip(nodes, cluster_labels)}


def groups_are_stable(prev_groups: Dict[str, int], 
                      curr_groups: Dict[str, int]) -> bool:
    """
    Check if node group structure has stabilized.
    
    Two groupings are considered stable if nodes that were in the same group
    in the previous iteration are still in the same group in the current iteration.
    
    Args:
        prev_groups: Previous iteration's group assignments
        curr_groups: Current iteration's group assignments
        
    Returns:
        True if group structure is stable
    """
    if set(prev_groups.keys()) != set(curr_groups.keys()):
        return False
    
    # Create partition representation
    def get_partition(groups: Dict[str, int]) -> Set[frozenset]:
        partition = {}
        for node, group_id in groups.items():
            if group_id not in partition:
                partition[group_id] = set()
            partition[group_id].add(node)
        return {frozenset(s) for s in partition.values()}
    
    prev_partition = get_partition(prev_groups)
    curr_partition = get_partition(curr_groups)
    
    return prev_partition == curr_partition


def semantic_wl_iterations(G: nx.DiGraph, 
                           max_iterations: int = 5,
                           similarity_threshold: float = 0.90,
                           use_weighted_aggregation: bool = False,
                           verbose: bool = False) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Run Semantic WL iterations until convergence or max iterations.
    
    Args:
        G: NetworkX DiGraph
        max_iterations: Maximum number of iterations (safety limit)
        similarity_threshold: Threshold for node grouping
        use_weighted_aggregation: Use embedding-based weighted aggregation
        verbose: Print iteration details
        
    Returns:
        Tuple of (final_node_labels, final_node_embeddings)
    """
    if len(G.nodes()) == 0:
        return {}, {}
    
    # Step 2: Initialize node labels and embeddings
    node_labels, node_embeddings = initialize_node_labels(G)
    
    # Compute initial groups
    prev_groups = compute_node_groups(node_embeddings, similarity_threshold)
    
    if verbose:
        print(f"Initial node labels: {node_labels}")
        print(f"Initial groups: {prev_groups}")
    
    # Step 3 & 4: WL iterations with convergence check
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        # Perform relabeling
        node_labels, node_embeddings = wl_relabel_iteration(
            G, node_labels, node_embeddings, use_weighted_aggregation
        )
        
        # Compute new groups
        curr_groups = compute_node_groups(node_embeddings, similarity_threshold)
        
        if verbose:
            print(f"Node labels: {node_labels}")
            print(f"Groups: {curr_groups}")
        
        # Check stopping condition
        if groups_are_stable(prev_groups, curr_groups):
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break
        
        prev_groups = curr_groups
    
    return node_labels, node_embeddings


def compute_graph_embedding(node_embeddings: Dict[str, np.ndarray],
                            G: nx.DiGraph = None,
                            pooling: str = 'mean') -> np.ndarray:
    """
    Step 5: Graph Embedding Construction
    
    Compute graph embedding using various pooling strategies.
    
    Args:
        node_embeddings: Dictionary mapping nodes to embeddings
        G: NetworkX DiGraph (required for weighted pooling)
        pooling: Pooling strategy - 'mean', 'max', 'weighted', 'attention'
        
    Returns:
        Graph embedding vector
    """
    dim = get_embedding_dim()
    
    if len(node_embeddings) == 0:
        return np.zeros(dim)
    
    nodes = list(node_embeddings.keys())
    embeddings = np.vstack([node_embeddings[n] for n in nodes])
    
    if pooling == 'mean':
        return np.mean(embeddings, axis=0)
    
    elif pooling == 'max':
        return np.max(embeddings, axis=0)
    
    elif pooling == 'weighted' and G is not None:
        # Weight by degree centrality
        centrality = nx.degree_centrality(G)
        weights = np.array([centrality.get(node, 1.0) for node in nodes])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        return np.average(embeddings, axis=0, weights=weights)
    
    elif pooling == 'attention':
        # Self-attention pooling: query is mean, keys are node embeddings
        query = np.mean(embeddings, axis=0, keepdims=True)
        attention_scores = sklearn_cosine_similarity(query, embeddings)[0]
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
        return np.average(embeddings, axis=0, weights=attention_weights)
    
    else:
        # Default to mean
        return np.mean(embeddings, axis=0)


def compute_edge_embedding(G: nx.DiGraph) -> np.ndarray:
    """
    Compute embedding from relation predicates.
    
    Args:
        G: NetworkX DiGraph with 'relation' edge attributes
        
    Returns:
        Edge embedding vector (mean of all relation embeddings)
    """
    dim = get_embedding_dim()
    
    relations = [data.get('relation', 'related_to') for _, _, data in G.edges(data=True)]
    
    if not relations:
        return np.zeros(dim)
    
    relation_embeddings = get_batch_embeddings(relations)
    embeddings = np.vstack([relation_embeddings[r] for r in relations])
    
    return np.mean(embeddings, axis=0)


def compute_combined_graph_embedding(G: nx.DiGraph,
                                     node_embeddings: Dict[str, np.ndarray],
                                     node_weight: float = 0.7,
                                     pooling: str = 'weighted') -> np.ndarray:
    """
    Compute combined graph embedding from nodes and edges.
    
    Args:
        G: NetworkX DiGraph
        node_embeddings: Dictionary mapping nodes to embeddings
        node_weight: Weight for node embedding (edge weight = 1 - node_weight)
        pooling: Pooling strategy for node embeddings
        
    Returns:
        Combined graph embedding vector
    """
    node_emb = compute_graph_embedding(node_embeddings, G, pooling)
    edge_emb = compute_edge_embedding(G)
    
    return node_weight * node_emb + (1 - node_weight) * edge_emb


def compute_cross_graph_alignment_similarity(emb1_dict: Dict[str, np.ndarray],
                                             emb2_dict: Dict[str, np.ndarray],
                                             method: str = 'hungarian') -> float:
    """
    Compute similarity with optimal node alignment between graphs.
    
    Args:
        emb1_dict: Node embeddings for graph 1
        emb2_dict: Node embeddings for graph 2
        method: 'hungarian' for optimal matching, 'greedy' for fast matching
        
    Returns:
        Alignment-based similarity score
    """
    if len(emb1_dict) == 0 or len(emb2_dict) == 0:
        return 0.0
    
    nodes1, nodes2 = list(emb1_dict.keys()), list(emb2_dict.keys())
    emb1 = np.vstack([emb1_dict[n] for n in nodes1])
    emb2 = np.vstack([emb2_dict[n] for n in nodes2])
    
    # Compute pairwise similarity matrix
    sim_matrix = sklearn_cosine_similarity(emb1, emb2)
    
    if method == 'hungarian':
        # Optimal assignment using Hungarian algorithm
        # Convert to cost matrix (1 - similarity)
        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_sims = sim_matrix[row_ind, col_ind]
        return float(np.mean(matched_sims))
    
    elif method == 'greedy':
        # Greedy matching: each node in smaller graph matches its best counterpart
        matched_sims = []
        used = set()
        
        # Match from smaller to larger graph
        if len(nodes1) <= len(nodes2):
            for i in range(len(nodes1)):
                # Find best unused match
                for j in np.argsort(-sim_matrix[i]):
                    if j not in used:
                        matched_sims.append(sim_matrix[i, j])
                        used.add(j)
                        break
        else:
            for j in range(len(nodes2)):
                for i in np.argsort(-sim_matrix[:, j]):
                    if i not in used:
                        matched_sims.append(sim_matrix[i, j])
                        used.add(i)
                        break
        
        return float(np.mean(matched_sims)) if matched_sims else 0.0
    
    else:
        # Default: max similarity per node in graph1
        return float(np.mean(np.max(sim_matrix, axis=1)))


def semantic_wl_kernel_similarity(kg1_triplets: List[List[str]], 
                                   kg2_triplets: List[List[str]],
                                   max_iterations: int = 5,
                                   similarity_threshold: float = 0.90,
                                   pooling: str = 'weighted',
                                   include_edges: bool = True,
                                   node_weight: float = 0.7,
                                   use_weighted_aggregation: bool = False,
                                   use_cross_graph_alignment: bool = False,
                                   alignment_method: str = 'hungarian',
                                   similarity_combination: str = 'mean',
                                   verbose: bool = False) -> Dict:
    """
    Compute Semantic WL Kernel similarity between two knowledge graphs.
    
    This function implements the full pipeline:
    1. Graph Construction
    2. Initial Node Labels
    3. WL Relabeling Iterations
    4. Node Grouping with Stopping Condition
    5. Graph Embedding Construction
    6. Graph Similarity via Cosine Similarity
    
    Args:
        kg1_triplets: First knowledge graph as list of triplets
        kg2_triplets: Second knowledge graph as list of triplets
        max_iterations: Maximum WL iterations (default: 5)
        similarity_threshold: Node grouping threshold (default: 0.90)
        pooling: Node pooling strategy - 'mean', 'max', 'weighted', 'attention'
        include_edges: Include edge embeddings in graph embedding (default: True)
        node_weight: Weight for node vs edge embeddings when include_edges=True
        use_weighted_aggregation: Use embedding-based weighted aggregation in WL
        use_cross_graph_alignment: Use cross-graph node alignment for similarity
        alignment_method: 'hungarian' or 'greedy' for cross-graph alignment
        similarity_combination: 'mean', 'max', or 'weighted' when using both methods
        verbose: Print detailed output
        
    Returns:
        Dictionary containing:
        - similarity: Final cosine similarity score
        - graph1_labels: Final node labels for graph 1
        - graph2_labels: Final node labels for graph 2
        - graph1_embedding: Graph 1 embedding vector
        - graph2_embedding: Graph 2 embedding vector
        - alignment_similarity: (if use_cross_graph_alignment) Node alignment similarity
    """
    # Filter valid triplets
    kg1_triplets = [t for t in kg1_triplets if len(t) == 3]
    kg2_triplets = [t for t in kg2_triplets if len(t) == 3]
    
    dim = get_embedding_dim()
    
    if verbose:
        print("=" * 60)
        print("SEMANTIC WEISFEILER-LEHMAN KERNEL (Enhanced)")
        print("=" * 60)
        print(f"Model: {_current_model_name} ({dim} dims)")
        print(f"Pooling: {pooling}, Include edges: {include_edges}")
        print(f"Weighted aggregation: {use_weighted_aggregation}")
        print(f"Cross-graph alignment: {use_cross_graph_alignment}")
    
    # Handle edge cases
    if len(kg1_triplets) == 0 or len(kg2_triplets) == 0:
        result = {
            'similarity': 0.0,
            'graph1_labels': {},
            'graph2_labels': {},
            'graph1_embedding': np.zeros(dim),
            'graph2_embedding': np.zeros(dim)
        }
        if use_cross_graph_alignment:
            result['alignment_similarity'] = 0.0
        return result
    
    # Step 1: Build directed graphs
    if verbose:
        print("\n[Step 1] Building directed graphs...")
    G1 = build_directed_graph(kg1_triplets)
    G2 = build_directed_graph(kg2_triplets)
    
    if verbose:
        print(f"Graph 1: {len(G1.nodes())} nodes, {len(G1.edges())} edges")
        print(f"Graph 2: {len(G2.nodes())} nodes, {len(G2.edges())} edges")
    
    # Steps 2-4: Run Semantic WL iterations
    if verbose:
        print("\n[Steps 2-4] Running Semantic WL iterations for Graph 1...")
    labels1, embeddings1 = semantic_wl_iterations(
        G1, max_iterations, similarity_threshold, use_weighted_aggregation, verbose
    )
    
    if verbose:
        print("\n[Steps 2-4] Running Semantic WL iterations for Graph 2...")
    labels2, embeddings2 = semantic_wl_iterations(
        G2, max_iterations, similarity_threshold, use_weighted_aggregation, verbose
    )
    
    # Step 5: Compute graph embeddings (with optional edge inclusion)
    if verbose:
        print("\n[Step 5] Computing graph embeddings...")
    
    if include_edges:
        graph1_embedding = compute_combined_graph_embedding(G1, embeddings1, node_weight, pooling)
        graph2_embedding = compute_combined_graph_embedding(G2, embeddings2, node_weight, pooling)
    else:
        graph1_embedding = compute_graph_embedding(embeddings1, G1, pooling)
        graph2_embedding = compute_graph_embedding(embeddings2, G2, pooling)
    
    # Step 6: Compute similarity
    if verbose:
        print("\n[Step 6] Computing similarity...")
    
    # Base cosine similarity
    embedding_similarity = cosine_sim(graph1_embedding, graph2_embedding)
    
    # Optional: Cross-graph node alignment similarity
    alignment_similarity = None
    if use_cross_graph_alignment:
        if verbose:
            print(f"  Computing cross-graph alignment ({alignment_method})...")
        alignment_similarity = compute_cross_graph_alignment_similarity(
            embeddings1, embeddings2, alignment_method
        )
        
        # Combine similarities
        if similarity_combination == 'mean':
            similarity = (embedding_similarity + alignment_similarity) / 2
        elif similarity_combination == 'max':
            similarity = max(embedding_similarity, alignment_similarity)
        elif similarity_combination == 'weighted':
            # Weight embedding similarity more (0.6) since it captures global structure
            similarity = 0.6 * embedding_similarity + 0.4 * alignment_similarity
        else:
            similarity = embedding_similarity
    else:
        similarity = embedding_similarity
    
    # Step 7: Output
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print("\nFinal Node Labels for Graph 1:")
        for node, label in labels1.items():
            print(f"  {node}: {label[:100]}..." if len(label) > 100 else f"  {node}: {label}")
        
        print("\nFinal Node Labels for Graph 2:")
        for node, label in labels2.items():
            print(f"  {node}: {label[:100]}..." if len(label) > 100 else f"  {node}: {label}")
        
        print(f"\nGraph 1 Embedding (first 10 dims): {graph1_embedding[:10]}")
        print(f"Graph 2 Embedding (first 10 dims): {graph2_embedding[:10]}")
        print(f"\nEmbedding Cosine Similarity: {embedding_similarity:.6f}")
        if alignment_similarity is not None:
            print(f"Alignment Similarity: {alignment_similarity:.6f}")
        print(f"Final Combined Similarity: {similarity:.6f}")
    
    result = {
        'similarity': similarity,
        'embedding_similarity': embedding_similarity,
        'graph1_labels': labels1,
        'graph2_labels': labels2,
        'graph1_embedding': graph1_embedding,
        'graph2_embedding': graph2_embedding
    }
    
    if alignment_similarity is not None:
        result['alignment_similarity'] = alignment_similarity
    
    return result


# Convenience function matching the existing API pattern
def calculate_semantic_wl_similarity(kg1_triplets: List[List[str]], 
                                      kg2_triplets: List[List[str]],
                                      verbose: bool = False,
                                      enhanced: bool = True) -> float:
    """
    Calculate semantic WL similarity between two knowledge graphs.
    
    This is a convenience function that returns only the similarity score.
    
    Args:
        kg1_triplets: First knowledge graph as list of triplets
        kg2_triplets: Second knowledge graph as list of triplets
        verbose: Print detailed output
        enhanced: Use all enhancements (weighted pooling, edges, alignment)
        
    Returns:
        Cosine similarity score (0-1)
    """
    if enhanced:
        result = semantic_wl_kernel_similarity(
            kg1_triplets, kg2_triplets,
            pooling='weighted',
            include_edges=True,
            use_weighted_aggregation=True,
            use_cross_graph_alignment=True,
            alignment_method='hungarian',
            similarity_combination='weighted',
            verbose=verbose
        )
    else:
        # Original behavior for backward compatibility
        result = semantic_wl_kernel_similarity(
            kg1_triplets, kg2_triplets,
            pooling='mean',
            include_edges=False,
            use_weighted_aggregation=False,
            use_cross_graph_alignment=False,
            verbose=verbose
        )
    return result['similarity']


# Preset configurations for easy use
def get_preset_config(preset: str = 'balanced') -> Dict:
    """
    Get preset configuration for the similarity computation.
    
    Available presets:
    - 'fast': Minimal computation, good for large datasets
    - 'balanced': Good balance of speed and accuracy (default)
    - 'accurate': Maximum accuracy, slower
    - 'legacy': Original implementation behavior
    
    Args:
        preset: Preset name
        
    Returns:
        Dictionary of configuration parameters
    """
    presets = {
        'fast': {
            'max_iterations': 3,
            'pooling': 'mean',
            'include_edges': False,
            'use_weighted_aggregation': False,
            'use_cross_graph_alignment': False
        },
        'balanced': {
            'max_iterations': 5,
            'pooling': 'weighted',
            'include_edges': True,
            'node_weight': 0.7,
            'use_weighted_aggregation': False,
            'use_cross_graph_alignment': True,
            'alignment_method': 'greedy',
            'similarity_combination': 'weighted'
        },
        'accurate': {
            'max_iterations': 7,
            'pooling': 'attention',
            'include_edges': True,
            'node_weight': 0.7,
            'use_weighted_aggregation': True,
            'use_cross_graph_alignment': True,
            'alignment_method': 'hungarian',
            'similarity_combination': 'weighted'
        },
        'legacy': {
            'max_iterations': 5,
            'pooling': 'mean',
            'include_edges': False,
            'use_weighted_aggregation': False,
            'use_cross_graph_alignment': False
        }
    }
    return presets.get(preset, presets['balanced'])


# Example usage and testing
if __name__ == "__main__":
    # Test knowledge graphs
    kg1 = [
        ['Herbert Blankenhorn', 'date of birth', '15 December 1904'],
        ['Herbert Blankenhorn', 'place of birth', 'Mülhausen'],
        ['Herbert Blankenhorn', 'date of death', '10 August 1991'],
        ['Herbert Blankenhorn', 'place of death', 'Badenweiler']
    ]
    
    kg2 = [
        ['Herbert Blankenhorn', 'born on', '15 December 1904'],
        ['Herbert Blankenhorn', 'birthplace', 'Mülhausen'],
        ['Herbert Blankenhorn', 'died on', '10 August 1991'],
        ['Herbert Blankenhorn', 'death place', 'Badenweiler']
    ]
    
    claim = [
        ["Marie Curie", "discovered", "Radium"],
        ["Marie Curie", "won", "Nobel Prize in Physics"],
        ["Marie Curie", "won", "Nobel Prize in Chemistry"]
    ]
    
    evidence = [
        ["Marie Curie", "found", "Radium"],
        ["Marie Curie", "received", "Nobel Prize in Physics"],
        ["Marie Curie", "was awarded", "Nobel Prize in Chemistry"]
    ]
    
    incorrect_evidence = [
        ["Albert Einstein", "found", "Radium"],
        ["Marie Curie", "received", "Grammy"],
        ["Marie Curie", "was awarded", "Emmy"]
    ]
    
    apples1 = [["Apples", "type of", "fruit"], ["Apples", "grow on", "trees"]]
    apples2 = [["Apples", "type of", "fruit"], ["Apples", "grow in", "tree"]]
    
    # Test with different presets
    print("\n" + "=" * 80)
    print("COMPARING DIFFERENT CONFIGURATIONS")
    print("=" * 80)
    
    test_cases = [
        ("Herbert Blankenhorn", kg1, kg2),
        ("Marie Curie (correct)", claim, evidence),
        ("Marie Curie (incorrect)", claim, incorrect_evidence),
        ("Apples", apples1, apples2)
    ]
    
    presets_to_test = ['legacy', 'fast', 'balanced', 'accurate']
    
    print(f"\n{'Test Case':<25} | {'Legacy':>10} | {'Fast':>10} | {'Balanced':>10} | {'Accurate':>10}")
    print("-" * 80)
    
    for name, kg_a, kg_b in test_cases:
        scores = []
        for preset in presets_to_test:
            config = get_preset_config(preset)
            result = semantic_wl_kernel_similarity(kg_a, kg_b, **config, verbose=False)
            scores.append(result['similarity'])
        print(f"{name:<25} | {scores[0]:>10.4f} | {scores[1]:>10.4f} | {scores[2]:>10.4f} | {scores[3]:>10.4f}")
    
    # Detailed verbose test with enhanced configuration
    print("\n" + "=" * 80)
    print("DETAILED TEST: Marie Curie (Enhanced Configuration)")
    print("=" * 80)
    
    result = semantic_wl_kernel_similarity(
        claim, evidence,
        pooling='weighted',
        include_edges=True,
        use_weighted_aggregation=True,
        use_cross_graph_alignment=True,
        alignment_method='hungarian',
        similarity_combination='weighted',
        verbose=True
    )
    
    # Test convenience function
    print("\n" + "=" * 80)
    print("CONVENIENCE FUNCTION TEST")
    print("=" * 80)
    
    print(f"\nLegacy mode: {calculate_semantic_wl_similarity(claim, evidence, enhanced=False):.4f}")
    print(f"Enhanced mode: {calculate_semantic_wl_similarity(claim, evidence, enhanced=True):.4f}")
    
    # Cache statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE: Embedding Cache Statistics")
    print("=" * 80)
    print(f"Cached embeddings: {len(_embedding_cache)}")
    print(f"Current model: {_current_model_name} ({get_embedding_dim()} dims)")
