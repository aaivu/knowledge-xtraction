"""
Semantic Weisfeiler-Lehman (WL) Kernel for Knowledge Graph Similarity

This module implements a semantic variant of the WL kernel that uses SBERT embeddings
to compute similarity between knowledge graphs represented as triplet lists.
"""

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from typing import List, Tuple, Dict, Set, Optional

# Initialize SBERT model as specified
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_sbert_embedding(text: str) -> np.ndarray:
    """
    Compute SBERT embedding for a given text.
    
    Args:
        text: Input text string
        
    Returns:
        numpy array of embedding
    """
    embedding = sbert_model.encode(text, convert_to_tensor=True)
    return embedding.detach().cpu().numpy()


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
    and compute SBERT embedding for each node label.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Tuple of (node_labels dict, node_embeddings dict)
    """
    node_labels = {}
    node_embeddings = {}
    
    for node in G.nodes():
        node_labels[node] = str(node)
        node_embeddings[node] = get_sbert_embedding(str(node))
    
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
                         node_embeddings: Dict[str, np.ndarray]) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Step 3: WL Relabeling Iteration (Semantic Relabeling)
    
    Update each node's label by incorporating its neighbors and relation semantics.
    
    Args:
        G: NetworkX DiGraph
        node_labels: Current node labels
        node_embeddings: Current node embeddings
        
    Returns:
        Tuple of (new_node_labels, new_node_embeddings)
    """
    new_labels = {}
    new_embeddings = {}
    
    for node in G.nodes():
        current_label = node_labels[node]
        
        # Get sorted neighbor descriptions
        neighbor_descs = get_neighbor_descriptions(G, node, node_labels)
        
        # Construct new label
        if neighbor_descs:
            new_label = current_label + " | " + " | ".join(neighbor_descs)
        else:
            new_label = current_label
        
        new_labels[node] = new_label
        new_embeddings[node] = get_sbert_embedding(new_label)
    
    return new_labels, new_embeddings


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
                           verbose: bool = False) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Run Semantic WL iterations until convergence or max iterations.
    
    Args:
        G: NetworkX DiGraph
        max_iterations: Maximum number of iterations (safety limit)
        similarity_threshold: Threshold for node grouping
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
        node_labels, node_embeddings = wl_relabel_iteration(G, node_labels, node_embeddings)
        
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


def compute_graph_embedding(node_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Step 5: Graph Embedding Construction
    
    Compute graph embedding as the mean of all node embeddings.
    
    Args:
        node_embeddings: Dictionary mapping nodes to embeddings
        
    Returns:
        Graph embedding vector (mean pooling)
    """
    if len(node_embeddings) == 0:
        return np.zeros(384)  # all-MiniLM-L6-v2 has 384 dimensions
    
    embeddings = np.vstack(list(node_embeddings.values()))
    return np.mean(embeddings, axis=0)


def semantic_wl_kernel_similarity(kg1_triplets: List[List[str]], 
                                   kg2_triplets: List[List[str]],
                                   max_iterations: int = 5,
                                   similarity_threshold: float = 0.90,
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
        verbose: Print detailed output
        
    Returns:
        Dictionary containing:
        - similarity: Final cosine similarity score
        - graph1_labels: Final node labels for graph 1
        - graph2_labels: Final node labels for graph 2
        - graph1_embedding: Graph 1 embedding vector
        - graph2_embedding: Graph 2 embedding vector
    """
    # Filter valid triplets
    kg1_triplets = [t for t in kg1_triplets if len(t) == 3]
    kg2_triplets = [t for t in kg2_triplets if len(t) == 3]
    
    if verbose:
        print("=" * 60)
        print("SEMANTIC WEISFEILER-LEHMAN KERNEL")
        print("=" * 60)
    
    # Handle edge cases
    if len(kg1_triplets) == 0 or len(kg2_triplets) == 0:
        return {
            'similarity': 0.0,
            'graph1_labels': {},
            'graph2_labels': {},
            'graph1_embedding': np.zeros(384),
            'graph2_embedding': np.zeros(384)
        }
    
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
        G1, max_iterations, similarity_threshold, verbose
    )
    
    if verbose:
        print("\n[Steps 2-4] Running Semantic WL iterations for Graph 2...")
    labels2, embeddings2 = semantic_wl_iterations(
        G2, max_iterations, similarity_threshold, verbose
    )
    
    # Step 5: Compute graph embeddings
    if verbose:
        print("\n[Step 5] Computing graph embeddings...")
    graph1_embedding = compute_graph_embedding(embeddings1)
    graph2_embedding = compute_graph_embedding(embeddings2)
    
    # Step 6: Compute cosine similarity
    if verbose:
        print("\n[Step 6] Computing similarity...")
    similarity = cosine_sim(graph1_embedding, graph2_embedding)
    
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
        print(f"\nFinal Cosine Similarity Score: {similarity:.6f}")
    
    return {
        'similarity': similarity,
        'graph1_labels': labels1,
        'graph2_labels': labels2,
        'graph1_embedding': graph1_embedding,
        'graph2_embedding': graph2_embedding
    }


# Convenience function matching the existing API pattern
def calculate_semantic_wl_similarity(kg1_triplets: List[List[str]], 
                                      kg2_triplets: List[List[str]],
                                      verbose: bool = False) -> float:
    """
    Calculate semantic WL similarity between two knowledge graphs.
    
    This is a convenience function that returns only the similarity score.
    
    Args:
        kg1_triplets: First knowledge graph as list of triplets
        kg2_triplets: Second knowledge graph as list of triplets
        verbose: Print detailed output
        
    Returns:
        Cosine similarity score (0-1)
    """
    result = semantic_wl_kernel_similarity(kg1_triplets, kg2_triplets, verbose=verbose)
    return result['similarity']


# Example usage and testing
if __name__ == "__main__":
    # Test case 1: Similar graphs about Herbert Blankenhorn
    print("\n" + "=" * 80)
    print("TEST CASE 1: Herbert Blankenhorn Information")
    print("=" * 80)
    
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
    
    result = semantic_wl_kernel_similarity(kg1, kg2, verbose=True)
    print(f"\nSimilarity: {result['similarity']:.6f}")
    
    # Test case 2: Marie Curie
    print("\n" + "=" * 80)
    print("TEST CASE 2: Marie Curie - Claim vs Evidence")
    print("=" * 80)
    
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
    
    result = semantic_wl_kernel_similarity(claim, evidence, verbose=True)
    print(f"\nSimilarity: {result['similarity']:.6f}")
    
    # Test case 3: Dissimilar evidence
    print("\n" + "=" * 80)
    print("TEST CASE 3: Marie Curie - Claim vs Incorrect Evidence")
    print("=" * 80)
    
    incorrect_evidence = [
        ["Albert Einstein", "found", "Radium"],
        ["Marie Curie", "received", "Grammy"],
        ["Marie Curie", "was awarded", "Emmy"]
    ]
    
    result = semantic_wl_kernel_similarity(claim, incorrect_evidence, verbose=True)
    print(f"\nSimilarity: {result['similarity']:.6f}")
    
    # Test case 4: Apple example
    print("\n" + "=" * 80)
    print("TEST CASE 4: Apples - Nearly identical")
    print("=" * 80)
    
    apples1 = [["Apples", "type of", "fruit"], ["Apples", "grow on", "trees"]]
    apples2 = [["Apples", "type of", "fruit"], ["Apples", "grow in", "tree"]]
    
    result = semantic_wl_kernel_similarity(apples1, apples2, verbose=True)
    print(f"\nSimilarity: {result['similarity']:.6f}")
