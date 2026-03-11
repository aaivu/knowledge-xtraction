from kganalytica import (
    construct_graph,
    construct_graphs,
    construct_kgs,
    get_aligned_triplets,
    get_missing_triplets,
    kg_similarity,  
)

def test_imports():
    assert construct_graph is not None
    assert construct_graphs is not None
    assert construct_kgs is not None
    assert get_aligned_triplets is not None
    assert get_missing_triplets is not None
    assert kg_similarity is not None