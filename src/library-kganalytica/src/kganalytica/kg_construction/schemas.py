from typing import List, Any, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


class Triplet(BaseModel):
    subject: str = Field(..., description="The initiating entity or event")
    relation: str = Field(..., description="The standardized relationship label")
    object: str = Field(..., description="The target entity, attribute, or value")


class Graph(BaseModel):
    triples: List[Triplet] = Field(default_factory=list, description="List of triples [subject, relation, object]")

    @field_validator('triples', mode='before')
    @classmethod
    def parse_triples_list(cls, v: Any) -> Any:
        # Handle case where LLM outputs lists instead of dicts (common with prompt examples)
        if isinstance(v, list):
            parsed_triples = []
            for item in v:
                if isinstance(item, list):
                    if len(item) >= 3:
                        parsed_triples.append({
                            'subject': str(item[0]), 
                            'relation': str(item[1]), 
                            'object': str(item[2])
                        })
                    else:
                        # Skip malformed triples
                        continue
                else:
                    parsed_triples.append(item)
            return parsed_triples
        return v


class DynamicKnowledgeGraphs(BaseModel):
    """Dynamic knowledge graphs container supporting any number of graphs."""
    graphs: Dict[str, Graph] = Field(default_factory=dict, description="Dictionary of graphs keyed by graph_1, graph_2, etc.")
    
    @model_validator(mode='before')
    @classmethod
    def parse_graph_keys(cls, values: Any) -> Any:
        """Parse graph_1, graph_2, etc. into the graphs dictionary."""
        if isinstance(values, dict):
            graphs = {}
            other_fields = {}
            for key, value in values.items():
                if key.startswith('graph_') and key[6:].isdigit():
                    if isinstance(value, dict):
                        graphs[key] = Graph(**value)
                    elif isinstance(value, Graph):
                        graphs[key] = value
                    else:
                        graphs[key] = Graph(triples=[])
                else:
                    other_fields[key] = value
            if graphs:
                other_fields['graphs'] = graphs
            return other_fields
        return values
    
    def get_graph(self, index: int) -> Graph:
        """Get graph by index (1-based)."""
        return self.graphs.get(f'graph_{index}', Graph(triples=[]))
    
    def set_graph(self, index: int, graph: Graph):
        """Set graph by index (1-based)."""
        self.graphs[f'graph_{index}'] = graph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {key: {"triples": [t.model_dump() for t in graph.triples]} 
                for key, graph in self.graphs.items()}
    
    @property
    def num_graphs(self) -> int:
        """Return number of graphs."""
        return len(self.graphs)


# Keep backwards compatibility
class KnowledgeGraphs(BaseModel):
    """Fixed 3-graph structure for backwards compatibility."""
    graph_1: Graph
    graph_2: Graph
    graph_3: Graph