"""Knowledge Graph extraction from text paragraphs using LLMs."""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from schemas import Graph, DynamicKnowledgeGraphs, Triplet
from llms.llm_factory import LLMFactorySelector


class KGExtractor:
    """Extracts Knowledge Graphs from text paragraphs using LLMs."""
    
    def __init__(self, model_name: str, template_path: str = "templates/extraction.txt",
                 num_triplets: Optional[int] = None, temperature: float = 0.0):
        """
        Initialize the extractor.
        
        Args:
            model_name: Name of the LLM to use for extraction
            template_path: Path to the extraction prompt template
            num_triplets: Target number of triplets per graph (if None, LLM decides freely)
            temperature: Temperature for LLM sampling (0.0-1.0)
        """
        self.model_name = model_name
        self.num_triplets = num_triplets
        self.temperature = temperature
        self.llm = LLMFactorySelector.get_factory(model_name)
        self.template = self._load_template(template_path)

    
    def _load_template(self, template_path: Optional[str] = None) -> str:
        """Load the extraction prompt template."""
        if template_path is None:
            raise ValueError("Template path must be provided")
        
        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Extraction template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _build_prompt(
        self, 
        paragraphs: List[str], 
        failed_triplets: Optional[Dict[str, List[List[str]]]] = None,
        approved_triplets: Optional[Dict[str, List[List[str]]]] = None
    ) -> str:
        """Build extraction prompt with feedback from previous attempts."""
        n = len(paragraphs)
        
        text_parts = []
        for i, para in enumerate(paragraphs, 1):
            text_parts.append(f"TEXT {i}:\n{para}")
        
        texts_section = "\n\n".join(text_parts)
        
        prompt = self.template.replace("{n}", str(n))
        if self.num_triplets is not None:
            prompt = prompt.replace("{num_triplets}", str(self.num_triplets))
        prompt += f"\n{texts_section}\n\n"
        
        if self.num_triplets is not None:
            prompt += f"Extract approximately {self.num_triplets} triplets per text.\n\n"
        
        # Add section for approved triplets that must be included
        if approved_triplets:
            prompt += "================ APPROVED TRIPLETS (MUST INCLUDE) ==================\n\n"
            prompt += "The following triplets have been VERIFIED and MUST be included. Include these exactly:\n\n"
            for graph_key, triplets in approved_triplets.items():
                graph_num = graph_key.replace("graph_", "")
                prompt += f"Approved triplets for TEXT {graph_num} (include these):\n"
                for triplet in triplets:
                    prompt += f"  + {triplet}\n"
            prompt += "\n"
        
        # Add cautionary section for failed triplets if provided
        if failed_triplets:
            prompt += "================ FORBIDDEN TRIPLETS (DO NOT USE) ==================\n\n"
            prompt += "The following triplets FAILED verification. Do NOT include these:\n\n"
            for graph_key, triplets in failed_triplets.items():
                graph_num = graph_key.replace("graph_", "")
                prompt += f"Forbidden triplets for TEXT {graph_num} (do not use):\n"
                for triplet in triplets:
                    prompt += f"  - {triplet}\n"
            prompt += "\n"
        
        prompt += "Output the graphs:"
        
        return prompt
    
    def _parse_triplets_array(self, content: str) -> Dict[str, List]:
        """Parse triplets array string into structured format."""
        content = content.strip()
        
        # Handle empty content
        if not content or content == '[]':
            return {"triples": []}
        
        # Find the array in the content
        # Look for [[...]] pattern
        array_match = re.search(r'(\[.*\])', content, re.DOTALL)
        if not array_match:
            return {"triples": []}
        
        array_str = array_match.group(1)
        
        try:
            # Replace single quotes with double quotes for JSON parsing
            array_str = array_str.replace("'", '"')
            triplets = json.loads(array_str)
            
            # Validate it's a list of lists/triplets
            if isinstance(triplets, list):
                valid_triplets = []
                for t in triplets:
                    if isinstance(t, list) and len(t) >= 3:
                        valid_triplets.append(t[:3])  # Take first 3 elements
                return {"triples": valid_triplets}
        except json.JSONDecodeError:
            pass
        
        return {"triples": []}
    
    def _parse_response(self, response: str, num_graphs: int) -> DynamicKnowledgeGraphs:
        """Parse LLM response into DynamicKnowledgeGraphs."""
        data = {}
        
        # Try simple format first: graph_N: [[...], ...]
        # Use a more robust approach - find graph_N: then capture until next graph_ or end
        lines = response.split('\n')
        current_graph = None
        current_content = []
        
        for line in lines:
            # Check if this line starts a new graph
            graph_match = re.match(r'graph_(\d+)\s*:', line)
            if graph_match:
                # Save previous graph if exists
                if current_graph is not None:
                    data[current_graph] = self._parse_triplets_array(''.join(current_content))
                # Start new graph
                current_graph = f"graph_{graph_match.group(1)}"
                # Content starts after "graph_N:"
                remaining = line.split(':', 1)[1] if ':' in line else ''
                current_content = [remaining]
            elif current_graph is not None:
                current_content.append(line)
        
        # Save last graph
        if current_graph is not None:
            data[current_graph] = self._parse_triplets_array(''.join(current_content))
        
        # If simple format didn't work, try JSON format
        if not data:
            # Fallback to JSON format
            json_str = response
            
            # Handle markdown code blocks
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Try to find raw JSON object
                match = re.search(r"(\{.*\})", response, re.DOTALL)
                if match:
                    json_str = match.group(1)
            
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                data = {}
        
        for i in range(1, num_graphs + 1):
            graph_key = f"graph_{i}"
            if graph_key not in data:
                data[graph_key] = {"triples": []}
        
        return DynamicKnowledgeGraphs(**data)
    
    def extract(self, paragraphs: List[str], failed_triplets: Optional[Dict[str, List[List[str]]]] = None,
        approved_triplets: Optional[Dict[str, List[List[str]]]] = None) -> DynamicKnowledgeGraphs:
        """Extract knowledge graphs from paragraphs."""
        if not paragraphs:
            return DynamicKnowledgeGraphs(graphs={})
        
        num_graphs = len(paragraphs)
        prompt = self._build_prompt(paragraphs, failed_triplets, approved_triplets)
        
        try:
            response = self.llm.get_answer(prompt)
            kgs = self._parse_response(response, num_graphs)
            return kgs
            
        except Exception as e:
            logging.error(f"Extraction failed: {e}")
            return DynamicKnowledgeGraphs(
                graphs={f"graph_{i}": Graph(triples=[]) for i in range(1, num_graphs + 1)}
            )
    
    def extract_single(self, paragraph: str, graph_index: int = 1) -> Graph:
        """Extract single knowledge graph from one paragraph."""
        kgs = self.extract([paragraph])
        return kgs.get_graph(1)
    
    def clear(self):
        """Release resources."""
        if hasattr(self.llm, 'clear_model'):
            self.llm.clear_model()