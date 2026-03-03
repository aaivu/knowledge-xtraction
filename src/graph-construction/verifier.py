
"""Triplet verification against source text using LLMs."""

import logging
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

from schemas import Graph, Triplet, DynamicKnowledgeGraphs
from llms.llm_factory import LLMFactorySelector


@dataclass
class VerificationResult:
    """Result of verifying a single triplet."""
    triplet: Triplet
    is_verified: bool
    confidence: float
    reason: str = ""


class TripletVerifier:
    """Verifies extracted triplets against source text."""
    
    def __init__(
        self, 
        verification_model1: str, 
        verification_model2: str = None,  # Kept for backward compatibility
        template_path: str = "templates/verification_batch.txt"
    ):
        """
        Initialize the verifier with LLM for batch verification.
        
        Args:
            verification_model1: Name of the verification LLM
            verification_model2: Deprecated - kept for backward compatibility
            template_path: Path to the batch verification prompt template
        """
        self.model_name = verification_model1
        self.template_path = Path(template_path)
        
        self._llm = None
        
        self.template = self._load_template()
        logging.info(f"[Verifier] Initialized with: {verification_model1}")
    
    @property
    def llm(self):
        """Get LLM instance (lazy-loaded)."""
        if self._llm is None:
    
            self._llm = LLMFactorySelector.get_factory(self.model_name)
        return self._llm
    
    # Keep old property names for backward compatibility
    @property
    def llm1(self):
        return self.llm
    
    @property
    def llm2(self):
        return self.llm
    
    def _load_template(self) -> str:
        """Load the verification prompt template."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Verification template not found: {self.template_path}")
        
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _build_batch_prompt(self, text: str, triplets: List[Triplet]) -> str:
        """Build verification prompt for multiple triplets."""
        # Format triplets as numbered list
        triplet_lines = []
        for i, triplet in enumerate(triplets, 1):
            triplet_str = f'{i}. ["{triplet.subject}", "{triplet.relation}", "{triplet.object}"]'
            triplet_lines.append(triplet_str)
        
        triplets_section = "\n".join(triplet_lines)
        
        prompt = self.template.replace("{text}", text).replace("{triplets}", triplets_section)
        return prompt
    
    def _parse_batch_response(self, response: Dict, triplets: List[Triplet], default_confidence: float) -> List[VerificationResult]:
        """Parse batch verification response into VerificationResults."""
        results = []
        
        for i, triplet in enumerate(triplets, 1):
            key = str(i)
            
            if key in response:
                item = response[key]
                is_verified = item.get("verified", False)
                # Ensure boolean interpretation
                if isinstance(is_verified, str):
                    is_verified = is_verified.lower() in ("true", "yes", "1")
                reason = item.get("reason", "")
            else:
                logging.warning(f"Triplet {i} not in response")
                is_verified = False
                reason = "Not found in verification response"
            
            results.append(VerificationResult(
                triplet=triplet,
                is_verified=is_verified,
                confidence=default_confidence,
                reason=reason
            ))
        
        return results
    
    def verify_graph(self, text: str, graph: Graph) -> Tuple[Graph, List[VerificationResult]]:
        """Verify all triplets in a graph against source text."""
        if not graph.triples:
            return Graph(triples=[]), []
        
        # Build batch prompt
        prompt = self._build_batch_prompt(text, graph.triples)
        
        try:
            response_data, confidence = self.llm.batch_verify(prompt)
            results = self._parse_batch_response(response_data, graph.triples, confidence)
        except Exception as e:
            logging.error(f"Batch verification failed: {e}")
            results = [
                VerificationResult(
                    triplet=triplet,
                    is_verified=False,
                    confidence=0.0,
                    reason=f"Error: {str(e)}"
                )
                for triplet in graph.triples
            ]
        
        verified_triplets = [r.triplet for r in results if r.is_verified]
        verified_graph = Graph(triples=verified_triplets)
        
        return verified_graph, results
    
    def verify_all_graphs(
        self, 
        paragraphs: List[str], 
        kgs: DynamicKnowledgeGraphs
    ) -> Tuple[DynamicKnowledgeGraphs, Dict[str, List[VerificationResult]]]:
        """Verify all graphs against their corresponding paragraphs."""
        verified_graphs = {}
        all_results = {}
        
        for i, para in enumerate(paragraphs, 1):
            graph_key = f"graph_{i}"
            graph = kgs.get_graph(i)
            
            verified_graph, results = self.verify_graph(para, graph)
            verified_graphs[graph_key] = verified_graph
            all_results[graph_key] = results
        
        return DynamicKnowledgeGraphs(graphs=verified_graphs), all_results
    
    def has_rejected_triplets(self, results: Dict[str, List[VerificationResult]]) -> bool:
        """Check if any triplets were rejected across all graphs."""
        for graph_results in results.values():
            for result in graph_results:
                if not result.is_verified:
                    return True
        return False
    
    def get_rejection_summary(self, results: Dict[str, List[VerificationResult]]) -> Dict[str, int]:
        """Get count of rejected triplets per graph."""
        summary = {}
        for graph_key, graph_results in results.items():
            rejected = sum(1 for r in graph_results if not r.is_verified)
            summary[graph_key] = rejected
        return summary
    
    def get_failed_triplets(self, results: Dict[str, List[VerificationResult]]) -> Dict[str, List[List[str]]]:
        """Extract failed triplets per graph."""
        failed = {}
        for graph_key, graph_results in results.items():
            failed_list = []
            for result in graph_results:
                if not result.is_verified:
                    triplet = result.triplet
                    failed_list.append([triplet.subject, triplet.relation, triplet.object])
            if failed_list:
                failed[graph_key] = failed_list
        return failed
    
    def get_verified_triplets(self, results: Dict[str, List[VerificationResult]]) -> Dict[str, List[List[str]]]:
        """Extract verified triplets per graph."""
        verified = {}
        for graph_key, graph_results in results.items():
            verified_list = []
            for result in graph_results:
                if result.is_verified:
                    triplet = result.triplet
                    verified_list.append([triplet.subject, triplet.relation, triplet.object])
            if verified_list:
                verified[graph_key] = verified_list
        return verified
    
    def clear(self):
        """Release resources."""
        if self._llm is not None and hasattr(self._llm, 'clear_model'):
            self._llm.clear_model()
        self._llm = None
