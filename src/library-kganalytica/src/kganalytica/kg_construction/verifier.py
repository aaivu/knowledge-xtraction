"""
Triplet Verification Module.

This module verifies if extracted triplets are grounded in the source text.
Uses batch verification for efficiency - all triplets from a graph are verified
in a single LLM call, with confidence scores from token probabilities.
"""

import logging
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

from .schemas import Graph, Triplet, DynamicKnowledgeGraphs
from .llms.llm_factory import LLMFactorySelector


@dataclass
class VerificationResult:
    """Result of verifying a single triplet."""
    triplet: Triplet
    is_verified: bool
    confidence: float
    reason: str = ""


class TripletVerifier:
    """Verifies triplets against source text using batch verification."""
    
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
        
        # Lazy-load LLM
        self._llm = None
        
        self.template = self._load_template()
        logging.info(f"[Verifier] Initialized with batch verification using: {verification_model1}")
    
    @property
    def llm(self):
        """Lazy-load LLM."""
        if self._llm is None:
            logging.info(f"[Verifier] Loading LLM: {self.model_name}")
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
        """
        Build verification prompt for multiple triplets.
        
        Args:
            text: Source text to verify against
            triplets: List of triplets to verify
            
        Returns:
            Formatted prompt string
        """
        # Format triplets as numbered list
        triplet_lines = []
        for i, triplet in enumerate(triplets, 1):
            triplet_str = f'{i}. ["{triplet.subject}", "{triplet.relation}", "{triplet.object}"]'
            triplet_lines.append(triplet_str)
        
        triplets_section = "\n".join(triplet_lines)
        
        prompt = self.template.replace("{text}", text).replace("{triplets}", triplets_section)
        return prompt
    
    def _parse_batch_response(self, response: Dict, triplets: List[Triplet], default_confidence: float) -> List[VerificationResult]:
        """
        Parse batch verification JSON response into VerificationResults.
        
        Args:
            response: JSON response dict from LLM
            triplets: Original list of triplets (to match indices)
            default_confidence: Default confidence to use
            
        Returns:
            List of VerificationResult for each triplet
        """
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
                # Triplet not in response - treat as not verified
                logging.warning(f"[Verifier] Triplet {i} not found in response, marking as not verified")
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
        """
        Verify all triplets in a graph against the source text using batch verification.
        
        Args:
            text: Source text that the graph was extracted from
            graph: Graph containing triplets to verify
            
        Returns:
            Tuple of (verified_graph, verification_results)
            - verified_graph: Graph containing only verified triplets
            - verification_results: List of results for all triplets
        """
        if not graph.triples:
            return Graph(triples=[]), []
        
        # Build batch prompt
        prompt = self._build_batch_prompt(text, graph.triples)
        
        try:
            # Call LLM with batch verification (single call for all triplets)
            response_data, confidence = self.llm.batch_verify(prompt)
            
            # Parse response into results
            results = self._parse_batch_response(response_data, graph.triples, confidence)
            
        except Exception as e:
            logging.error(f"[Verifier] Batch verification failed: {e}")
            # On failure, mark all as not verified
            results = [
                VerificationResult(
                    triplet=triplet,
                    is_verified=False,
                    confidence=0.0,
                    reason=f"Verification error: {str(e)}"
                )
                for triplet in graph.triples
            ]
        
        # Build verified graph
        verified_triplets = [r.triplet for r in results if r.is_verified]
        verified_graph = Graph(triples=verified_triplets)
        
        return verified_graph, results
    
    def verify_all_graphs(
        self, 
        paragraphs: List[str], 
        kgs: DynamicKnowledgeGraphs
    ) -> Tuple[DynamicKnowledgeGraphs, Dict[str, List[VerificationResult]]]:
        """
        Verify all graphs against their corresponding paragraphs.
        
        Args:
            paragraphs: List of source paragraphs
            kgs: Knowledge graphs to verify
            
        Returns:
            Tuple of (verified_kgs, all_results)
        """
        verified_graphs = {}
        all_results = {}
        
        for i, para in enumerate(paragraphs, 1):
            graph_key = f"graph_{i}"
            graph = kgs.get_graph(i)
            
            para_preview = para[:100] + "..." if len(para) > 100 else para
            logging.info(f"[Verifier] Verifying {graph_key} ({len(graph.triples)} triplets)...")
            logging.info(f"[Verifier]   Paragraph: {para_preview}")
            
            verified_graph, results = self.verify_graph(para, graph)
            verified_graphs[graph_key] = verified_graph
            all_results[graph_key] = results
            
            verified_count = sum(1 for r in results if r.is_verified)
            logging.info(f"[Verifier] {graph_key}: {verified_count}/{len(results)} triplets verified")
            
            # Log each triplet result
            for idx, result in enumerate(results, 1):
                t = result.triplet
                status = "VERIFIED" if result.is_verified else "REJECTED"
                reason_info = f" - {result.reason}" if result.reason else ""
                logging.info(f"[Verifier]   [{idx}] {status}: ({t.subject}, {t.relation}, {t.object}){reason_info}")
        
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
        """
        Get failed triplets per graph for feedback to extractor.
        
        Args:
            results: Verification results dictionary
            
        Returns:
            Dictionary mapping graph_key to list of failed triplets as [subject, relation, object]
        """
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
        """
        Get verified triplets per graph for feedback to extractor.
        
        Args:
            results: Verification results dictionary
            
        Returns:
            Dictionary mapping graph_key to list of verified triplets as [subject, relation, object]
        """
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
        """Clean up resources."""
        if self._llm is not None and hasattr(self._llm, 'clear_model'):
            self._llm.clear_model()
        self._llm = None
        logging.info("[Verifier] Resources cleared")
