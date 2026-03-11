"""
Configuration for the Verified Knowledge Graph Construction Pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ExtractionConfig:
    """Configuration for graph extraction."""
    # LLM for extraction (larger, more capable model)
    extraction_model: str = "gemini-2.5-flash"
    
    # Temperature for extraction (0 for deterministic)
    temperature: float = 0.0
    
    # Maximum tokens for extraction response
    max_tokens: int = 8000
    
    # Number of retry attempts for extraction
    max_retries: int = 3


@dataclass
class VerificationConfig:
    """Configuration for triplet verification."""
    # LLM for verification (smaller, efficient model)
    verification_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Run verification twice independently
    double_verification: bool = True
    
    # Threshold for accepting a triplet (both verifications must pass if double_verification=True)
    acceptance_threshold: float = 0.5
    
    # Temperature for verification (0 for deterministic)
    temperature: float = 0.0
    
    # Maximum tokens for verification response
    max_tokens: int = 512
