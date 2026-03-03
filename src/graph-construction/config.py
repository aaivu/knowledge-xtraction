"""
Configuration for the Verified Knowledge Graph Construction Pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ExtractionConfig:
    """Configuration for knowledge graph extraction."""
    extraction_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 8000
    max_retries: int = 3


@dataclass
class VerificationConfig:
    """Configuration for triplet verification."""
    verification_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    double_verification: bool = True
    acceptance_threshold: float = 0.5
    temperature: float = 0.0
    max_tokens: int = 512
