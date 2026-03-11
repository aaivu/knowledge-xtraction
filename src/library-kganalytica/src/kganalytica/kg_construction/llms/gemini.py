import os
import json
import logging
import time
import re
from google import genai
from ..schemas import KnowledgeGraphs
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiFactory():
    # Rate limit: 15 RPM = 1 request per 4 seconds
    MIN_REQUEST_INTERVAL = 4.0  # seconds between requests
    
    def __init__(self, model_name: str):
        logging.info(f"[Gemini] Loading model: {model_name}")
        self.model = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)
        self.cached_content = None
        self._last_request_time = 0.0
    
    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests (15 RPM = 4s between requests)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            wait_time = self.MIN_REQUEST_INTERVAL - elapsed
            logging.debug(f"[Gemini] Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        self._last_request_time = time.time()
    
    def _handle_rate_limit_error(self, error: Exception, attempt: int) -> float:
        """
        Parse rate limit error and return recommended wait time.
        Returns wait time in seconds, or 0 if not a rate limit error.
        """
        error_str = str(error)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            # Try to extract retry delay from error message
            match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str, re.IGNORECASE)
            if match:
                wait_time = float(match.group(1)) + 1.0  # Add 1s buffer
            else:
                # Exponential backoff: 15, 30, 60 seconds
                wait_time = min(15 * (2 ** attempt), 60)
            logging.info(f"[Gemini] Rate limit hit, waiting {wait_time:.1f}s before retry...")
            return wait_time
        return 0

    def setup_cache(self, system_prompt: str, ttl_seconds: int = 1200):
        try:
            logging.info(f"Setting up context cache for model {self.model}...")
            # Create the cache
            self.cached_content = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    ttl=f"{ttl_seconds}s",
                )
            )
            logging.info(f"Cache created successfully: {self.cached_content.name}")
        except Exception as e:
            logging.warning(f"Failed to create cache. Proceeding without cache. Error: {e}")
            self.cached_content = None

    def get_answer(self, query: str) -> str:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=query
                )
                return response.text
                
            except Exception as e:
                wait_time = self._handle_rate_limit_error(e, attempt)
                if wait_time > 0 and attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                logging.warning(f"[Gemini] Error: {e}")
        raise RuntimeError("Failed to get response from Gemini model after multiple attempts.")

    def get_answer_with_confidence(self, query: str) -> tuple:
        """
        Get answer with confidence score based on avg_logprobs.
        Falls back to default confidence if logprobs not supported.
        
        Returns:
            Tuple of (response_text, confidence_score)
        """
        import math
        
        # First try with logprobs
        try:
            self._wait_for_rate_limit()
            response = self.client.models.generate_content(
                model=self.model,
                contents=query,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic for verification
                    response_logprobs=True,
                    logprobs=5
                )
            )
            
            content = response.text
            
            # Get confidence from avg_logprobs if available
            confidence = 0.75  # Default
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'avg_logprobs') and candidate.avg_logprobs is not None:
                    # Convert log probability to probability
                    confidence = math.exp(candidate.avg_logprobs)
                elif hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    # Fallback: compute from token logprobs
                    logprobs_result = candidate.logprobs_result
                    if hasattr(logprobs_result, 'chosen_candidates') and logprobs_result.chosen_candidates:
                        probs = [math.exp(c.log_probability) for c in logprobs_result.chosen_candidates[:5]]
                        confidence = sum(probs) / len(probs) if probs else 0.75
            
            return content, confidence
            
        except Exception as e:
            error_str = str(e)
            # If logprobs not supported, fall back to regular generation with default confidence
            if "Logprobs is not enabled" in error_str or "INVALID_ARGUMENT" in error_str:
                logging.debug(f"[Gemini] Logprobs not supported for {self.model}, using default confidence")
                try:
                    content = self.get_answer(query)
                    return content, 0.75  # Default confidence when logprobs unavailable
                except Exception as fallback_e:
                    logging.warning(f"[Gemini] Fallback generation failed: {fallback_e}")
                    raise
            # Handle rate limit - fall back to get_answer which has retry logic
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logging.debug(f"[Gemini] Rate limit in get_answer_with_confidence, falling back to get_answer")
                content = self.get_answer(query)
                return content, 0.75
            else:
                logging.warning(f"[Gemini] Error generating with confidence: {e}")
                raise RuntimeError("Failed to get response from Gemini model.")
    
    def final_kg_constructor(self, text1: str, text2: str, text3: str, system_prompt: str):
        query = f"TEXT 1:\n{text1}\n\nTEXT 2:\n{text2}\n\nTEXT 3:\n{text3}"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                config_params = {
                    "response_mime_type": "application/json",
                    "response_schema": KnowledgeGraphs,
                    "temperature": 0
                }

                # Use cached content if available, otherwise use system_prompt
                if self.cached_content:
                    config_params["cached_content"] = self.cached_content.name
                else:
                    config_params["system_instruction"] = system_prompt

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[query],
                    config=types.GenerateContentConfig(**config_params),
                )
                kgs = KnowledgeGraphs(**json.loads(response.text))
                return kgs
                
            except Exception as e:
                wait_time = self._handle_rate_limit_error(e, attempt)
                if wait_time > 0 and attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                logging.warning(f"KG Construction Error: {e}")
                
        raise RuntimeError("Failed to generate Knowledge Graph after multiple attempts.")

    def batch_verify(self, prompt: str) -> tuple:
        """
        Verify multiple triplets at once and return JSON results.
        
        Args:
            prompt: The verification prompt with multiple triplets
            
        Returns:
            Tuple of (json_response_dict, avg_confidence)
        """
        import math
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                
                content = response.text
                data = json.loads(content)
                
                # Get confidence from avg_logprobs if available
                confidence = 0.75  # Default
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'avg_logprobs') and candidate.avg_logprobs is not None:
                        confidence = math.exp(candidate.avg_logprobs)
                
                return data, confidence
                
            except json.JSONDecodeError as e:
                logging.warning(f"[Gemini] JSON parse error in batch_verify: {e}")
            except Exception as e:
                wait_time = self._handle_rate_limit_error(e, attempt)
                if wait_time > 0 and attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                logging.warning(f"[Gemini] Error in batch_verify: {e}")
        raise RuntimeError("Failed to batch verify from Gemini after multiple attempts.")

    def clear_model(self):
        # Clean up cache if it exists
        if self.cached_content:
             try:
                 # Note: client.caches.delete(...) might be the way, 
                 # but for now we just clear the reference as TTL handles expiration
                 # or we strictly delete it.
                 # self.client.caches.delete(name=self.cached_content.name)
                 pass
             except Exception as e:
                 logging.warning(f"Error clearing cache: {e}")
        
        self.model = None
        self.client = None
        self.cached_content = None