import os
import logging
import json
import time
import re
from groq import Groq
from ..schemas import KnowledgeGraphs
from ...utils.env_utils import load_environment

load_environment()



class GroqFactory:
    def __init__(self, model_name: str):
        logging.info(f"[Groq] Loading model: {model_name}")
        self.model_name = model_name
        
        # Load all available Groq API keys
        self.api_keys = self._load_all_api_keys()
        if not self.api_keys:
            raise ValueError("No GROQ_API_KEY found in environment variables")
        
        self.current_key_index = 0
        logging.info(f"[Groq] Loaded {len(self.api_keys)} API keys")
        self.client = Groq(api_key=self.api_keys[self.current_key_index])
    
    def _load_all_api_keys(self) -> list:
        """Load all available Groq API keys from environment."""
        keys = []
        # First key without number
        key = os.getenv("GROQ_API_KEY")
        if key:
            keys.append(key)
        # Keys with numbers (2-20 to allow for expansion)
        for i in range(2, 21):
            key = os.getenv(f"GROQ_API_KEY{i}")
            if key:
                keys.append(key)
        return keys
    
    def _switch_api_key(self) -> bool:
        """Switch to the next available API key. Returns True if switched, False if only one key."""
        if len(self.api_keys) <= 1:
            return False
        
        next_index = (self.current_key_index + 1) % len(self.api_keys)
        
        # Log when cycling back to first key
        if next_index == 0:
            logging.info(f"[Groq] Cycled through all {len(self.api_keys)} API keys, rotating back to key 1")
        
        self.current_key_index = next_index
        self.client = Groq(api_key=self.api_keys[self.current_key_index])
        logging.info(f"[Groq] Switched to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        return True
    
    def _extract_retry_time(self, error_msg: str) -> float:
        """Extract retry time from rate limit error message."""
        # Look for patterns like "8m30.624s" or "30s" or "2m"
        match = re.search(r'try again in (\d+)m(\d+\.?\d*)s', str(error_msg))
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        
        match = re.search(r'try again in (\d+\.?\d*)s', str(error_msg))
        if match:
            return float(match.group(1))
        
        match = re.search(r'try again in (\d+)m', str(error_msg))
        if match:
            return int(match.group(1)) * 60
        
        return 60  # Default wait time

    def _get_key_identifier(self) -> str:
        """Get a masked identifier for the current API key."""
        key = self.api_keys[self.current_key_index]
        # Show first 8 and last 4 characters
        if len(key) > 12:
            return f"{key[:8]}...{key[-4:]}"
        return key[:4] + "..."

    def _handle_rate_limit(self, error, attempt: int) -> tuple:
        """
        Calculate wait time for rate limit errors.
        Returns tuple of (wait_time, should_switch_key).
        If wait_time > 10s, will attempt to switch API key instead.
        """
        error_str = str(error)
        
        # Handle expired/invalid API key - switch immediately
        if '401' in error_str or 'expired_api_key' in error_str.lower() or 'invalid api key' in error_str.lower():
            key_id = self._get_key_identifier()
            logging.warning(f"[Groq] API key {self.current_key_index + 1}/{len(self.api_keys)} ({key_id}) is expired/invalid")
            if self._switch_api_key():
                logging.info(f"[Groq] Switched to API key {self.current_key_index + 1}/{len(self.api_keys)} ({self._get_key_identifier()})")
                return 0.5, True  # Minimal delay, key was switched
            else:
                logging.error(f"[Groq] All API keys are expired/invalid!")
                return 5, False
        
        if '429' in error_str or 'rate_limit' in error_str.lower():
            wait_time = self._extract_retry_time(error_str)
            
            # If wait time > 10 seconds, try to switch API key
            if wait_time > 10:
                if self._switch_api_key():
                    logging.info(f"[Groq] Rate limit wait time was {wait_time:.1f}s, switched to new API key instead")
                    return 1, True  # Small delay after switching, key was switched
                else:
                    # No more keys available, must wait
                    wait_time = min(wait_time + 5, 600)
                    logging.info(f"[Groq] No more API keys available. Waiting {wait_time:.1f}s before retry...")
                    return wait_time, False
            else:
                # Short wait time, just wait
                wait_time = min(wait_time + 5, 600)
                logging.info(f"[Groq] Rate limited. Waiting {wait_time:.1f}s before retry...")
                return wait_time, False
        else:
            # Exponential backoff for other errors
            return min(2 ** attempt * 5, 60), False

    def get_answer(self, query: str) -> str:
        max_retries = 5
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_completion_tokens=8192
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.warning(f"[Groq] Error generating text (attempt {attempt+1}/{max_retries}) using key {self.current_key_index + 1} ({self._get_key_identifier()}): {e}")
                if attempt < max_retries - 1:
                    wait_time, key_switched = self._handle_rate_limit(e, attempt)
                    time.sleep(wait_time)
                    if key_switched:
                        attempt = 0  # Reset attempts after switching key
                        continue
                attempt += 1
        raise RuntimeError("Failed to get response from Groq model after multiple attempts.")

    def get_answer_with_confidence(self, query: str) -> tuple:
        """
        Get answer with confidence score.
        Groq doesn't support logprobs, so we return a default confidence.
        
        Returns:
            Tuple of (response_text, confidence_score)
        """
        max_retries = 5
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0,  # Deterministic for verification
                    max_completion_tokens=1024
                )
                
                content = response.choices[0].message.content
                # Groq doesn't support logprobs, use default confidence
                confidence = 0.75
                
                return content, confidence
            except Exception as e:
                logging.warning(f"[Groq] Error generating text with confidence (attempt {attempt+1}/{max_retries}) using key {self.current_key_index + 1} ({self._get_key_identifier()}): {e}")
                if attempt < max_retries - 1:
                    wait_time, key_switched = self._handle_rate_limit(e, attempt)
                    time.sleep(wait_time)
                    if key_switched:
                        attempt = 0  # Reset attempts after switching key
                        continue
                attempt += 1
        raise RuntimeError("Failed to get response from Groq model after multiple attempts.")

    def final_kg_constructor(self, text1: str, text2: str, text3: str, system_prompt: str):
        query = f"TEXT 1:\n{text1}\n\nTEXT 2:\n{text2}\n\nTEXT 3:\n{text3}"
        
        max_retries = 5
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt + "\n\nIMPORTANT: Output strictly valid JSON."},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0,
                    max_completion_tokens=8192,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                return KnowledgeGraphs(**data)

            except Exception as e:
                logging.warning(f"[Groq] KG Construction Error (attempt {attempt+1}/{max_retries}) using key {self.current_key_index + 1} ({self._get_key_identifier()}): {e}")
                if attempt < max_retries - 1:
                    wait_time, key_switched = self._handle_rate_limit(e, attempt)
                    time.sleep(wait_time)
                    if key_switched:
                        attempt = 0  # Reset attempts after switching key
                        continue
                attempt += 1
                
        raise RuntimeError("Failed to generate Knowledge Graph from Groq after multiple attempts.")

    def batch_verify(self, prompt: str) -> tuple:
        """
        Verify multiple triplets at once and return JSON results.
        Groq doesn't support logprobs, so we return default confidence.
        
        Args:
            prompt: The verification prompt with multiple triplets
            
        Returns:
            Tuple of (json_response_dict, avg_confidence)
        """
        max_retries = 5
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_completion_tokens=4096,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Groq doesn't support logprobs, use default confidence
                avg_confidence = 0.75
                
                return data, avg_confidence
            except json.JSONDecodeError as e:
                logging.warning(f"[Groq] JSON parse error in batch_verify (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt * 2)
                attempt += 1
            except Exception as e:
                logging.warning(f"[Groq] Error in batch_verify (attempt {attempt+1}/{max_retries}) using key {self.current_key_index + 1} ({self._get_key_identifier()}): {e}")
                if attempt < max_retries - 1:
                    wait_time, key_switched = self._handle_rate_limit(e, attempt)
                    time.sleep(wait_time)
                    if key_switched:
                        attempt = 0  # Reset attempts after switching key
                        continue
                attempt += 1
        raise RuntimeError("Failed to batch verify from Groq after multiple attempts.")

    def clear_model(self):
        """Clean up resources."""
        self.client = None
        logging.info("[Groq] Resources cleared")
