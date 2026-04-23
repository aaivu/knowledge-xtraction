import os
import logging
import time
import re
from groq import Groq



class GroqFactory:
    def __init__(self, model_name: str, api_key: str | None = None):
        logging.info(f"[Groq] Loading model: {model_name}")
        self.model_name = model_name
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("No API key provided and GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=key)
    
    def _parse_retry_time(self, error_msg: str) -> float:
        m = re.search(r"try again in (\d+)m(\d+\.?\d*)s", str(error_msg))
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m = re.search(r"try again in (\d+\.?\d*)s", str(error_msg))
        if m:
            return float(m.group(1))
        m = re.search(r"try again in (\d+)m", str(error_msg))
        if m:
            return int(m.group(1)) * 60
        return 60.0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                err = str(e)
                if attempt < max_retries - 1:
                    if "429" in err or "rate_limit" in err.lower():
                        wait = self._parse_retry_time(err) + 5
                        logging.warning(f"[Groq] Rate limited, waiting {wait:.1f}s (attempt {attempt+1})")
                        time.sleep(wait)
                    else:
                        wait = min(2 ** attempt * 5, 60)
                        logging.warning(f"[Groq] Error (attempt {attempt+1}): {e}, retrying in {wait}s")
                        time.sleep(wait)
                else:
                    raise RuntimeError(f"[Groq] Failed after {max_retries} attempts: {e}") from e


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
