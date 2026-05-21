import os
import logging
import time
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqFactory:
    """Groq API client for Llama, Mixtral models with rate-limit handling."""
    def __init__(self, model_name: str):
        """Initialize Groq client with API key from environment."""
        logging.info(f"[Groq] Loading model: {model_name}")
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)

    def _parse_retry_time(self, error_msg: str) -> float:
        """Extract wait time from rate-limit error message."""
        m = re.search(r"try again in (\d+)m(\d+\.?\d*)s", str(error_msg))
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m = re.search(r"try again in (\d+\.?\d*)s", str(error_msg))
        return float(m.group(1)) if m else 60.0

    def query_model(self, query: str) -> str:
        """Query Groq API with exponential backoff on rate limits."""
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}],
                    temperature=0.0,
                    max_completion_tokens=4096,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait = min(self._parse_retry_time(error_str) + 5, 120)
                    logging.warning(f"[Groq] Rate limited. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                elif attempt < 4:
                    wait = min(2 ** attempt * 5, 60)
                    logging.warning(f"[Groq] Error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq: failed after 5 attempts.")
