import os
import logging
import time
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()


class GeminiFactory:
    """Google Gemini API client with rate-limit enforcement."""
    MIN_REQUEST_INTERVAL = 4.0  # Enforce 15 RPM

    def __init__(self, model_name: str):
        """Initialize Gemini client with API key from environment."""
        logging.info(f"[Gemini] Loading model: {model_name}")
        self.model = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Enforce request rate limit (15 RPM)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _backoff(self, error: Exception, attempt: int) -> float:
        """Calculate backoff wait time from error; exponential if retryable."""
        s = str(error)
        if "429" in s or "RESOURCE_EXHAUSTED" in s:
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", s, re.IGNORECASE)
            return float(m.group(1)) + 1.0 if m else min(15 * (2 ** attempt), 60)
        return 0.0

    def query_model(self, query: str) -> str:
        """Query the model with rate-limit handling."""
        for attempt in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(model=self.model, contents=query)
                return response.text
            except Exception as e:
                wait = self._backoff(e, attempt)
                if wait > 0 and attempt < 4:
                    logging.warning(f"[Gemini] Rate limit, waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    logging.warning(f"[Gemini] Error: {e}")
        raise RuntimeError("Gemini: failed after 5 attempts.")