import os
import logging
import time
import re
from google import genai

class GeminiFactory:
    MIN_REQUEST_INTERVAL = 4.0  # 15 RPM cap

    def __init__(self, model_name: str, api_key: str | None = None):
        logging.info(f"[Gemini] Loading model: {model_name}")
        self.model = model_name
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("No API key provided and GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=key)
        self._last_request_time = 0.0
    
    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _parse_retry_time(self, error_msg: str) -> float:
        m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error_msg), re.IGNORECASE)
        if m:
            return float(m.group(1)) + 1.0
        return 0.0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                err = str(e)
                if attempt < max_retries - 1:
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        wait = self._parse_retry_time(err) or min(15 * (2 ** attempt), 60)
                        logging.warning(f"[Gemini] Rate limited, waiting {wait:.1f}s (attempt {attempt+1})")
                        time.sleep(wait)
                    else:
                        wait = min(2 ** attempt * 5, 60)
                        logging.warning(f"[Gemini] Error (attempt {attempt+1}): {e}, retrying in {wait}s")
                        time.sleep(wait)
                else:
                    raise RuntimeError(f"[Gemini] Failed after {max_retries} attempts: {e}") from e


