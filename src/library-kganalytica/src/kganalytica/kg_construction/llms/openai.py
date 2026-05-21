import os
import logging
import time
from openai import OpenAI


class OpenAIFactory:
    def __init__(self, model_name: str, api_key: str | None = None):
        logging.info(f"[OpenAI] Loading model: {model_name}")
        self.model_name = model_name
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("No API key provided and OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=key)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)
                    logging.warning(f"[OpenAI] Error (attempt {attempt+1}): {e}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"[OpenAI] Failed after {max_retries} attempts: {e}") from e



