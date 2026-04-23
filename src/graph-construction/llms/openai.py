import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIFactory:
    def __init__(self, model_name: str):
        logging.info(f"[OpenAI] Loading model: {model_name}")
        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

    def get_answer(self, query: str) -> str:
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}],
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.warning(f"[OpenAI] Error: {e}")
        raise RuntimeError("OpenAI: failed after 3 attempts.")

    def clear_model(self):
        self.client = None
