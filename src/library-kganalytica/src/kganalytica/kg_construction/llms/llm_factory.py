from .openai import OpenAIFactory
from .gemini import GeminiFactory
from .hf import HuggingFaceFactory
from .groq import GroqFactory

class LLMFactorySelector:
    _model_cache: dict = {}

    @staticmethod
    def get_factory(model_name: str, api_key: str | None = None):
        cache_key = (model_name, api_key)
        if cache_key in LLMFactorySelector._model_cache:
            return LLMFactorySelector._model_cache[cache_key]

        model_name_lower = model_name.lower()

        if "gemini" in model_name_lower:
            instance = GeminiFactory(model_name, api_key=api_key)
        elif "gpt" in model_name_lower or "openai" in model_name_lower:
            instance = OpenAIFactory(model_name, api_key=api_key)
        elif any(x in model_name_lower for x in ("llama", "mixtral", "groq", "mistral", "qwen", "deepseek")):
            instance = GroqFactory(model_name, api_key=api_key)
        else:
            instance = HuggingFaceFactory(model_name, api_key=api_key)

        LLMFactorySelector._model_cache[cache_key] = instance
        return instance

    @staticmethod
    def clear_cache():
        LLMFactorySelector._model_cache.clear()
