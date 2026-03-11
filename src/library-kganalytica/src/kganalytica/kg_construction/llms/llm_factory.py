from .openai import OpenAIFactory
from .gemini import GeminiFactory
from .hf import HuggingFaceFactory
from .groq import GroqFactory

class LLMFactorySelector:
    # Cache to store loaded model instances
    _model_cache = {}
    
    @staticmethod
    def get_factory(model_name: str):
        # Check if model is already loaded
        if model_name in LLMFactorySelector._model_cache:
            return LLMFactorySelector._model_cache[model_name]
        
        model_name_lower = model_name.lower()

        if "gemini" in model_name_lower:
            instance = GeminiFactory(model_name)
        elif "gpt" in model_name_lower or "openai" in model_name_lower:
            instance = OpenAIFactory(model_name)
        elif "groq" in model_name_lower or "llama" in model_name_lower or "mixtral" in model_name_lower:
            instance = GroqFactory(model_name)
        else:
            instance = HuggingFaceFactory(model_name)
        
        # Cache the instance
        LLMFactorySelector._model_cache[model_name] = instance
        return instance
    
    @staticmethod
    def clear_cache():
        """Clear all cached model instances."""
        for model_name, instance in LLMFactorySelector._model_cache.items():
            if hasattr(instance, 'clear_model'):
                instance.clear_model()
        LLMFactorySelector._model_cache.clear()