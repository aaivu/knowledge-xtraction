import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

class HuggingFaceFactory:
    def __init__(self, model_name: str, api_key: str | None = None):  # api_key unused for local models
        logging.info(f"[HF] Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            # Attempt 4-bit quantization for efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            logging.info(f"[HF] Model {model_name} loaded in 4-bit quantized mode.")

        except Exception as e:
            logging.warning(f"[HF] 4-bit quantization failed ({e}). Falling back to FP16.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        logging.info(f"[HF] Model {model_name} loaded successfully!")

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        for attempt in range(3):
            try:
                kwargs: dict = {"max_new_tokens": max_tokens, "return_full_text": False}
                if temperature > 0:
                    kwargs["do_sample"] = True
                    kwargs["temperature"] = temperature
                else:
                    kwargs["do_sample"] = False
                result = self.pipeline(prompt, **kwargs)
                return result[0]["generated_text"].strip()
            except Exception as e:
                logging.warning(f"[HF] Error (attempt {attempt+1}): {e}")
        raise RuntimeError("[HF] Failed after 3 attempts")

