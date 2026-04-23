import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


class HuggingFaceFactory:
    def __init__(self, model_name: str):
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

    def get_answer(self, query: str) -> str:
        for _ in range(3):
            try:
                result = self.pipeline(
                    query, max_new_tokens=4096, return_full_text=False, do_sample=False
                )
                return result[0]["generated_text"].strip()
            except Exception as e:
                logging.warning(f"[HF] Error: {e}")
        raise RuntimeError("HuggingFace: failed after 3 attempts.")

    def clear_model(self):
        del self.model, self.tokenizer, self.pipeline
        torch.cuda.empty_cache()

