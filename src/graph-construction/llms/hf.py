import logging
import json
import re
from schemas import KnowledgeGraphs
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline)
import torch

class HuggingFaceFactory():
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
        logging.info(f"[HF] Model {model_name} loaded successfully!")

    def get_answer(self, query: str, kg: bool = False) -> str:
        for _ in range(3):  # Retry mechanism
            try:
                # Use return_full_text=False to avoid getting the input back
                # Increased max_new_tokens to 80000 to prevent truncation of graph_3
                
                generation_kwargs = {
                    "max_new_tokens": 80000,
                    "return_full_text": False,
                }
                 
                if kg:
                    generation_kwargs.update({
                        "do_sample": False,   # Greedy decoding (deterministic)
                        "temperature": 0.0,   # Optional, ignored when do_sample=False
                        "top_p": 1.0,         # Safe default
                        "top_k": 0            # Disable top-k sampling
                    })
                
                result = self.pipeline(query, **generation_kwargs)
                generated_text = result[0]["generated_text"]
                return generated_text.strip()
            except Exception as e:
                logging.warning(f"[HF] Error generating text: {e}")
        raise RuntimeError("Failed to get response from Hugging Face model after multiple attempts.")

    def get_answer_with_confidence(self, query: str) -> tuple:
        """
        Get answer with confidence score based on token probabilities.
        
        Returns:
            Tuple of (response_text, confidence_score)
        """
        for _ in range(3):
            try:
                inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,  # Short response for verification
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Decode the generated text
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Calculate confidence from scores
                if outputs.scores:
                    probs = []
                    for i, score in enumerate(outputs.scores[:5]):  # First 5 tokens
                        # Get the probability of the chosen token
                        token_id = generated_ids[i].item() if i < len(generated_ids) else None
                        if token_id is not None:
                            softmax_scores = torch.softmax(score[0], dim=-1)
                            prob = softmax_scores[token_id].item()
                            probs.append(prob)
                    confidence = sum(probs) / len(probs) if probs else 0.5
                else:
                    confidence = 0.5
                
                return generated_text, confidence
            except Exception as e:
                logging.warning(f"[HF] Error generating with confidence: {e}")
        raise RuntimeError("Failed to get response from HuggingFace model after multiple attempts.")

    def final_kg_constructor(self, text1: str, text2: str, text3: str, system_prompt: str):
        query = f"TEXT 1:\n{text1}\n\nTEXT 2:\n{text2}\n\nTEXT 3:\n{text3}"
        
        # Construct messages for chat template
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{query}\n\nIMPORTANT: You must generate the full JSON including graph_1, graph_2, and graph_3. Do not stop early."}
        ]
        
        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for models without chat template support
            full_prompt = f"[INST] {messages[0]['content']} [/INST]"
        
        logging.info(f"Generating KG with {self.model_name}...")
        
        response_text = self.get_answer(query=full_prompt, kg=True)
        
        # Extract JSON from the response (it might be wrapped in ```json ... ```)
        try:
             # Try to find JSON block
            match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Fallback: try to find start and end of JSON
                match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = response_text
            
            # Additional debug logging
            logging.debug(f"[HF] Raw response extracted for JSON: {json_str[:200]}...")

            data = json.loads(json_str)

            # Ensure all required graphs exist (Handling incomplete generation)
            for graph_key in ['graph_1', 'graph_2', 'graph_3']:
                if graph_key not in data:
                    logging.warning(f"[HF] Missing {graph_key} in response. using empty graph.")
                    logging.info(f"[HF] Full response text causing missing key: {response_text}")
                    data[graph_key] = {"triples": []}
            
            # Log empty triples if found (to debug empty outputs)
            if not data.get('graph_1', {}).get('triples') and not data.get('graph_2', {}).get('triples'):
                 logging.warning(f"[HF] Generated graphs are empty. Response text: {response_text}")

            return KnowledgeGraphs(**data)
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from {self.model_name} response: {e}")
            logging.debug(f"Raw response: {response_text}")
            raise
        except Exception as e:
            logging.error(f"Error constructing KG: {e}")
            raise

    def batch_verify(self, prompt: str) -> tuple:
        """
        Verify multiple triplets at once and return JSON results.
        
        Args:
            prompt: The verification prompt with multiple triplets
            
        Returns:
            Tuple of (json_response_dict, avg_confidence)
        """
        for _ in range(3):
            try:
                # Apply chat template if available
                if hasattr(self.tokenizer, "apply_chat_template"):
                    messages = [{"role": "user", "content": prompt}]
                    full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    full_prompt = f"[INST] {prompt} [/INST]"
                
                # Generate with controlled output
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Decode the generated text
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Extract JSON from response
                match = re.search(r"```json\s*(.*?)\s*```", generated_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    match = re.search(r"(\{.*\})", generated_text, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    else:
                        json_str = generated_text
                
                data = json.loads(json_str)
                
                # Calculate confidence from scores
                if outputs.scores:
                    probs = []
                    for i, score in enumerate(outputs.scores[:10]):
                        token_id = generated_ids[i].item() if i < len(generated_ids) else None
                        if token_id is not None:
                            softmax_scores = torch.softmax(score[0], dim=-1)
                            prob = softmax_scores[token_id].item()
                            probs.append(prob)
                    confidence = sum(probs) / len(probs) if probs else 0.75
                else:
                    confidence = 0.75
                
                return data, confidence
            except json.JSONDecodeError as e:
                logging.warning(f"[HF] JSON parse error in batch_verify: {e}")
            except Exception as e:
                logging.warning(f"[HF] Error in batch_verify: {e}")
        raise RuntimeError("Failed to batch verify from HuggingFace after multiple attempts.")

    def clear_model(self):
        """Unload model to free GPU memory."""
        if self.model_name:
            logging.info(f"[HF] Clearing model: {self.model_name}")

        del self.pipeline
        del self.model
        del self.tokenizer
        self.pipeline = None
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("[HF] Model cleared successfully.")
