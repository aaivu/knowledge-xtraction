import os
import logging
import json
from dotenv import load_dotenv
from openai import OpenAI
from schemas import KnowledgeGraphs

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
        for _ in range(3):  # Retry mechanism
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.warning(f"[OpenAI] Error generating text: {e}")
        raise RuntimeError("Failed to get response from OpenAI model after multiple attempts.")

    def get_answer_with_confidence(self, query: str) -> tuple:
        """
        Get answer with confidence score based on logprobs.
        
        Returns:
            Tuple of (response_text, confidence_score)
        """
        import math
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0,  # Deterministic for verification
                    logprobs=True,
                    top_logprobs=1
                )
                
                content = response.choices[0].message.content
                
                # Calculate confidence from logprobs of first few tokens
                logprobs_content = response.choices[0].logprobs
                if logprobs_content and logprobs_content.content:
                    # Average probability of first few tokens
                    probs = []
                    for token_info in logprobs_content.content[:5]:  # First 5 tokens
                        prob = math.exp(token_info.logprob)
                        probs.append(prob)
                    confidence = sum(probs) / len(probs) if probs else 0.5
                else:
                    confidence = 0.5  # Default if no logprobs
                
                return content, confidence
            except Exception as e:
                logging.warning(f"[OpenAI] Error generating text with confidence: {e}")
        raise RuntimeError("Failed to get response from OpenAI model after multiple attempts.")

    def final_kg_constructor(self, text1: str, text2: str, text3: str, system_prompt: str):
        query = f"TEXT 1:\n{text1}\n\nTEXT 2:\n{text2}\n\nTEXT 3:\n{text3}"
        
        for _ in range(3):
            try:
                # Use JSON mode if model supports it (gpt-4-turbo, gpt-3.5-turbo-0125, etc.)
                # Otherwise, rely on prompt/schema.
                # We can enforce JSON format with "response_format={"type": "json_object"}"
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt + "\n\nIMPORTANT: Output strictly valid JSON."},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                return KnowledgeGraphs(**data)

            except Exception as e:
                logging.warning(f"[OpenAI] KG Construction Error: {e}")
                
        raise RuntimeError("Failed to generate Knowledge Graph from OpenAI after multiple attempts.")

    def batch_verify(self, prompt: str) -> tuple:
        """
        Verify multiple triplets at once and return JSON results with confidence.
        Uses logprobs to calculate confidence for each verification.
        
        Args:
            prompt: The verification prompt with multiple triplets
            
        Returns:
            Tuple of (json_response_dict, avg_confidence)
        """
        import math
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    logprobs=True,
                    top_logprobs=5
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Calculate confidence from logprobs
                logprobs_content = response.choices[0].logprobs
                if logprobs_content and logprobs_content.content:
                    probs = []
                    for token_info in logprobs_content.content:
                        prob = math.exp(token_info.logprob)
                        probs.append(prob)
                    avg_confidence = sum(probs) / len(probs) if probs else 0.75
                else:
                    avg_confidence = 0.75
                
                return data, avg_confidence
            except json.JSONDecodeError as e:
                logging.warning(f"[OpenAI] JSON parse error in batch_verify: {e}")
            except Exception as e:
                logging.warning(f"[OpenAI] Error in batch_verify: {e}")
        raise RuntimeError("Failed to batch verify from OpenAI after multiple attempts.")

    def clear_model(self):
        self.client = None
