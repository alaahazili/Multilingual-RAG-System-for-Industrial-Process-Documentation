import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SelfHostedGenerationService:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize self-hosted text generation service.
        
        Args:
            model_name: Hugging Face model name (default: microsoft/Phi-3-mini-4k-instruct)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._is_loaded = False
        
        # Create persistent storage directory
        self.model_cache_dir = Path("models_cache")
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Create specific model directory
        model_slug = model_name.replace("/", "_")
        self.model_dir = self.model_cache_dir / model_slug
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info(f"Model cache directory: {self.model_dir}")
        
    async def load_model(self):
        """Load the model asynchronously."""
        if self._is_loaded:
            return
            
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model_sync)
            
            self._is_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_model_sync(self):
        """Load model synchronously (run in thread pool)."""
        try:
            # Check if model is already cached locally
            if self.model_dir.exists() and any(self.model_dir.iterdir()):
                logger.info(f"Loading model from local cache: {self.model_dir}")
                
                # Load tokenizer from local cache
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_dir),
                    use_fast=True,
                    trust_remote_code=True
                )
                
                # Load model from local cache
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_dir),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                logger.info(f"Downloading model from Hugging Face: {self.model_name}")
                logger.info("This will happen only once. Model will be cached for future use.")
                
                # Download and cache tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    trust_remote_code=True,
                    cache_dir=str(self.model_cache_dir)
                )
                
                # Download and cache model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=str(self.model_cache_dir)
                )
                
                # Save model and tokenizer locally
                logger.info(f"Saving model to local cache: {self.model_dir}")
                self.tokenizer.save_pretrained(str(self.model_dir))
                self.model.save_pretrained(str(self.model_dir))
                
                logger.info(f"Model downloaded and cached to: {self.model_dir}")
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def generate(self, question: str, contexts: List[str]) -> str:
        """Generate an answer based on the question and contexts."""
        if not self._is_loaded:
            await self.load_model()
        
        # Create context block (limit context length to avoid token overflow)
        context_block = "\n\n".join(f"[Context {i+1}]\n{c[:500]}" for i, c in enumerate(contexts[:3]))  # Limit to 3 contexts, 500 chars each
        
        # Create system prompt for PAC technical content
        system_prompt = (
            "You are a technical expert assistant for PAC (Process Automation Control) systems. "
            "Answer questions strictly based on the provided context from PAC control philosophy documents. "
            "Provide precise, technical answers in 30-60 words. "
            "Always include specific equipment tags (e.g., 07-TK-001), measurements, and technical specifications when mentioned. "
            "For equipment dimensions, specifications, or technical details, be exact and include units. "
            "If the specific information is not present in the context, say 'I don't have enough information' rather than guessing."
        )
        
        # Create the full prompt for Phi-3
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\nAnswer this technical question about PAC systems using the provided context. Be precise with equipment tags, measurements, and specifications.\n\nQuestion: {question}\n\nContext:\n{context_block}<|end|>\n<|assistant|>\n"

        try:
            # Generate response in thread pool with very short timeout
            loop = asyncio.get_event_loop()
            answer = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self._generate_sync, prompt),
                timeout=8.0  # 8 second timeout for very fast response
            )
            
            return self._constrain_answer_length(answer, target_length=50)
            
        except asyncio.TimeoutError:
            logger.error("Generation timed out after 8 seconds")
            return "Response timeout. Please ask a shorter question about PAC equipment, measurements, or specifications."
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Sorry, I encountered an error while generating the response."
    
    def _generate_sync(self, prompt: str) -> str:
        """Generate response synchronously (run in thread pool)."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response with ultra-fast settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,   # Very short responses for speed
                    temperature=0.1,     # Very low temperature for deterministic responses
                    top_p=0.5,           # Lower top_p for faster generation
                    do_sample=False,     # Disable sampling for speed
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,  # No repetition penalty for speed
                    num_beams=1,         # Greedy decoding for speed
                    use_cache=True,      # Enable caching for speed
                    early_stopping=False # Disable early stopping for speed
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error in _generate_sync: {e}")
            raise
    
    def _constrain_answer_length(self, answer: str, target_length: int = 80) -> str:
        """Constrain answer length to target word count while preserving meaning."""
        if not answer:
            return answer
        
        words = answer.split()
        if len(words) <= target_length:
            return answer
        
        # Try to find a natural break point
        sentences = answer.split('.')
        constrained_answer = ""
        word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = sentence.split()
            if word_count + len(sentence_words) <= target_length:
                if constrained_answer:
                    constrained_answer += ". " + sentence
                else:
                    constrained_answer = sentence
                word_count += len(sentence_words)
            else:
                break
        
        # If we have a partial sentence, try to complete it naturally
        if constrained_answer and not constrained_answer.endswith('.'):
            constrained_answer += '.'
        
        # Ensure we don't exceed target length even with the fallback
        if constrained_answer:
            final_words = constrained_answer.split()
            if len(final_words) > target_length:
                constrained_answer = " ".join(final_words[:target_length]) + "."
        
        return constrained_answer if constrained_answer else answer[:target_length * 5]  # Fallback to character limit
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "model_type": "causal_lm",
            "cache_dir": str(self.model_dir)
        }
