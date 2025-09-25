"""
Ollama Generation Service
Purpose: Use Mistral-7B-Instruct via Ollama for the generative layer.
Provides instruction-tuned responses with solid general knowledge fallback.
"""

import asyncio
import logging
import json
import time
from typing import List, Optional, Dict, Any
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama service"""
    base_url: str = "http://localhost:11434"
    model_name: str = "mistral:instruct"
    timeout: int = 30
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


class OllamaGenerationService:
    """Ollama-based text generation service using Mistral-7B-Instruct"""
    
    def __init__(self, config: OllamaConfig = None):
        """
        Initialize Ollama generation service.
        
        Args:
            config: Ollama configuration
        """
        self.config = config or OllamaConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self._is_available = False
        
        logger.info(f"Initialized Ollama service with model: {self.config.model_name}")
        
    async def check_availability(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = await self.client.get(f"{self.config.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                if self.config.model_name in available_models:
                    self._is_available = True
                    logger.info(f"Ollama service available with model: {self.config.model_name}")
                    return True
                else:
                    logger.warning(f"Model {self.config.model_name} not found. Available: {available_models}")
                    return False
            else:
                logger.error(f"Ollama service not responding: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check Ollama availability: {e}")
            return False
            
    def _create_rag_prompt(self, question: str, contexts: List[str], conversation_history: str = "") -> str:
        """
        Create RAG prompt for Mistral-7B-Instruct.
        
        Args:
            question: User question
            contexts: Retrieved document chunks
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt for Mistral
        """
        # System instruction
        system_instruction = """You are a helpful assistant for the PAC (Process Automation Control) system documentation. 
Your task is to answer questions based on the provided document context. 

CRITICAL GUIDELINES:
1. ALWAYS search the provided document context FIRST before making any statements
2. If the information is clearly present in the document, state it CONFIDENTLY and directly
3. Do NOT say "not explicitly named" or "may not be mentioned" if the information is actually in the document
4. Be precise, technical, and helpful with medium-length responses (50-120 words)
5. Preserve technical details, formulas, and specifications exactly as written
6. If asked about formulas, provide the exact formula from the document
7. Reference specific content rather than generic section numbers
8. Do NOT refer to contexts as "Context 1, 2, 3" - instead refer to the actual content
9. Do NOT use phrases like "in Section X of the document" - be more specific about the content
10. If the document contains the answer, do NOT suggest looking elsewhere - provide the answer directly
11. Only use general knowledge if the document truly doesn't contain the information
12. When asked for a complete list of items, provide ALL items found in the document context
13. If multiple items are mentioned in different contexts, combine them into a comprehensive response

Document Context:"""

        # Format contexts clearly for better understanding
        context_text = "\n\n".join([f"DOCUMENT EXCERPT:\n{ctx}" for ctx in contexts])
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            history_text = f"\n\nConversation History:\n{conversation_history}\n"
        
        # Complete prompt
        prompt = f"""{system_instruction}

{context_text}{history_text}

User Question: {question}

Assistant:"""
        
        return prompt
        
    async def generate(
        self, 
        question: str, 
        contexts: List[str], 
        conversation_history: str = "",
        **kwargs
    ) -> str:
        """
        Generate response using Mistral-7B-Instruct via Ollama.
        
        Args:
            question: User question
            contexts: Retrieved document chunks
            conversation_history: Previous conversation turns
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        try:
            # Check availability
            if not self._is_available:
                available = await self.check_availability()
                if not available:
                    raise Exception("Ollama service not available")
            
            # Create prompt
            prompt = self._create_rag_prompt(question, contexts, conversation_history)
            
            # Prepare request payload
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.config.max_tokens,
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                    "top_k": kwargs.get("top_k", self.config.top_k),
                    "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
                }
            }
            
            logger.info(f"Generating response with Ollama for question: {question[:50]}...")
            start_time = time.time()
            
            # Make request to Ollama
            response = await self.client.post(
                f"{self.config.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            # Parse response
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.2f}s")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
            
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            # First check availability to update _is_available status
            is_available = await self.check_availability()
            
            if is_available:
                return {
                    "model": self.config.model_name,
                    "status": "loaded",
                    "type": "ollama",
                    "provider": "mistral",
                    "parameters": "7.2B",
                    "format": "gguf"
                }
            else:
                return {
                    "model": self.config.model_name,
                    "status": "not_available",
                    "type": "ollama",
                    "provider": "mistral",
                    "error": "Model not found or service not responding"
                }
                
        except Exception as e:
            return {
                "model": self.config.model_name,
                "status": "error",
                "type": "ollama",
                "provider": "mistral",
                "error": str(e)
            }
            
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
        logger.info("Ollama service client closed")
