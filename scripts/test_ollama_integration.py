#!/usr/bin/env python3
"""
Test Ollama Integration
Verifies that Mistral-7B-Instruct is properly connected to the project.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.generative_layer.ollama_generation import OllamaGenerationService, OllamaConfig


async def test_ollama_connection():
    """Test Ollama connection and model availability"""
    print("🔍 Testing Ollama Connection...")
    
    try:
        # Initialize Ollama service
        config = OllamaConfig(
            base_url="http://localhost:11434",
            model_name="mistral:instruct",
            timeout=30
        )
        
        ollama_service = OllamaGenerationService(config)
        
        # Check availability
        print("📡 Checking Ollama availability...")
        is_available = await ollama_service.check_availability()
        
        if is_available:
            print("✅ Ollama service is available!")
            
            # Get model info
            print("📊 Getting model information...")
            model_info = await ollama_service.get_model_info()
            print(f"✅ Model: {model_info.get('model')}")
            print(f"✅ Status: {model_info.get('status')}")
            print(f"✅ Parameters: {model_info.get('parameters')}")
            
            return True
        else:
            print("❌ Ollama service not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False


async def test_generation():
    """Test text generation with Ollama"""
    print("\n🤖 Testing Text Generation...")
    
    try:
        # Initialize service
        ollama_service = OllamaGenerationService()
        
        # Test question and context
        test_question = "What is the PAC control philosophy?"
        test_contexts = [
            "PAC stands for Process Automation Control system. It manages industrial processes including slurry pipelines, pump stations, and control systems.",
            "The control philosophy includes supervisory control, data acquisition (SCADA), telecommunications, and pipeline advisory software."
        ]
        
        print(f"📝 Question: {test_question}")
        print("🔄 Generating response...")
        
        # Generate response
        response = await ollama_service.generate(
            question=test_question,
            contexts=test_contexts
        )
        
        print("✅ Generation successful!")
        print(f"📄 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing generation: {e}")
        return False


async def test_rag_integration():
    """Test RAG integration with conversation memory"""
    print("\n🔗 Testing RAG Integration...")
    
    try:
        # Initialize service
        ollama_service = OllamaGenerationService()
        
        # Test conversation
        conversation_history = "User: What is PAC?\nAssistant: PAC stands for Process Automation Control system."
        
        test_question = "What are the main components?"
        test_contexts = [
            "The main components include feeder pipelines, head station, main line, and terminal station.",
            "Control systems include SCADA, telecommunications, and pipeline advisory software."
        ]
        
        print(f"📝 Follow-up Question: {test_question}")
        print("🔄 Generating response with conversation history...")
        
        # Generate response with conversation history
        response = await ollama_service.generate(
            question=test_question,
            contexts=test_contexts,
            conversation_history=conversation_history
        )
        
        print("✅ RAG integration successful!")
        print(f"📄 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG integration: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 STARTING OLLAMA INTEGRATION TEST")
    print("=" * 50)
    
    # Test connection
    connection_ok = await test_ollama_connection()
    
    if connection_ok:
        # Test generation
        generation_ok = await test_generation()
        
        # Test RAG integration
        rag_ok = await test_rag_integration()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Connection Test: {'✅ PASSED' if connection_ok else '❌ FAILED'}")
        print(f"Generation Test: {'✅ PASSED' if generation_ok else '❌ FAILED'}")
        print(f"RAG Integration Test: {'✅ PASSED' if rag_ok else '❌ FAILED'}")
        
        if all([connection_ok, generation_ok, rag_ok]):
            print("\n🎉 ALL TESTS PASSED! Ollama integration is working perfectly!")
            print("✅ Your project is ready to use Mistral-7B-Instruct via Ollama!")
        else:
            print("\n⚠️ Some tests failed. Please check the errors above.")
    else:
        print("\n❌ Connection test failed. Please ensure Ollama is running with mistral:instruct.")


if __name__ == "__main__":
    asyncio.run(main())
