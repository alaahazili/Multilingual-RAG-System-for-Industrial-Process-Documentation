#!/usr/bin/env python3
"""
PAC RAG Chatbot Setup Script
Automated setup and verification script for the RAG chatbot system.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("🤖 PAC RAG Chatbot Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_dependencies():
    """Check if required system dependencies are available."""
    print("\n📦 Checking system dependencies...")
    
    dependencies = {
        "docker": "Docker (for Qdrant)",
        "git": "Git (for version control)"
    }
    
    all_available = True
    for cmd, description in dependencies.items():
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {description} - Available")
            else:
                print(f"❌ {description} - Not found")
                all_available = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {description} - Not found")
            all_available = False
    
    return all_available

def create_virtual_environment():
    """Create Python virtual environment."""
    print("\n🔧 Setting up Python virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_python_dependencies():
    """Install Python dependencies."""
    print("\n📥 Installing Python dependencies...")
    
    requirements_file = Path("backend/requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Use the virtual environment's pip
        if os.name == 'nt':  # Windows
            pip_path = Path("venv/Scripts/pip.exe")
        else:  # Unix-like
            pip_path = Path("venv/bin/pip")
        
        if not pip_path.exists():
            print("❌ Virtual environment pip not found")
            return False
        
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], 
                      check=True, timeout=300)
        print("✅ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False

def check_qdrant():
    """Check if Qdrant is running."""
    print("\n🗄️ Checking Qdrant database...")
    
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
        else:
            print("❌ Qdrant is not responding correctly")
            return False
    except ImportError:
        print("⚠️ requests not available, skipping Qdrant check")
        return True
    except Exception as e:
        print(f"❌ Qdrant is not running: {e}")
        print("💡 Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False

def check_ollama():
    """Check if Ollama is running and has the required model."""
    print("\n🦙 Checking Ollama...")
    
    try:
        import requests
        
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("❌ Ollama is not running")
                print("💡 Install Ollama from https://ollama.ai and run: ollama serve")
                return False
        except Exception:
            print("❌ Ollama is not running")
            print("💡 Install Ollama from https://ollama.ai and run: ollama serve")
            return False
        
        # Check if mistral model is available
        models = response.json().get("models", [])
        mistral_models = [m for m in models if "mistral" in m.get("name", "").lower()]
        
        if mistral_models:
            print("✅ Ollama is running with Mistral model")
            return True
        else:
            print("⚠️ Ollama is running but Mistral model not found")
            print("💡 Install Mistral with: ollama pull mistral:instruct")
            return False
            
    except ImportError:
        print("⚠️ requests not available, skipping Ollama check")
        return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "models_cache",
        "data/processed",
        "static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env file from template."""
    print("\n⚙️ Setting up environment configuration...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create basic .env file
    env_content = """# PAC RAG Chatbot Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documents

# Model Configuration
MODEL_CACHE_DIR=models_cache

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:instruct

# Embedding Configuration
EMBEDDING_MODEL=intfloat/multilingual-e5-large
VECTOR_SIZE=1024
EMBEDDING_BATCH_SIZE=4

# Retrieval Configuration
TOP_K=10
RERANK_TOP_K=5
SIMILARITY_THRESHOLD=0.1
USE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Conversation Settings
MAX_CONVERSATION_HISTORY=5
SESSION_TIMEOUT=30
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def run_tests():
    """Run basic tests to verify setup."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Add backend to Python path
        backend_path = Path("backend").absolute()
        sys.path.insert(0, str(backend_path))
        
        # Try importing main modules
        test_imports = [
            "app.main",
            "app.services.embedding_layer.embedding_service",
            "app.services.retrieval_layer.retrieval_service",
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                print(f"✅ {module} - Import successful")
            except ImportError as e:
                print(f"❌ {module} - Import failed: {e}")
                return False
        
        print("✅ All basic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete! Next Steps:")
    print("=" * 60)
    print()
    print("1. 🔒 Set up enterprise documents (if needed):")
    print("   python scripts/setup_enterprise_docs.py")
    print()
    print("2. 📄 Process documents:")
    print("   python scripts/process_html_document.py")
    print("   python scripts/create_embeddings.py")
    print()
    print("3. 🚀 Start the application:")
    print("   cd backend")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("4. 🌐 Access the chatbot:")
    print("   Web Interface: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print()
    print("5. 🧪 Run full tests:")
    print("   python scripts/test_complete_rag_architecture.py")
    print()
    print("📚 For more information, check the README.md file")
    print("🔒 For security guidelines, check SECURITY_GUIDE.md")
    print("🆘 For help, check CONTRIBUTING.md or open an issue")

def main():
    """Main setup function."""
    print_header()
    
    # Track setup progress
    steps = [
        ("Python Version", check_python_version),
        ("System Dependencies", check_dependencies),
        ("Virtual Environment", create_virtual_environment),
        ("Python Dependencies", install_python_dependencies),
        ("Directories", create_directories),
        ("Environment Config", create_env_file),
        ("Qdrant Database", check_qdrant),
        ("Ollama Service", check_ollama),
        ("Basic Tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        try:
            success = step_func()
            if not success:
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    print("\n" + "=" * 60)
    if failed_steps:
        print("⚠️ Setup completed with issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease resolve the issues above before proceeding.")
    else:
        print("✅ Setup completed successfully!")
        print_next_steps()

if __name__ == "__main__":
    main()
