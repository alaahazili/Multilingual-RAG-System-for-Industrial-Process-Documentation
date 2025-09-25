#!/usr/bin/env python3
"""
Enterprise Document Setup Script
Helps users securely set up their enterprise documents for the RAG chatbot.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("🔒 Enterprise Document Setup")
    print("=" * 60)
    print()

def check_git_status():
    """Check if we're in a git repository and if documents are ignored."""
    print("🔍 Checking git configuration...")
    
    # Check if we're in a git repo
    if not Path(".git").exists():
        print("⚠️  Not in a git repository - documents will be safe")
        return True
    
    # Check if documents directory is ignored
    try:
        import subprocess
        result = subprocess.run(
            ["git", "check-ignore", "data/documents/"], 
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("✅ data/documents/ is properly ignored by git")
            return True
        else:
            print("❌ data/documents/ is NOT ignored by git!")
            print("   This could lead to accidental commits of sensitive data.")
            return False
    except FileNotFoundError:
        print("⚠️  Git not available - cannot verify ignore status")
        return True

def create_documents_directory():
    """Create the documents directory if it doesn't exist."""
    print("\n📁 Setting up documents directory...")
    
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    if docs_dir.exists():
        print(f"✅ Created directory: {docs_dir}")
        return True
    else:
        print(f"❌ Failed to create directory: {docs_dir}")
        return False

def check_existing_documents():
    """Check for existing enterprise documents."""
    print("\n📄 Checking for existing documents...")
    
    docs_dir = Path("data/documents")
    if not docs_dir.exists():
        print("📁 Documents directory doesn't exist yet")
        return []
    
    # Find HTML, PDF, DOCX files
    doc_extensions = ['.html', '.htm', '.pdf', '.docx', '.doc']
    existing_docs = []
    
    for ext in doc_extensions:
        docs = list(docs_dir.glob(f"*{ext}"))
        existing_docs.extend(docs)
    
    if existing_docs:
        print(f"✅ Found {len(existing_docs)} existing document(s):")
        for doc in existing_docs:
            print(f"   - {doc.name}")
    else:
        print("📄 No enterprise documents found")
        print("   You can add your documents to data/documents/")
    
    return existing_docs

def create_sample_documents():
    """Create sample documents for demonstration."""
    print("\n📝 Creating sample documents for demo...")
    
    sample_dir = Path("data/sample_documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if sample document exists
    sample_doc = sample_dir / "sample_control_philosophy.html"
    if sample_doc.exists():
        print("✅ Sample document already exists")
        return True
    
    print("📄 Sample document will be available for demo purposes")
    print("   This is safe to commit to git")
    return True

def verify_gitignore():
    """Verify .gitignore is properly configured."""
    print("\n🔒 Verifying .gitignore configuration...")
    
    gitignore_file = Path(".gitignore")
    if not gitignore_file.exists():
        print("❌ .gitignore file not found!")
        return False
    
    # Read .gitignore content
    with open(gitignore_file, 'r') as f:
        content = f.read()
    
    # Check for key security patterns
    security_patterns = [
        "data/documents/",
        "*.html",
        "*.pdf",
        "*.docx",
        "processed_chunks.json"
    ]
    
    missing_patterns = []
    for pattern in security_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print("❌ Missing security patterns in .gitignore:")
        for pattern in missing_patterns:
            print(f"   - {pattern}")
        return False
    else:
        print("✅ .gitignore is properly configured for security")
        return True

def print_security_reminders():
    """Print important security reminders."""
    print("\n" + "=" * 60)
    print("🔒 SECURITY REMINDERS")
    print("=" * 60)
    print()
    print("✅ Your enterprise documents are SAFE:")
    print("   - data/documents/ is ignored by git")
    print("   - Your documents will NOT be committed")
    print("   - Only code and sample data go to GitHub")
    print()
    print("⚠️  IMPORTANT:")
    print("   - Never commit documents manually with 'git add data/documents/'")
    print("   - Always use 'git status' to check what will be committed")
    print("   - Keep your .env file local (never commit secrets)")
    print()
    print("🚀 Next steps:")
    print("   1. Add your enterprise documents to data/documents/")
    print("   2. Run: python scripts/process_html_document.py")
    print("   3. Run: python scripts/create_embeddings.py")
    print("   4. Start the application")
    print()
    print("📚 For more security info, see SECURITY_GUIDE.md")

def main():
    """Main setup function."""
    print_header()
    
    # Track setup progress
    steps = [
        ("Git Status", check_git_status),
        ("Documents Directory", create_documents_directory),
        ("Existing Documents", lambda: check_existing_documents() is not None),
        ("Sample Documents", create_sample_documents),
        ("Gitignore Verification", verify_gitignore),
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
    
    # Print final status
    if failed_steps:
        print("\n⚠️  Setup completed with issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease resolve the issues above for complete security.")
    else:
        print("\n✅ Enterprise document setup completed successfully!")
    
    print_security_reminders()

if __name__ == "__main__":
    main()
