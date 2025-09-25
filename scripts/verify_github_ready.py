#!/usr/bin/env python3
"""
GitHub Repository Verification Script
Verifies that the repository is safe to push to GitHub without exposing sensitive data.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def print_header():
    """Print verification header."""
    print("=" * 60)
    print("🔍 GitHub Repository Safety Verification")
    print("=" * 60)
    print()

def check_git_status():
    """Check what files are staged and ready to commit."""
    print("📋 Checking git status...")
    
    try:
        # Check staged files
        result = subprocess.run(
            ["git", "status", "--porcelain"], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("❌ Not in a git repository or git error")
            return False
        
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        if not lines:
            print("✅ No files staged for commit")
            return True
        
        print(f"📄 Found {len(lines)} files in git status:")
        
        staged_files = []
        untracked_files = []
        modified_files = []
        
        for line in lines:
            status = line[:2]
            filename = line[3:]
            
            if status[0] in ['A', 'M']:  # Added or Modified
                staged_files.append(filename)
            elif status[1] in ['M', '?']:  # Modified or Untracked
                if status[1] == '?':
                    untracked_files.append(filename)
                else:
                    modified_files.append(filename)
        
        # Check for sensitive files
        sensitive_patterns = [
            'data/documents/',
            '.env',
            'secrets.json',
            'config.json',
            'processed_chunks.json',
            'conversations/',
            'chat_history/',
            '*.html',
            '*.pdf',
            '*.docx'
        ]
        
        sensitive_files = []
        
        for file in staged_files + untracked_files + modified_files:
            for pattern in sensitive_patterns:
                if pattern.replace('*', '') in file or file.endswith(pattern.replace('*', '')):
                    sensitive_files.append(file)
                    break
        
        if sensitive_files:
            print("\n🚨 SENSITIVE FILES DETECTED:")
            for file in sensitive_files:
                print(f"   ❌ {file}")
            print("\n⚠️  DO NOT COMMIT THESE FILES!")
            print("   They contain sensitive enterprise data.")
            return False
        else:
            print("\n✅ No sensitive files detected")
            return True
            
    except FileNotFoundError:
        print("❌ Git not available")
        return False

def check_gitignore():
    """Verify .gitignore is properly configured."""
    print("\n🔒 Checking .gitignore configuration...")
    
    gitignore_file = Path(".gitignore")
    if not gitignore_file.exists():
        print("❌ .gitignore file not found!")
        return False
    
    with open(gitignore_file, 'r') as f:
        content = f.read()
    
    required_patterns = [
        "data/documents/",
        "*.html",
        "*.pdf", 
        "*.docx",
        "processed_chunks.json",
        ".env",
        "models_cache/"
    ]
    
    missing_patterns = []
    for pattern in required_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print("❌ Missing required patterns in .gitignore:")
        for pattern in missing_patterns:
            print(f"   - {pattern}")
        return False
    else:
        print("✅ .gitignore is properly configured")
        return True

def check_document_directories():
    """Check if enterprise documents exist and are ignored."""
    print("\n📁 Checking document directories...")
    
    # Check if enterprise documents exist
    docs_dir = Path("data/documents")
    enterprise_docs = []
    
    if docs_dir.exists():
        for ext in ['.html', '.pdf', '.docx', '.doc']:
            docs = list(docs_dir.glob(f"*{ext}"))
            enterprise_docs.extend(docs)
    
    if enterprise_docs:
        print(f"📄 Found {len(enterprise_docs)} enterprise document(s)")
        
        # Check if they're ignored
        try:
            result = subprocess.run(
                ["git", "check-ignore", "data/documents/"], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print("✅ Enterprise documents are properly ignored")
                return True
            else:
                print("❌ Enterprise documents are NOT ignored!")
                print("   They could be accidentally committed.")
                return False
        except FileNotFoundError:
            print("⚠️  Git not available - cannot verify ignore status")
            return True
    else:
        print("📄 No enterprise documents found")
        return True

def check_sample_data():
    """Check if sample data is available for demonstration."""
    print("\n📝 Checking sample data...")
    
    sample_dir = Path("data/sample_documents")
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*"))
        if sample_files:
            print(f"✅ Found {len(sample_files)} sample file(s)")
            return True
    
    print("⚠️  No sample data found")
    print("   Consider adding sample documents for demonstration")
    return True

def check_configuration_files():
    """Check configuration files for sensitive data."""
    print("\n⚙️ Checking configuration files...")
    
    config_files = [
        ".env",
        "config.json", 
        "secrets.json",
        "enterprise_config.json"
    ]
    
    sensitive_configs = []
    for config_file in config_files:
        if Path(config_file).exists():
            sensitive_configs.append(config_file)
    
    if sensitive_configs:
        print("🚨 Sensitive configuration files found:")
        for config in sensitive_configs:
            print(f"   ❌ {config}")
        print("   These should not be committed to git!")
        return False
    else:
        print("✅ No sensitive configuration files found")
        return True

def check_repository_structure():
    """Check if repository has proper structure for GitHub."""
    print("\n🏗️ Checking repository structure...")
    
    required_files = [
        "README.md",
        "LICENSE",
        ".gitignore",
        "backend/requirements.txt",
        "backend/app/main.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ Repository structure is complete")
        return True

def print_recommendations():
    """Print recommendations for GitHub push."""
    print("\n" + "=" * 60)
    print("📋 GitHub Push Recommendations")
    print("=" * 60)
    print()
    print("✅ SAFE TO PUSH:")
    print("   - Python source code")
    print("   - Configuration templates")
    print("   - Documentation files")
    print("   - Sample data")
    print()
    print("❌ NEVER PUSH:")
    print("   - Enterprise documents")
    print("   - Processed data files")
    print("   - Configuration files with secrets")
    print("   - Conversation history")
    print("   - Model cache files")
    print()
    print("🔧 Before pushing:")
    print("   1. Run: git status")
    print("   2. Review staged files carefully")
    print("   3. Ensure no sensitive data is included")
    print("   4. Test with: git diff --cached")
    print()
    print("🚀 Push commands:")
    print("   git add .")
    print("   git commit -m 'Initial commit: PAC RAG Chatbot'")
    print("   git push origin main")

def main():
    """Main verification function."""
    print_header()
    
    # Track verification progress
    checks = [
        ("Git Status", check_git_status),
        ("Gitignore Configuration", check_gitignore),
        ("Document Directories", check_document_directories),
        ("Sample Data", check_sample_data),
        ("Configuration Files", check_configuration_files),
        ("Repository Structure", check_repository_structure),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            success = check_func()
            if not success:
                failed_checks.append(check_name)
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            failed_checks.append(check_name)
    
    # Print final status
    print("\n" + "=" * 60)
    if failed_checks:
        print("⚠️  Repository verification FAILED!")
        print("\nFailed checks:")
        for check in failed_checks:
            print(f"   - {check}")
        print("\n🚨 DO NOT PUSH TO GITHUB until issues are resolved!")
        print("   Review the SECURITY_GUIDE.md for detailed instructions.")
    else:
        print("✅ Repository verification PASSED!")
        print("\n🎉 Your repository is safe to push to GitHub!")
        print("   All sensitive data is properly protected.")
    
    print_recommendations()

if __name__ == "__main__":
    main()
