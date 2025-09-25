#!/usr/bin/env python3
"""
Test Imports Script
Tests all imports to ensure the data layer works correctly.
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def test_imports():
    """Test all imports from the data layer"""
    print("🔍 Testing Data Layer Imports...")
    print("=" * 50)
    
    try:
        # Test HTML Processor
        print("📄 Testing HTML Processor...")
        from app.services.data_layer.html_processor import HTMLProcessor
        print("✅ HTMLProcessor imported successfully")
        
        # Test Document Chunker
        print("📝 Testing Document Chunker...")
        from app.services.data_layer.chunker import DocumentChunker, ChunkConfig
        print("✅ DocumentChunker imported successfully")
        
        # Test Table Formula Processor
        print("📊 Testing Table Formula Processor...")
        from app.services.data_layer.table_formula_processor import TableFormulaProcessor
        print("✅ TableFormulaProcessor imported successfully")
        
        # Test Data Layer Init
        print("📦 Testing Data Layer Init...")
        from app.services.data_layer import HTMLProcessor as HTMLProcessor2, DocumentChunker as DocumentChunker2, TableFormulaProcessor as TableFormulaProcessor2
        print("✅ Data layer __init__ imported successfully")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        print("✅ Data layer is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_html_processor():
    """Test HTML processor initialization"""
    print("\n🔍 Testing HTML Processor Initialization...")
    print("=" * 50)
    
    try:
        from app.services.data_layer.html_processor import HTMLProcessor
        
        # Initialize processor
        processor = HTMLProcessor()
        print("✅ HTMLProcessor initialized successfully")
        
        # Get processing stats
        stats = processor.get_processing_stats()
        print(f"✅ Processing stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ HTML Processor test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_chunker():
    """Test document chunker initialization"""
    print("\n🔍 Testing Document Chunker Initialization...")
    print("=" * 50)
    
    try:
        from app.services.data_layer.chunker import DocumentChunker, ChunkConfig
        
        # Initialize chunker
        config = ChunkConfig()
        chunker = DocumentChunker(config)
        print("✅ DocumentChunker initialized successfully")
        
        # Get chunking stats
        stats = chunker.get_chunking_stats()
        print(f"✅ Chunking stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document Chunker test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main test function"""
    print("🚀 STARTING IMPORT TESTS")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test HTML processor
        html_ok = test_html_processor()
        
        # Test chunker
        chunker_ok = test_chunker()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Imports Test: {'✅ PASSED' if imports_ok else '❌ FAILED'}")
        print(f"HTML Processor Test: {'✅ PASSED' if html_ok else '❌ FAILED'}")
        print(f"Document Chunker Test: {'✅ PASSED' if chunker_ok else '❌ FAILED'}")
        
        if all([imports_ok, html_ok, chunker_ok]):
            print("\n🎉 ALL TESTS PASSED! Data layer is ready to use.")
            print("✅ You can now proceed with HTML document processing.")
        else:
            print("\n⚠️ Some tests failed. Please check the errors above.")
    else:
        print("\n❌ Import tests failed. Cannot proceed with other tests.")

if __name__ == "__main__":
    main()
