#!/usr/bin/env python3
"""
Generate images from Mermaid diagrams for LinkedIn
Requires: pip install playwright
"""

import os
import asyncio
from playwright.async_api import async_playwright

async def generate_mermaid_images():
    """Generate images from Mermaid diagrams"""
    
    # Mermaid diagram codes
    diagrams = {
        "user_pipeline": """
flowchart LR
    U1[👤 User<br/>Asks Question<br/>"What is PAC?"]
    D1[📄 HTML Document<br/>General_Control_Philosophy.html<br/>3,272 lines]
    D2[✂️ Text Chunks<br/>52 chunks<br/>~500 tokens each]
    E1[🧮 E5-Large<br/>intfloat/multilingual-e5-large<br/>1024 dimensions]
    E2[💾 Qdrant DB<br/>Vector Storage<br/>52 shards]
    Q1[❓ User Query<br/>"What is PAC?"]
    Q2[🔍 Vector Search<br/>Top-10 results]
    Q3[🎯 BGE Reranker<br/>BAAI/bge-reranker-v2-m3<br/>Top-5 results]
    G1[🧠 Mistral-7B<br/>mistral:instruct<br/>7.2B parameters]
    G2[💬 Generated Answer<br/>"PAC stands for Process Automation Control..."]
    M1[🧠 Conversation Memory<br/>Last 5 turns]
    M2[📱 Web Interface<br/>Answer + Sources]
    U2[👤 User<br/>Receives Answer<br/>"PAC stands for Process Automation Control..."]
    
    U1 --> Q1
    D1 --> D2
    D2 --> E1
    E1 --> E2
    Q1 --> E1
    E2 --> Q2
    Q2 --> Q3
    Q3 --> G1
    G1 --> G2
    M1 --> G1
    G2 --> M2
    M2 --> U2
    
    classDef user fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef document fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef embedding fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef retrieval fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef generation fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef interface fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    
    class U1,U2 user
    class D1,D2 document
    class E1,E2 embedding
    class Q1,Q2,Q3 retrieval
    class G1,G2 generation
    class M1,M2 interface
        """,
        
        "simplified_flow": """
graph TD
    A[👤 User Question] --> B[E5-Large Embedding]
    B --> C[Qdrant Vector Search]
    C --> D[BGE Reranker]
    D --> E[Mistral-7B Generation]
    E --> F[👤 User Answer]
    
    G[📄 HTML Document] --> H[Text Chunks]
    H --> I[E5-Large Embeddings]
    I --> J[Qdrant Storage]
    
    J -.-> C
    
    classDef user fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef model fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class A,F user
    class B,C,D,E model
    class G,H,I,J data
        """
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        for name, diagram in diagrams.items():
            # Create HTML with Mermaid
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            </head>
            <body>
                <div class="mermaid">
                    {diagram}
                </div>
                <script>
                    mermaid.initialize({{ startOnLoad: true }});
                </script>
            </body>
            </html>
            """
            
            # Save HTML file
            html_file = f"{name}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Navigate to HTML file
            await page.goto(f"file://{os.path.abspath(html_file)}")
            
            # Wait for diagram to render
            await page.wait_for_timeout(3000)
            
            # Take screenshot
            await page.screenshot(path=f"{name}.png", full_page=True)
            print(f"✅ Generated {name}.png")
            
            # Clean up HTML file
            os.remove(html_file)
        
        await browser.close()

if __name__ == "__main__":
    print("🚀 Generating Mermaid diagrams as images...")
    asyncio.run(generate_mermaid_images())
    print("✅ All images generated successfully!")
    print("\n📁 Generated files:")
    print("   - user_pipeline.png")
    print("   - simplified_flow.png")
    print("\n💡 You can now use these images on LinkedIn!")

