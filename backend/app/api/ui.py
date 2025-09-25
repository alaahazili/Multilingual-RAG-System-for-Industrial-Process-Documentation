from fastapi import APIRouter, Response

router = APIRouter()

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OCP Assistant</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root{
      --white:#ffffff; --black:#000000; --green:#2E7D32; --light-green:#E8F5E9; --dark-green:#1B5E20;
      --gray:#F9F9F9; --dark-gray:#333333; --border:#e0e0e0;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:var(--gray);color:var(--black);font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;font-size:15px;line-height:1.5}
    .app{display:grid;grid-template-columns:320px 1fr;min-height:100vh}
    
    /* Sidebar */
    .sidebar{background:var(--white);border-right:1px solid var(--border);padding:32px 24px;box-shadow:2px 0 4px rgba(0,0,0,0.05)}
    .logo{display:flex;align-items:center;margin-bottom:40px;padding-bottom:24px;border-bottom:1px solid var(--border)}
    .logo img{height:40px;margin-right:16px;flex-shrink:0;border-radius:8px}
    .logo-text{font-size:20px;font-weight:600;color:var(--green);line-height:1.2}
    .section{margin-bottom:32px}
    .section-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;color:var(--dark-gray);margin-bottom:12px;opacity:0.8}
    .select{width:100%;padding:14px 16px;border:2px solid var(--light-green);border-radius:8px;background:var(--white);color:var(--black);font-size:14px;transition:all 0.2s ease}
    .select:focus{outline:none;border-color:var(--green);box-shadow:0 0 0 3px rgba(46,125,50,0.1)}
    .pill{display:inline-block;background:var(--light-green);color:var(--green);border-radius:6px;padding:8px 14px;margin:4px 6px 4px 0;font-size:12px;font-weight:500;border:1px solid rgba(46,125,50,0.2)}
    .info-text{font-size:13px;color:var(--dark-gray);line-height:1.5;opacity:0.8}
    
    /* Main Content */
    .content{display:flex;flex-direction:column;height:100vh;background:var(--gray)}
    .chat{flex:1;overflow:auto;padding:24px;background:var(--white);margin:16px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);border:1px solid var(--light-green)}
    
    /* Chat Bubbles */
    .bubble{max-width:75%;padding:16px 20px;border-radius:16px;margin:16px 0;line-height:1.6;font-size:15px}
    .user{background:var(--green);color:var(--white);margin-left:auto;text-align:right;border-bottom-right-radius:6px;box-shadow:0 2px 8px rgba(46,125,50,0.2)}
    .bot{background:var(--white);color:var(--black);border:1px solid var(--light-green);box-shadow:0 2px 8px rgba(0,0,0,0.1);border-bottom-left-radius:6px}
    
    /* Sources */
    .sources{margin-top:12px;padding:12px;background:var(--light-green);border-radius:8px;border-left:3px solid var(--green)}
    .source-item{font-size:12px;color:var(--green);margin:4px 0;font-style:italic}
    
    /* Composer */
    .composer{display:flex;gap:12px;padding:16px;background:var(--white);margin:16px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
    .composer .input{flex:1;padding:16px;border:2px solid var(--light-green);border-radius:12px;font-size:15px;min-width:0;transition:border-color 0.2s}
    .composer .input:focus{outline:none;border-color:var(--green)}
    .composer .input::placeholder{color:#999}
    .composer .btn{flex-shrink:0;width:100px;padding:16px;background:var(--green);color:var(--white);border:none;border-radius:12px;font-size:15px;font-weight:500;cursor:pointer;transition:all 0.2s}
    .composer .btn:hover{background:var(--dark-green);transform:translateY(-1px);box-shadow:0 4px 12px rgba(46,125,50,0.3)}
    .composer .btn:disabled{background:var(--border);cursor:not-allowed;transform:none;box-shadow:none}
    
    /* Welcome Message */
    .welcome{text-align:center;color:var(--dark-gray);font-size:16px;margin:40px 0;background:var(--white);padding:32px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid var(--light-green)}
    .welcome h2{color:var(--green);margin-bottom:16px;font-size:24px;font-weight:600}
    .welcome p{margin-bottom:16px;line-height:1.6}
    .welcome ul{text-align:left;margin:16px 0;padding-left:24px}
    .welcome li{margin:8px 0;color:#555}
    .welcome .example{background:var(--light-green);padding:16px;border-radius:8px;margin-top:16px;border-left:3px solid var(--green)}
    
    /* Metadata */
    .meta{font-size:12px;color:var(--green);margin-top:8px;font-style:italic}
    
    /* Responsive */
    @media (max-width: 768px) {
      .app{grid-template-columns:1fr}
      .sidebar{order:2;border-right:none;border-top:1px solid var(--border)}
      .bubble{max-width:90%}
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="logo">
        <img src="/api/ocp-logo" alt="OCP" />
        <div class="logo-text">OCP Assistant</div>
      </div>
      
      <div class="section">
        <div class="section-title">Document Selection</div>
        <select id="docSelect" class="select">
          <option value="">All Documents</option>
        </select>
      </div>
      
      <div class="section">
        <div class="section-title">Section Filter</div>
        <select id="sectionSelect" class="select">
          <option value="">All Sections</option>
        </select>
      </div>
      
      <div class="section">
        <div class="section-title">System Info</div>
        <div class="pill">Multilingual E5</div>
        <div class="pill">Qdrant Vector DB</div>
        <div class="pill">1024 Dimensions</div>
      </div>
      
      <div class="section">
        <div class="section-title">Instructions</div>
        <div class="info-text">Select a specific document or section to filter your search and get more targeted responses from the OCP technical documentation.</div>
      </div>
    </aside>

    <main class="content">
      <div id="chat" class="chat">
      </div>
      <div class="composer">
        <input id="msg" class="input" placeholder="Type your question about OCP docs..." />
        <button id="sendBtn" class="btn" onclick="send()">Send</button>
      </div>
    </main>
  </div>

  <script>
    const chatEl = document.getElementById('chat');
    const msgEl = document.getElementById('msg');
    const sendBtn = document.getElementById('sendBtn');
    const docSel = document.getElementById('docSelect');
    const resultsEl = document.getElementById('results');
    const state = { messages: [], sessionId: null };

    function addMessage(role, text, sources=[]) {
      state.messages.push({role, text, sources});
      render();
    }

    function render(){
      chatEl.innerHTML = '';
      
      // Show welcome message if no messages exist
      if(state.messages.length === 0) {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome';
        welcomeDiv.innerHTML = `
          <h2>👋 Welcome to OCP Assistant</h2>
          <p>I'm here to help you explore OCP technical documentation.</p>
          <ul>
            <li>✅ Ask questions in plain language</li>
            <li>✅ Filter by document or section</li>
            <li>✅ Get answers with references to the original docs</li>
          </ul>
          <div class="example">
            💡 <strong>Example:</strong> "What is the operating pressure in Main_Pipeline section 5.2?"
          </div>
        `;
        chatEl.appendChild(welcomeDiv);
        return;
      }
      
      // Render messages
      for(const m of state.messages){
        const wrap = document.createElement('div');
        wrap.className = 'bubble ' + (m.role==='user'?'user':'bot');
        wrap.innerText = m.text;
        chatEl.appendChild(wrap);
        if(m.role==='assistant' && m.sources && m.sources.length){
          const src = document.createElement('div');
          src.className='sources meta';
          src.innerHTML = m.sources.map(s => {
            const sectionInfo = s.section || 'General Document';
            return `<div class="source-item">• <strong>${s.document}</strong> — ${sectionInfo}</div>`
          }).join('');
          chatEl.appendChild(src);
        }
      }
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    async function loadDocs(){
      const r = await fetch('/api/documents');
      const data = await r.json();
      const docs = data.documents || [];
      docSel.innerHTML = '<option value="">All documents</option>';
      for(const d of docs){
        const opt = document.createElement('option');
        opt.value = d.doc_code; opt.textContent = d.document || d.doc_code;
        docSel.appendChild(opt);
      }
    }

    async function send(){
      const text = msgEl.value.trim();
      if(!text) return;
      msgEl.value='';
      addMessage('user', text);
      sendBtn.disabled = true; sendBtn.textContent='...';
      try{
        const docCode = docSel.value || null;
        const sectionCode = document.getElementById('sectionSelect').value || null;
        const requestBody = {
          message: text, 
          limit: 6, 
          document_filter: docCode,
          session_id: state.sessionId
        };
        const res = await fetch('/api/chat', {
          method:'POST', 
          headers:{'Content-Type':'application/json'}, 
          body: JSON.stringify(requestBody)
        });
        const data = await res.json();
        
        // Store session ID from response if provided
        if (data.session_id && !state.sessionId) {
          state.sessionId = data.session_id;
        }
        
        addMessage('assistant', data.answer || '(no answer)', data.sources || []);
      }catch(e){
        addMessage('assistant', 'Error: '+e);
      }finally{
        sendBtn.disabled=false; sendBtn.textContent='Send';
      }
    }

    msgEl.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
    loadDocs();
    render(); // Show welcome message on page load
  </script>
</body>
</html>
"""

@router.get("/", include_in_schema=False)
async def ui_root() -> Response:
  return Response(content=HTML, media_type="text/html")

@router.get("/api/ocp-logo", include_in_schema=False)
async def serve_logo() -> Response:
  import os
  logo_path = os.path.join(os.path.dirname(__file__), "Ocp-group.png")
  if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
      return Response(content=f.read(), media_type="image/png")
  else:
    # Return a simple SVG placeholder if logo not found
    svg = '''<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
      <rect width="40" height="40" fill="#00a651"/>
      <text x="20" y="25" text-anchor="middle" fill="white" font-family="Arial" font-size="16" font-weight="bold">OCP</text>
    </svg>'''
    return Response(content=svg, media_type="image/svg+xml")


