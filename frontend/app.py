import os
import requests
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Hybrid RAG Search Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimalist styling: dark-mode responsive, thin typography, black accents, no emojis, clean headers
st.markdown("""
<style>
    /* Remove deploy button on top */
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    .stDeployButton,
    button[data-testid="stBaseButton-header"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Keep the 3-dot Main Menu button visible with native theme choices (Light, Dark, System) */
    #MainMenu, [data-testid="stMainMenu"] {
        visibility: visible !important;
        display: inline-flex !important;
    }

    /* Hide specifically Print, Record Screencast, About, Community, and Dividers from the 3-dot menu list */
    [data-testid="stMainMenuItem-print"],
    [data-testid="stMainMenuItem-recordScreencast"],
    [data-testid="stMainMenuItem-about"],
    [data-testid="stMainMenuItem-community"],
    [data-testid="stMainMenuDivider"],
    hr[data-testid="stMainMenuDivider"],
    li:has([data-testid="stMainMenuItem-print"]),
    li:has([data-testid="stMainMenuItem-recordScreencast"]),
    li:has([data-testid="stMainMenuItem-about"]),
    li:has([data-testid="stMainMenuItem-community"]) {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }

    /* Hide the default "Made with Streamlit v..." text and copy button in 3-dot popover */
    .stMenuVersionCopyButton,
    button.stMenuVersionCopyButton,
    [data-testid="stMainMenuPopover"] .stMenuVersionCopyButton {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        width: 0px !important;
        padding: 0px !important;
        margin: 0px !important;
    }
    
    /* Ensure the original version text span is never visible */
    [data-testid="stMainMenuPopover"] [data-testid="stMainMenuList"] ~ div span,
    [data-testid="stMainMenuPopover"] .eomjd1k18 {
        font-size: 0px !important;
        color: transparent !important;
        visibility: hidden !important;
        display: none !important;
    }

    /* Style the replacement link inside the 3-dot menu popover */
    [data-testid="stMainMenuPopover"] [data-testid="stMainMenuList"] ~ div {
        padding: 8px 16px 10px 16px !important;
        border-top: 1px solid rgba(128, 128, 128, 0.15) !important;
        cursor: pointer !important;
    }
    [data-testid="stMainMenuPopover"] [data-testid="stMainMenuList"] ~ div:not(:has(a))::before {
        content: "Made by Dheeraj" !important;
        font-size: 13px !important;
        font-weight: 300 !important;
        color: var(--text-color) !important;
        opacity: 0.8 !important;
        display: block !important;
        text-decoration: underline !important;
        text-underline-offset: 3px !important;
    }
    [data-testid="stMainMenuPopover"] [data-testid="stMainMenuList"] ~ div a {
        color: var(--text-color) !important;
        text-decoration: underline !important;
        text-underline-offset: 3px !important;
        font-size: 13px !important;
        font-weight: 300 !important;
        opacity: 0.8 !important;
    }

    /* Clean App & Sidebar Footers */
    .app-footer {
        text-align: center;
        padding: 32px 0 20px 0;
        margin-top: 48px;
        font-size: 13px;
        font-weight: 300;
        color: var(--text-color) !important;
        opacity: 0.75;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
    }
    .app-footer a, .sidebar-footer a {
        color: var(--text-color) !important;
        text-decoration: underline !important;
        text-underline-offset: 3px !important;
        font-weight: 400 !important;
        transition: opacity 0.15s ease;
    }
    .app-footer a:hover, .sidebar-footer a:hover {
        opacity: 0.6;
    }
    .sidebar-footer {
        text-align: center;
        font-size: 12px;
        font-weight: 300;
        color: var(--text-color) !important;
        opacity: 0.65;
        margin-top: 24px;
        padding-top: 12px;
    }

    /* Hide Streamlit Footer & Watermark ("Made with Streamlit") */
    footer, [data-testid="stFooter"] {
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }

    /* Hide decorative status widgets */
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* Global Fonts & Weights */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Thin Headings - fully visible in Light and Dark mode */
    h1, h2, h3, h4, .app-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: -0.025em !important;
        color: var(--text-color) !important;
    }
    
    .app-header {
        font-size: 28px;
        font-weight: 300;
        margin-bottom: 2px;
    }
    
    .app-subhead {
        font-size: 13px;
        font-weight: 400;
        color: var(--text-color) !important;
        opacity: 0.65;
        letter-spacing: -0.01em;
        margin-bottom: 20px;
    }

    /* Accent Color: Black buttons */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
        border-radius: 4px !important;
        transition: opacity 0.15s ease;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
        color: #FFFFFF !important;
        opacity: 0.9;
    }

    /* Secondary buttons */
    .stButton > button {
        border-radius: 4px !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        color: var(--text-color) !important;
    }
    .stButton > button:hover {
        border-color: var(--text-color) !important;
        color: var(--text-color) !important;
    }

    /* Tabs active accent */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--text-color) !important;
        border-bottom-color: #000000 !important;
    }

    /* Minimalist status pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 400;
        letter-spacing: 0.02em;
        margin-bottom: 12px;
    }
    .status-pill.online {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-pill.offline {
        background-color: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Theme-adaptive metadata tags */
    .meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 20px;
        font-size: 13px;
        color: var(--text-color) !important;
        opacity: 0.85;
    }
    .meta-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .meta-value {
        font-weight: 500;
        color: var(--text-color) !important;
    }

    /* Citation badge */
    .citation-badge {
        display: inline-block;
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Code & text snippets */
    pre, code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 12px !important;
    }

    /* Theme-adaptive table styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 13px;
        color: var(--text-color) !important;
    }
    th {
        font-weight: 500;
        text-align: left;
        background-color: var(--secondary-background-color) !important;
        padding: 10px 14px;
        border-bottom: 2px solid rgba(128, 128, 128, 0.25) !important;
        color: var(--text-color) !important;
    }
    td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15) !important;
        color: var(--text-color) !important;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Script to replace "Made with Streamlit v..." inside 3-dot menu popover with clickable "Made by Dheeraj"
st.html("""
<script>
(function() {
    function replaceStreamlitBranding() {
        const popovers = document.querySelectorAll('[data-testid="stMainMenuPopover"], .stMainMenuPopover');
        popovers.forEach(pop => {
            const versionRow = pop.querySelector('.stMenuVersionCopyButton')?.closest('div') 
                || pop.querySelector('[data-testid="stMainMenuList"] ~ div')
                || pop.querySelector('.eomjd1k15');
            if (versionRow && !versionRow.dataset.customized) {
                versionRow.dataset.customized = "true";
                versionRow.style.cursor = "pointer";
                versionRow.onclick = (e) => {
                    e.preventDefault();
                    window.open("https://dheerajkumar.pages.dev/", "_blank");
                };
                versionRow.innerHTML = '<a href="https://dheerajkumar.pages.dev/" target="_blank" rel="noopener noreferrer" style="display:inline-block; font-size:13px; font-weight:300; color:var(--text-color, inherit); text-decoration:underline; text-underline-offset:3px; padding:4px 0; opacity:0.85;">Made by Dheeraj</a>';
            }
        });

        const footers = document.querySelectorAll('footer, [data-testid="stFooter"]');
        footers.forEach(f => {
            if (f.innerText && f.innerText.includes('Made with Streamlit') && !f.dataset.customized) {
                f.dataset.customized = "true";
                f.innerHTML = '<div style="text-align:center; padding:16px 0; font-size:13px; font-weight:300; color:var(--text-color); opacity:0.8;">Made by <a href="https://dheerajkumar.pages.dev/" target="_blank" rel="noopener noreferrer" style="color:var(--text-color); text-decoration:underline; text-underline-offset:3px; font-weight:400;">Dheeraj</a></div>';
            }
        });
    }

    replaceStreamlitBranding();
    const observer = new MutationObserver(replaceStreamlitBranding);
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_javascript=True)

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


def check_backend(api_url: str):
    try:
        r = requests.get(f"{api_url}/api/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
        return False, None
    except Exception:
        return False, None


# Curated list of verified active free models on OpenRouter
OPENROUTER_FREE_MODELS = [
    "openrouter/free",
    "liquid/lfm-3-1.7b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "baidu/ernie-5.5-flash-preview:free",
    "dots-studio/dots-3-note-preview:free",
    "particle-ai/particle-1:free",
]


# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    
    api_url = st.text_input("Backend API Endpoint", value=st.session_state.api_url)
    st.session_state.api_url = api_url

    backend_ok, health = check_backend(api_url)
    if backend_ok:
        st.markdown('<div class="status-pill online">Backend Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill offline">Backend Offline</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### LLM Provider")
    
    default_provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    provider_choice = st.radio(
        "Provider",
        options=["OpenRouter (Free Models)", "Google Gemini"],
        index=0 if default_provider == "openrouter" else 1,
        help="OpenRouter offers free models with no credit card required."
    )

    if provider_choice.startswith("OpenRouter"):
        active_provider = "openrouter"
        env_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        # Clean placeholder value if untouched
        if env_openrouter_key.startswith("your_"):
            env_openrouter_key = ""

        llm_api_key = st.text_input(
            "OpenRouter Key",
            value=env_openrouter_key,
            type="password",
            help="Free key from https://openrouter.ai/keys"
        )
        if not llm_api_key:
            st.caption("Provide an OpenRouter API key above or set `OPENROUTER_API_KEY` in `.env`.")

        model_selection = st.selectbox(
            "Free Model",
            options=OPENROUTER_FREE_MODELS + ["Custom..."],
            index=0,
            help="All models ending with :free are free of charge on OpenRouter."
        )

        if model_selection == "Custom...":
            active_model = st.text_input("Custom Model ID", value="meta-llama/llama-3.3-70b-instruct:free")
        else:
            active_model = model_selection
    else:
        active_provider = "gemini"
        env_gemini_key = os.getenv("GEMINI_API_KEY", "")
        llm_api_key = st.text_input(
            "Gemini Key",
            value=env_gemini_key,
            type="password",
            help="Google AI Studio key. Leave blank to use .env value."
        )
        gemini_model_options = [
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3.0-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        active_model = st.selectbox("Model", options=gemini_model_options, index=0)

    st.markdown("---")
    st.markdown("#### Retrieval Parameters")
    top_k_retrieval = st.slider("Retrieval Candidates", min_value=5, max_value=30, value=10, step=1,
                                help="Candidate passages retrieved from FAISS + BM25 via RRF")
    top_k_rerank = st.slider("Reranked Top-K", min_value=1, max_value=10, value=5, step=1,
                             help="Passages sent to LLM prompt after Cross-Encoder scoring")

    st.markdown("---")
    if st.button("Reset Storage Indexes", help="Purges all indexed documents and vector files"):
        if backend_ok:
            try:
                requests.post(f"{api_url}/api/reset")
                st.session_state.messages = []
                st.success("Storage cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Backend offline.")

    st.markdown(
        '<div class="sidebar-footer">Made by <a href="https://dheerajkumar.pages.dev/" target="_blank" rel="noopener noreferrer">Dheeraj</a></div>',
        unsafe_allow_html=True
    )


# Header
st.markdown('<div class="app-header">Hybrid RAG Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subhead">Dense vector search, BM25 keyword matching, Reciprocal Rank Fusion, and Cross-Encoder reranking with grounded citations.</div>', unsafe_allow_html=True)

# Telemetry summary
if backend_ok and health:
    st.markdown(f"""
    <div class="meta-bar">
        <div class="meta-item">Documents: <span class="meta-value">{health.get('documents_count', 0)}</span></div>
        <div class="meta-item">Chunks: <span class="meta-value">{health.get('total_chunks', 0)}</span></div>
        <div class="meta-item">Provider: <span class="meta-value">{active_provider.title()}</span></div>
        <div class="meta-item">Model: <span class="meta-value">{active_model.split('/')[-1]}</span></div>
    </div>
    """, unsafe_allow_html=True)

# Main Navigation Tabs
tab_chat, tab_docs, tab_inspector, tab_docs_page = st.tabs([
    "Chat",
    "Documents",
    "Inspector",
    "Documentation"
])


# ----------------------------------------------------
# TAB 1: Chat
# ----------------------------------------------------
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown("**Sources:**")
                badges = "".join([
                    f"<span class='citation-badge'>{c['doc_name']}, p. {c['page_number']}</span>"
                    for c in msg["citations"]
                ])
                st.markdown(badges, unsafe_allow_html=True)

            if msg.get("retrieved_passages"):
                with st.expander(f"View Evidence Passages ({len(msg['retrieved_passages'])})"):
                    for idx, p in enumerate(msg["retrieved_passages"], start=1):
                        st.markdown(f"**Passage {idx}** — `{p['doc_name']}, Page {p['page_number']}`")
                        scores_line = f"Score: {p.get('rerank_score', 'N/A')} | RRF: {p.get('rrf_score', 0):.5f}"
                        if p.get('vector_rank'):
                            scores_line += f" | Vector Rank: #{p['vector_rank']}"
                        if p.get('bm25_rank'):
                            scores_line += f" | BM25 Rank: #{p['bm25_rank']}"
                        st.caption(scores_line)
                        st.code(p["text"], language=None)

    query = st.chat_input("Enter question about indexed documents...")
    if query:
        if not backend_ok:
            st.error("Backend offline. Start backend using: python -m backend.main")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner(f"Retrieving evidence and generating grounded answer using {active_model.split('/')[-1]}..."):
                    try:
                        payload = {
                            "query": query,
                            "top_k_retrieval": top_k_retrieval,
                            "top_k_rerank": top_k_rerank,
                            "provider": active_provider,
                            "api_key": llm_api_key if llm_api_key else None,
                            "model_name": active_model
                        }
                        resp = requests.post(f"{api_url}/api/query", json=payload, timeout=90)
                        if resp.status_code == 200:
                            data = resp.json()
                            answer = data.get("answer", "No response generated.")
                            citations = data.get("citations", [])
                            passages = data.get("retrieved_passages", [])
                            has_sufficient = data.get("has_sufficient_context", True)

                            if not has_sufficient:
                                st.warning("Documents do not contain sufficient information to answer this question.")

                            st.markdown(answer)

                            if citations:
                                st.markdown("**Sources:**")
                                badges = "".join([
                                    f"<span class='citation-badge'>{c['doc_name']}, p. {c['page_number']}</span>"
                                    for c in citations
                                ])
                                st.markdown(badges, unsafe_allow_html=True)

                            if passages:
                                with st.expander(f"View Evidence Passages ({len(passages)})"):
                                    for idx, p in enumerate(passages, start=1):
                                        st.markdown(f"**Passage {idx}** — `{p['doc_name']}, Page {p['page_number']}`")
                                        scores_line = f"Score: {p.get('rerank_score', 'N/A')} | RRF: {p.get('rrf_score', 0):.5f}"
                                        if p.get('vector_rank'):
                                            scores_line += f" | Vector Rank: #{p['vector_rank']}"
                                        if p.get('bm25_rank'):
                                            scores_line += f" | BM25 Rank: #{p['bm25_rank']}"
                                        st.caption(scores_line)
                                        st.code(p["text"], language=None)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "citations": citations,
                                "retrieved_passages": passages
                            })
                        else:
                            err = f"Request failed ({resp.status_code}): {resp.text}"
                            st.error(err)
                            st.session_state.messages.append({"role": "assistant", "content": err})
                    except Exception as e:
                        err = f"Failed to connect: {str(e)}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})


# ----------------------------------------------------
# TAB 2: Documents
# ----------------------------------------------------
with tab_docs:
    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Index Files", type="primary"):
            if not backend_ok:
                st.error("Backend offline.")
            else:
                progress = st.progress(0)
                status_text = st.empty()
                results = []
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})...")
                    files_payload = [("files", (file.name, file.getvalue(), "application/pdf"))]
                    try:
                        resp = requests.post(f"{api_url}/api/documents/upload", files=files_payload, timeout=120)
                        if resp.status_code == 200:
                            results.extend(resp.json())
                        else:
                            results.append({"filename": file.name, "status": "error", "error": resp.text})
                    except Exception as ex:
                        results.append({"filename": file.name, "status": "error", "error": str(ex)})
                    progress.progress((i + 1) / len(uploaded_files))
                status_text.text("Ingestion completed.")
                st.dataframe(results, use_container_width=True)
                st.rerun()

    st.markdown("---")
    st.markdown("#### Indexed Document Catalog")
    if backend_ok:
        try:
            docs_resp = requests.get(f"{api_url}/api/documents", timeout=5)
            if docs_resp.status_code == 200:
                docs = docs_resp.json()
                if not docs:
                    st.caption("No documents currently indexed.")
                else:
                    for d in docs:
                        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                        with c1:
                            st.write(d['doc_name'])
                        with c2:
                            st.caption(f"Pages: {d['page_count']} | Chunks: {d['chunk_count']}")
                        with c3:
                            st.caption(d.get('created_at', ''))
                        with c4:
                            if st.button("Remove", key=f"del_{d['doc_name']}"):
                                requests.delete(f"{api_url}/api/documents/{d['doc_name']}")
                                st.rerun()
            else:
                st.error("Unable to load documents.")
        except Exception as e:
            st.error(f"Error: {e}")


# ----------------------------------------------------
# TAB 3: Inspector
# ----------------------------------------------------
with tab_inspector:
    st.markdown("#### Retrieval & Reranking Diagnostics")
    insp_query = st.text_input("Enter search query to inspect rank progression:")
    col_x, col_y = st.columns(2)
    with col_x:
        pool_k = st.slider("Candidate Pool", 5, 20, 10, key="pool_k")
    with col_y:
        out_k = st.slider("Top Reranked Passages", 1, 10, 5, key="out_k")

    if st.button("Run Diagnostic"):
        if not insp_query.strip():
            st.warning("Enter a query.")
        elif not backend_ok:
            st.error("Backend offline.")
        else:
            with st.spinner("Analyzing rank progression..."):
                try:
                    resp = requests.post(
                        f"{api_url}/api/search",
                        json={"query": insp_query, "top_k_retrieval": pool_k, "top_k_rerank": out_k},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        passages = resp.json().get("passages", [])
                        if not passages:
                            st.caption("No matching passages found.")
                        else:
                            for idx, p in enumerate(passages, start=1):
                                with st.expander(f"Rank {idx}: {p['doc_name']} (Page {p['page_number']}) — Cross-Encoder: {p.get('rerank_score')}"):
                                    ca, cb, cc, cd = st.columns(4)
                                    ca.metric("Cross-Encoder", p.get("rerank_score", "N/A"))
                                    cb.metric("RRF Score", f"{p.get('rrf_score', 0):.5f}")
                                    cc.metric("Vector Rank", f"#{p.get('vector_rank')}" if p.get('vector_rank') else "Unranked")
                                    cd.metric("BM25 Rank", f"#{p.get('bm25_rank')}" if p.get('bm25_rank') else "Unranked")
                                    st.text_area("Passage Text", p["text"], height=120, disabled=True, key=f"txt_{idx}")
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(f"Failed: {e}")


# ----------------------------------------------------
# TAB 4: Documentation
# ----------------------------------------------------
with tab_docs_page:
    st.markdown("#### System Documentation & Reference")
    st.markdown("Clear, concise architectural overview, step-by-step usage guide, and complete parameter reference.")

    st.markdown("##### 1. Architecture & Execution Flow")
    st.markdown("""
The system employs a two-stage hybrid retrieval strategy coupled with strict context grounding:

1. **Document Ingestion (`backend.ingestion`)**: PyMuPDF extracts text page-by-page. A recursive text splitter chunks each page by paragraphs and sentences while retaining exact document name and 1-based page number metadata.
2. **Dual Indexing (`backend.storage`)**:
   - **Dense Index**: `sentence-transformers/all-MiniLM-L6-v2` encodes chunks into 384-dimensional normalized vectors, stored in a persistent FAISS `IndexIDMap(IndexFlatIP)` index for exact cosine similarity.
   - **Sparse Index**: Words are tokenized and indexed using `rank-bm25` (BM25Okapi) with serialized disk caching.
   - **Metadata Store**: SQLite (`metadata.db`) maps chunk IDs to document records and page citations.
3. **Hybrid Retrieval & RRF (`backend.retrieval`)**: The user query is searched in parallel across both FAISS and BM25. Results are merged using Reciprocal Rank Fusion:
   $$RRF(d) = \\sum_{m \\in \\{Vector, BM25\\}} \\frac{1}{k + r_m(d)}$$
4. **Cross-Encoder Reranking (`backend.reranking`)**: The candidate pool is reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`, which runs full cross-attention over `(query, passage)` pairs.
5. **Grounded Generation (`backend.generation`)**: An LLM (OpenRouter community free models or Google Gemini) receives the top reranked passages alongside strict instructions to only cite retrieved facts with `[Document, Page]` and state insufficient information if facts are missing.
""")

    st.markdown("##### 2. How to Use")
    st.markdown("""
- **Step 1: Configure Credentials**: Set `OPENROUTER_API_KEY` or `GEMINI_API_KEY` in `.env` (or enter it directly in the left sidebar).
- **Step 2: Choose Free Model**: Select **OpenRouter (Free Models)** and pick `openrouter/free` (or any individual free model such as `nvidia/nemotron-3.5-lightning:free` or `liquid/lfm-3-1.7b:free`).
- **Step 3: Upload Documents**: Open the **Documents** tab, drag and drop one or more PDF files, and click **Index Files**. Text is extracted, embedded, and persisted.
- **Step 4: Query & Chat**: In the **Chat** tab, ask any question. Answers will be returned with inline `[document.pdf, Page X]` citations and an expandable evidence viewer.
- **Step 5: Diagnostics**: Use the **Inspector** tab to examine how vector similarity and BM25 scores combine into the final Cross-Encoder ranking.
""")

    st.markdown("##### 3. Configuration Parameters Reference")
    st.markdown("""
| Parameter | Location | Default | Description & Tuning Guide |
|---|---|---|---|
| `OPENROUTER_API_KEY` | `.env` / Sidebar | `None` | OpenRouter API key for free community models (https://openrouter.ai/keys). |
| `OPENROUTER_MODEL` | `.env` / Sidebar | `openrouter/free` | Default free OpenRouter model for generation. |
| `GEMINI_API_KEY` | `.env` / Sidebar | `None` | Google AI Studio API key (alternative provider). |
| `GEMINI_MODEL` | `.env` / Sidebar | `gemini-3.8-flash` | Gemini model variant (`gemini-3.8-flash`, `gemini-3.5-flash`). |
| `LLM_PROVIDER` | `.env` / Sidebar | `openrouter` | Default LLM provider (`openrouter` or `gemini`). |
| `CHUNK_SIZE` | `backend/config.py` | `500` | Target character size per chunk. Smaller sizes isolate discrete facts; larger sizes preserve surrounding context. |
| `CHUNK_OVERLAP` | `backend/config.py` | `100` | Overlap in characters between adjacent chunks to prevent cutting context across split boundaries. |
| `TOP_K_RETRIEVAL` | Sidebar / API | `10` | Size of the candidate pool retrieved from the initial FAISS + BM25 search. |
| `TOP_K_RERANK` | Sidebar / API | `5` | Number of top reranked chunks passed into the LLM context prompt after Cross-Encoder evaluation. |
| `RRF_K` | `backend/config.py` | `60` | Smoothing constant for Reciprocal Rank Fusion ($1 / (k + rank)$). Dampens the impact of extreme outliers. |
| `EMBEDDING_MODEL` | `backend/config.py` | `all-MiniLM-L6-v2` | SentenceTransformer embedding model name (384-dimensional dense vectors on CPU). |
| `RERANKER_MODEL` | `backend/config.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder model name for joint query-passage scoring on CPU. |
| `BACKEND_HOST` | `.env` / Config | `127.0.0.1` | Local IP address for the FastAPI backend server. |
| `BACKEND_PORT` | `.env` / Config | `8000` | Port for the FastAPI backend server. |
| `STORAGE_DIR` | `backend/config.py` | `data` | Directory where `faiss.index`, `metadata.db`, and `bm25_corpus.pkl` are stored. |
""")

    st.markdown("##### 4. Verified Free OpenRouter Models & Auto-Fallback")
    st.markdown("""
All of the following models are verified free of charge on OpenRouter with zero token fees:
- `openrouter/free` (**Recommended**) — OpenRouter's meta-router that dynamically load-balances across available free models, eliminating single-provider downtime.
- `nvidia/nemotron-3.5-lightning:free` — Fast, highly capable reasoning and instruction-following model.
- `liquid/lfm-3-1.7b:free` — Ultra-fast, lightweight 1.7B parameter model.
- `google/gemma-4-31b-it:free` — Google Gemma 31B instruction model for high semantic precision.
- `google/gemma-4-26b-a4b-it:free` — Google Gemma 26B instruction model.
- `inclusionai/ling-3.2-flash:free` — Low-latency flash generation.
- `baidu/ernie-5.5-flash-preview:free` — Baidu ERNIE language model.
- `dots-studio/dots-3-note-preview:free` — Concise, factual summarization.

**Built-in 429 Resilience**: If a free provider returns HTTP 429 (rate-limited) or a temporary 5xx error, the backend generator automatically fails over through the candidate chain (`openrouter/free`, `nvidia/nemotron-3.5-lightning:free`, etc.) without crashing your chat session.
""")


# ----------------------------------------------------
# Global Page Footer
# ----------------------------------------------------
st.markdown(
    '<div class="app-footer">Made by <a href="https://dheerajkumar.pages.dev/" target="_blank" rel="noopener noreferrer">Dheeraj</a></div>',
    unsafe_allow_html=True
)
