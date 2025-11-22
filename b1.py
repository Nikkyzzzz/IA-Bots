
# ============================== b1.py — Banking Home ==============================
import streamlit as st
import streamlit.components.v1 as components
import base64
from pathlib import Path
from io import BytesIO

LEFT_LOGO_PATH = "logo.png"

def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else ("image/jpeg" if suffix in (".jpg",".jpeg") else "image/png")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _scroll_to_brand_if_needed(s):
    if s.pop("_force_scroll_top", False):
        components.html(
            """
            <script>
            (function(){
              function go(){
                try{
                  const doc = window.parent && window.parent.document ? window.parent.document : document;
                  const el = doc.getElementById("__ab_header__");
                  if(el && el.scrollIntoView){
                    el.scrollIntoView({block:"start", inline:"nearest"});
                  } else {
                    (window.parent||window).scrollTo(0,0);
                  }
                }catch(e){ (window.parent||window).scrollTo(0,0); }
              }
              go(); setTimeout(go, 50); setTimeout(go, 150); setTimeout(go, 300); setTimeout(go, 600);
            })();
            </script>
            """,
            height=0,
        )

def render_bank1():
    s = st.session_state
    st.set_page_config(
        page_title="Audit Bots • Banking Home",
        page_icon="🏦",
        layout="wide",
    )

    # Styles + header (same as zeropage)
    st.markdown(
        """
        <style>
          .ab-wrap * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
          .ab-title { font-weight:860; letter-spacing:.02em; background:linear-gradient(90deg,#1e3a8a 0%,#60a5fa 100%);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; color:transparent;
                      font-size:clamp(30px,3.2vw,38px); line-height:1.05; }
          .ab-underline { height:8px; border-radius:9999px; background:#ff7636; margin-top:8px; width:320px; max-width:30vw; }
          .ab-subtitle { margin-top:10px; font-weight:700; color:#0f172a; font-size:clamp(16px,1.6vw,20px); }
          .ab-spacer { height: 1.2em; }
          .hdr-grid { display:grid; grid-template-columns:268px 1fr; align-items:center; column-gap:16px; margin-bottom:8px; }
          .hdr-brand { display:flex; justify-content:flex-end; align-items:center; height:168px; position:relative; }
          #__ab_header__ { scroll-margin-top: 96px; }
          @media (max-width: 640px){ #__ab_header__{ scroll-margin-top: 120px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-wrap">', unsafe_allow_html=True)

    logo_src = _data_uri(LEFT_LOGO_PATH)
    st.markdown(
        f"""
        <div id="__ab_header__">
          <div class="hdr-grid">
            <div class="hdr-logo" style="width:268px;">
              {'<img src="' + logo_src + '" width="154" height="100" alt="Logo" style="display:block;" />' if logo_src else ''}
            </div>
            <div class="hdr-brand" style="text-align:right;">
              <div>
                <div class="ab-title">Audit&nbsp;Bots</div>
                <div class="ab-underline" style="margin-left:auto;"></div>
                <div class="ab-subtitle">File Upload</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    # --- File Upload UI ---
    st.subheader("Upload Required Files")

    # Create tabs for different file types
    tab1, tab2 = st.tabs(["📊 Banking Data Files", "📄 PDF Data Extraction"])
    
    with tab1:
        st.markdown('<div style="font-size:1.2rem;font-weight:700;margin-bottom:0.2em;">Loan Dump file</div>', unsafe_allow_html=True)
        uploaded_ccis = st.file_uploader("", type=["xlsx", "xls"], key="u_ccis")

        st.markdown('<div style="font-size:1.2rem;font-weight:700;margin-bottom:0.2em;">Blacklisted PIN code file</div>', unsafe_allow_html=True)
        uploaded_blacklist = st.file_uploader("", type=["xlsx", "xls"], key="u_blacklist")

        st.markdown('<div style="font-size:1.2rem;font-weight:700;margin-bottom:0.2em;">Loan Book Base Period</div>', unsafe_allow_html=True)
        uploaded_loan_mar = st.file_uploader("", type=["xlsx", "xls"], key="u_loan_mar")

        st.markdown('<div style="font-size:1.2rem;font-weight:700;margin-bottom:0.2em;">Loan Book Comparison Period</div>', unsafe_allow_html=True)
        uploaded_loan_jun = st.file_uploader("", type=["xlsx", "xls"], key="u_loan_jun")
    
    with tab2:
        st.markdown("### 📄 Upload PDF Files for Data Extraction")
        st.markdown("Upload PDF files containing project and loan information for automated data extraction.")
        
        # PDF file uploader
        uploaded_pdfs = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            key="u_pdfs",
            help="Upload PDF files containing loan and project details"
        )
        
        if uploaded_pdfs:
            st.success(f"✅ {len(uploaded_pdfs)} PDF file(s) uploaded successfully!")
            
            # Show file details
            st.markdown("**Uploaded Files:**")
            for i, file in enumerate(uploaded_pdfs, 1):
                st.markdown(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
            
            # Cache PDF files for processing
            st.session_state["u_pdfs_cached"] = uploaded_pdfs
            st.session_state["pdf_upload_ready"] = True
            
            # Start background processing immediately
            if "pdf_processing_started" not in st.session_state:
                st.session_state["pdf_processing_started"] = True
                st.session_state["pdf_processing_status"] = "processing"
                st.info("🔄 PDF processing has started in the background. Check the Data Extraction tab in results to see progress.")
        else:
            st.session_state.pop("pdf_upload_ready", None)
            st.session_state.pop("pdf_processing_started", None)

    # --- Proceed Button Logic ---
    proceed_condition = (uploaded_ccis is not None) or (uploaded_loan_mar and uploaded_loan_jun) or st.session_state.get("pdf_upload_ready", False)

    if proceed_condition:
        if st.button("Proceed ➜", type="primary", use_container_width=True):
            # Save file info in session state
            if uploaded_ccis:
                st.session_state["u_ccis_name"] = uploaded_ccis.name
                try:
                    st.session_state["u_ccis_bytes"] = uploaded_ccis.getvalue()
                except Exception:
                    st.session_state["u_ccis_bytes"] = None

            if uploaded_blacklist:
                st.session_state["u_blacklist_name"] = uploaded_blacklist.name
                try:
                    st.session_state["u_blacklist_bytes"] = uploaded_blacklist.getvalue()
                except Exception:
                    st.session_state["u_blacklist_bytes"] = None

            if uploaded_loan_mar:
                st.session_state["u_loan_mar_name"] = uploaded_loan_mar.name
                try:
                    st.session_state["u_loan_mar_bytes"] = uploaded_loan_mar.getvalue()
                except Exception:
                    st.session_state["u_loan_mar_bytes"] = None

            if uploaded_loan_jun:
                st.session_state["u_loan_jun_name"] = uploaded_loan_jun.name
                try:
                    st.session_state["u_loan_jun_bytes"] = uploaded_loan_jun.getvalue()
                except Exception:
                    st.session_state["u_loan_jun_bytes"] = None

            # Move to next page
            st.session_state.page = "bankprocess"   # triggers b2.py
            st.rerun()
