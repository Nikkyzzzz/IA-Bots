# ============================== firstpage.py — Home / Upload Router ==============================
import json
import base64
from pathlib import Path
import streamlit as st

from secondpage import render_process
from thirdpage import render_next
from fourthpage import render_fourth
from processpage import render_processpage
from fifthpage import render_fifth


# ================================== 1) App Config & Brand Controls ==================================
st.set_page_config(
    page_title="Audit Bots",
    page_icon="🤖",
    layout="wide",
    # initial_sidebar_state="collapsed",
)
st.markdown('<div class="ab-firstpage">', unsafe_allow_html=True)

LEFT_LOGO_PATH   = "logo.png"
LOGO_WIDTH       = 154
LOGO_HEIGHT      = 100

VALID_EXTS       = ("xlsx", "xls")
SUPPORTED_LABEL  = "(Supported formats: XLSX, XLS)"


# # ================================== 2) Global Styles ===================================
# st.markdown(
#     """
#     <style>
#       @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#       :root, .stApp, .block-container, body { background: #ffffff !important; color: #0f172a !important; }
#       * { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
#       .block-container { padding-top: 20px; max-width: 1280px; }

#       /* ===== Header (match secondpage.py) ===== */
#       .hdr-grid { display: grid; grid-template-columns: 268px 1fr; align-items: center; column-gap: 16px; margin-bottom: 8px; }
#       .hdr-brand { display: flex; align-items: center; justify-content: flex-end; height: 168px; position: relative; }
#       .ab-title {
#         font-weight:860; letter-spacing:.02em;
#         background: linear-gradient(90deg, #1e3a8a 0%, #60a5fa 100%);
#         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#         background-clip: text; color: transparent;
#         font-size: clamp(30px, 3.2vw, 38px); line-height: 1.05;
#       }
#       .ab-underline { height:8px; border-radius:9999px; background:#ff7636; margin-top:8px; width:320px; max-width:30vw; }
#       .ab-subtitle { margin-top:10px; font-weight:700; color:#0f172a; font-size: clamp(16px, 1.6vw, 20px); }

#       /* NEW: sub-heading under Upload Dataset — 4px smaller than previous (was 13–16px) */
#       .ab-subhead {
#         margin-top: 4px;
#         color: #334155;
#         font-weight: 600;
#         font-size: clamp(9px, 1.0vw, 12px); /* 4px smaller across the range */
#         opacity: .95;
#         display: inline-block;
#       }

#       .ab-spacer { height: 1.2em; }

#       /* Make scroll-to-header stop with the brand fully visible (works with other pages) */
#       #__ab_header__ { scroll-margin-top: 96px; }
#       @media (max-width: 640px){ #__ab_header__{ scroll-margin-top: 120px; } }

#       /* Cards / pills / uploader UI */
#       .card { padding: 1rem 1.25rem; border: 1px solid rgba(15,23,42,.06); border-radius: 14px; background: #ffffff; box-shadow: 0 2px 6px rgba(2,6,23,.06); }
#       .pill-row { display: flex; gap: .5rem; flex-wrap: wrap; margin: .25rem 0 1rem 0; }
#       .pill { border-radius: 999px; padding: .25rem .68rem; font-size: .85rem; border: 1px solid rgba(15,23,42,.12); background: #f8fafc; }
#       .pill.ok { border-color: rgba(22,163,74,.35); background: #ecfdf5; }
#       .pill.bad { border-color: rgba(220,38,38,.35); background: #fef2f2; }
#       .pill.neutral { opacity: .8; }

#       .stUploadedFile .uploadedFile .uploadedFileIcon { display: none !important; }
#       .success-file .uploadedFileName::before { content: "✅ "; color: #16a34a; font-size: 18px; }
#       .failed-file  .uploadedFileName::before { content: "❌ "; color: #dc2626; font-size: 18px; }

#       button[kind="primary"] { border-radius: 12px !important; font-weight: 700 !important; letter-spacing: .01em; }
#       .stFileUploader label { font-weight: 700 !important; font-size: 14px !important; }
#       .stFileUploader div[data-testid="stFileUploaderDropzone"] { border-radius: 12px; }

#       .sticky-continue {
#         position: sticky; bottom: 0; z-index: 50;
#         background: linear-gradient(180deg, rgba(255,255,255,0) 0%, #fff 22%);
#         padding-top: .75rem;
#       }
#       .sticky-continue button[kind="primary"] div p::after { content: "⏭"; margin-left: .4rem; vertical-align: -2px; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )



# ================================== 3) Session Bootstrap & Page Router ============================
if "page" not in st.session_state:
    st.session_state.page = "home"

def _cache_file_if_needed(key: str) -> None:
    s = st.session_state
    f = s.get(key)
    if f is not None:
        try:
            s[f"{key}_bytes"] = f.getvalue()
            s[f"{key}_name"]  = getattr(f, "name", "")
        except Exception:
            pass

def _clear_cache_if_removed(key: str) -> None:
    s = st.session_state
    if s.get(key, None) is None:
        s.pop(f"{key}_bytes", None)
        s.pop(f"{key}_name",  None)

def _bootstrap_bytes() -> None:
    for key in ("u_master", "u_p2p", "u_o2c", "u_h2r"):
        _clear_cache_if_removed(key)
        _cache_file_if_needed(key)

def _set_dynamic_title(page_key: str) -> None:
    title_map = {
        "home":        "Audit Bots • Home",
        "process":     "Audit Bots • Process",
        "fourth":      "Audit Bots • Review",
        "processpage": "Audit Bots • Processing",
        "fifth":       "Audit Bots • Results",
        "next":        "Audit Bots • Next",
    }
    title = title_map.get(page_key, "Audit Bots")
    st.markdown(f"<script>document.title = {json.dumps(title)};</script>", unsafe_allow_html=True)


# ================================== 4) Helpers ====================================================
def _is_valid_name(name: str | None) -> bool:
    if not name:
        return False
    n = name.lower()
    return any(n.endswith(f".{ext}") for ext in VALID_EXTS)

def _status_from_cache_or_file(uploader_key: str):
    s = st.session_state
    live_file = s.get(uploader_key, None)
    if live_file is not None:
        name = getattr(live_file, "name", "") or ""
        lower = name.lower()
        return any(lower.endswith(f".{ext}") for ext in ["xlsx", "xls"])
    s.pop(f"{uploader_key}_bytes", None)
    s.pop(f"{uploader_key}_name",  None)
    return None

def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _pill(label: str, state):
    cls = "neutral"
    if state is True:  cls = "ok"
    elif state is False: cls = "bad"
    return f'<span class="pill {cls}">{label}</span>'

# === Header that matches secondpage.py; “Supported formats” as sub-heading under Upload Dataset ===
def _render_template_header():
    logo_src = _data_uri(LEFT_LOGO_PATH)
    # two non-breaking spaces appended after the label for clean spacing
    label_with_spaces = f"{SUPPORTED_LABEL}&nbsp;&nbsp;"
    st.markdown(
        f"""
        <div id="__ab_header__">
          <div class="hdr-grid">
            <div class="hdr-logo">
              {'<img src="' + logo_src + f'" width="{LOGO_WIDTH}" height="{LOGO_HEIGHT}" alt="Logo" style="display:block;" />' if logo_src else ''}
            </div>
            <div class="hdr-brand" style="text-align:right;">
              <div>
                <div class="ab-title">Audit&nbsp;Bots</div>
                <div class="ab-underline" style="margin-left:auto;"></div>
                <div class="ab-subtitle">Upload Dataset</div>
                <div class="ab-subhead">{label_with_spaces}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================== 5) Home Page (upload router) =================================
def render_home():
    _render_template_header()  # ← same header look as secondpage.py, with sub-heading

    _bootstrap_bytes()

    master_ok = _status_from_cache_or_file("u_master")
    p2p_ok    = _status_from_cache_or_file("u_p2p")
    o2c_ok    = _status_from_cache_or_file("u_o2c")
    h2r_ok    = _status_from_cache_or_file("u_h2r")

    hide_master = any(x is True for x in (p2p_ok, o2c_ok, h2r_ok))
    hide_others = (master_ok is True)

    p2p_disp_ok = (True if master_ok else p2p_ok)
    o2c_disp_ok = (True if master_ok else o2c_ok)
    h2r_disp_ok = (True if master_ok else h2r_ok)

    st.markdown(
        f"""
        <div class="pill-row">
          {_pill("P2P Master", p2p_disp_ok)}
          {_pill("O2C Master", o2c_disp_ok)}
          {_pill("H2R Master", h2r_disp_ok)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Master uploader shown if no other file present
        if not hide_master:
            st.file_uploader("Master_Data file", key="u_master", type=list(VALID_EXTS))
        if not hide_others:
            st.file_uploader("P2P_Master file", key="u_p2p", type=list(VALID_EXTS))
            st.file_uploader("O2C_Master file", key="u_o2c", type=list(VALID_EXTS))
            st.file_uploader("H2R_Master file", key="u_h2r", type=list(VALID_EXTS))
        st.markdown("</div>", unsafe_allow_html=True)

    can_proceed = any(x is True for x in (master_ok, p2p_ok, o2c_ok, h2r_ok))
    label = "Process"

    st.markdown('<div class="sticky-continue">', unsafe_allow_html=True)
    if st.button(label, type="primary", disabled=not can_proceed, use_container_width=True):
        st.session_state.page = "process"
        st.session_state["_force_scroll_top"] = True
        st.rerun()


# ============================== ROUTER — FIXED ==============================

def run_router():
    s = st.session_state


    # ================= REQUIRED INTERNAL INITIALIZATIONS =================
    # These are needed by secondpage, thirdpage, fourthpage, selectionpage, processpage
    if "file_bytes_map" not in s:
        s.file_bytes_map = {}

    if "uploaded_files" not in s:
        s.uploaded_files = {}

    if "sheet_mapping_pairs" not in s:
        s.sheet_mapping_pairs = {}

    if "column_mapping_pairs" not in s:
        s.column_mapping_pairs = {}

    if "field_map" not in s:
        s.field_map = {}

    if "selected_bots" not in s:
        s.selected_bots = []

    if "processed_results" not in s:
        s.processed_results = {}
    # ======================================================================

    # default state
    if "page" not in s:
        s.page = "home"

    page = s.page

    if page == "home":
        _set_dynamic_title("home")
        render_home()

        # === AI Toggle (restored exactly like firstpagea.py) ===
        st.subheader("Optional Features")
        st.session_state.use_ai = st.checkbox("Enable AI Features", value=False)

    elif page == "process":
        _set_dynamic_title("process")
        render_process()   # from secondpage.py

    elif page == "next":
        _set_dynamic_title("next")
        render_next()      # from thirdpage.py

    elif page == "fourth":
        _set_dynamic_title("fourth")
        render_fourth()    # from fourthpage.py

    elif page == "selection":
        _set_dynamic_title("selection")
        from selectionpage import render_selectionpage
        render_selectionpage()

    elif page == "processpage":
        _set_dynamic_title("processpage")
        render_processpage()   # from processpage.py

    elif page == "fifth":
        _set_dynamic_title("fifth")
        render_fifth()     # from fifthpage.py

    else:
        # fallback
        _set_dynamic_title("home")
        render_home()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    run_router()
