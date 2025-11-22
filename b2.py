# ============================== b2.py — Banking Sheet Mapping ==============================
import hashlib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
import re
import base64
from pathlib import Path

LEFT_LOGO_PATH = "logo.png"

# ---------------------- Config: Required sheet names per workbook ----------------------
# Each uploaded workbook maps exactly one sheet with the same display name.
REQUIRED_SHEETS = {
    "Banking": ["Loan Dump"],
    "Blacklisted PIN CODE": ["Blacklisted PIN CODE"],
    "Loan Book (31.03.2025)": ["Loan Book (31.03.2025)"],
    "Loan Book (30.06.2025)": ["Loan Book (30.06.2025)"],
}

# ---------------------- Small helpers ----------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())

def _is_excel_name(name: str) -> bool:
    return name.lower().endswith((".xlsx", ".xls"))

def _get_filelike_and_label_for_cat(cat: str, s):
    """
    Return (label, filelike, name) for the given category based on session keys set in b1.py.
    Supports both *_bytes and original upload handles as fallback.
    """
    if cat == "Banking":
        if s.get("u_ccis_bytes"):
            return ("Loan Dump file", BytesIO(s["u_ccis_bytes"]), s.get("u_ccis_name", ""))
        if s.get("u_ccis"):
            return ("Loan Dump file", s["u_ccis"], getattr(s["u_ccis"], "name", ""))
    elif cat == "Blacklisted PIN CODE":
        if s.get("u_blacklist_bytes"):
            return ("Blacklisted PIN code file", BytesIO(s["u_blacklist_bytes"]), s.get("u_blacklist_name", ""))
        if s.get("u_blacklist"):
            return ("Blacklisted PIN code file", s["u_blacklist"], getattr(s["u_blacklist"], "name", ""))
    elif cat == "Loan Book (31.03.2025)":
        if s.get("u_loan_mar_bytes"):
            return ("Loan Book Base Period", BytesIO(s["u_loan_mar_bytes"]), s.get("u_loan_mar_name", ""))
        if s.get("u_loan_mar"):
            return ("Loan Book Base Period", s["u_loan_mar"], getattr(s["u_loan_mar"], "name", ""))
    elif cat == "Loan Book (30.06.2025)":
        if s.get("u_loan_jun_bytes"):
            return ("Loan Book Comparison Period", BytesIO(s["u_loan_jun_bytes"]), s.get("u_loan_jun_name", ""))
        if s.get("u_loan_jun"):
            return ("Loan Book Comparison Period", s["u_loan_jun"], getattr(s["u_loan_jun"], "name", ""))
    return ("", None, "")

def _auto_map(required, sheets):
    idx = { _norm(s): s for s in sheets }
    return { r: idx.get(_norm(r), "") for r in required }

def _get_friendly_display_name(cat: str, need: str) -> str:
    """Get user-friendly display name for category and requirement"""
    if cat == "Banking" and need == "Loan Dump":
        return "Loan Dump file"
    elif cat == "Blacklisted PIN CODE" and need == "Blacklisted PIN CODE":
        return "Blacklisted PIN code file"
    elif cat == "Loan Book (31.03.2025)" and need == "Loan Book (31.03.2025)":
        return "Loan Book Base Period"
    elif cat == "Loan Book (30.06.2025)" and need == "Loan Book (30.06.2025)":
        return "Loan Book Comparison Period"
    else:
        return f"{cat} • {need}"  # fallback

def _make_on_change(cat: str, need: str, required):
    # Note: kept un-annotated 'required' to avoid edge-version typing quirks.
    key_current = f"map_{cat}_{_norm(need)}"
    def _cb():
        s = st.session_state
        if s.get("_mapping_bootstrap_active", False):
            return
        new_val = s.get(key_current, "")
        if "cat_map" not in s:
            s.cat_map = {}
        prev_map = dict(s.cat_map.get(cat, {}))
        prev_current_val = prev_map.get(need, "")
        # De-duplicate selections across required items
        if new_val and new_val != prev_current_val:
            for other in required:
                if other == need:
                    continue
                other_key = f"map_{cat}_{_norm(other)}"
                if s.get(other_key, "") == new_val:
                    s[other_key] = ""
        s.cat_map[cat] = { r: s.get(f"map_{cat}_{_norm(r)}", "") for r in required }
    return _cb

def _file_cache_key(src, name: str) -> str:
    try:
        src.seek(0)
        data = src.read()
        src.seek(0)
        h = hashlib.md5(data).hexdigest()
        return f"{name}:{h}"
    except Exception:
        # If hashing fails for any reason, return a stable-ish key
        return f"{name}:nohash"

@st.cache_data(show_spinner=False)
def _sheet_names_cached(cache_key: str, file_bytes: bytes):
    with BytesIO(file_bytes) as f:
        xl = pd.ExcelFile(f)
        return xl.sheet_names

def _sheet_names(filelike, name_hint: str = ""):
    try:
        try:
            filelike.seek(0)
        except Exception:
            pass
        b = filelike.read()
        try:
            filelike.seek(0)
        except Exception:
            pass
        key = _file_cache_key(BytesIO(b), name_hint or "workbook")
        return _sheet_names_cached(key, b)
    except Exception:
        return []

def _bootstrap_bytes(s):
    """If only upload handles exist in session, populate *_bytes + *_name to normalize downstream logic."""
    # CCIS
    f = s.get("u_ccis")
    if f is not None and not s.get("u_ccis_bytes"):
        try:
            s["u_ccis_bytes"] = f.getvalue()
            s["u_ccis_name"]  = getattr(f, "name", "")
        except Exception:
            try:
                s["u_ccis_bytes"] = f.read()
                s["u_ccis_name"]  = getattr(f, "name", "")
                f.seek(0)
            except Exception:
                pass
    # Blacklist
    f = s.get("u_blacklist")
    if f is not None and not s.get("u_blacklist_bytes"):
        try:
            s["u_blacklist_bytes"] = f.getvalue()
            s["u_blacklist_name"]  = getattr(f, "name", "")
        except Exception:
            try:
                s["u_blacklist_bytes"] = f.read()
                s["u_blacklist_name"]  = getattr(f, "name", "")
                f.seek(0)
            except Exception:
                pass
    # Loan Book Mar
    f = s.get("u_loan_mar")
    if f is not None and not s.get("u_loan_mar_bytes"):
        try:
            s["u_loan_mar_bytes"] = f.getvalue()
            s["u_loan_mar_name"]  = getattr(f, "name", "")
        except Exception:
            try:
                s["u_loan_mar_bytes"] = f.read()
                s["u_loan_mar_name"]  = getattr(f, "name", "")
                f.seek(0)
            except Exception:
                pass
    # Loan Book Jun
    f = s.get("u_loan_jun")
    if f is not None and not s.get("u_loan_jun_bytes"):
        try:
            s["u_loan_jun_bytes"] = f.getvalue()
            s["u_loan_jun_name"]  = getattr(f, "name", "")
        except Exception:
            try:
                s["u_loan_jun_bytes"] = f.read()
                s["u_loan_jun_name"]  = getattr(f, "name", "")
                f.seek(0)
            except Exception:
                pass

def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else ("image/jpeg" if suffix in (".jpg",".jpeg") else "image/png")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _scroll_to_brand_if_needed(s):
    # Always scroll to top on page load, plus handle explicit scroll requests
    force_scroll = s.pop("_force_scroll_top", False)
    
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

def _prepare_scope_defaults(scope, s):
    s["_mapping_bootstrap_active"] = True
    try:
        for cat in scope:
            label, src, name = _get_filelike_and_label_for_cat(cat, s)
            if not src: 
                continue
            if name and not _is_excel_name(name):
                continue
            sheets = _sheet_names(src, name)
            if not sheets:
                continue
            required = REQUIRED_SHEETS[cat]
            base = s.cat_map.get(cat, {}) or _auto_map(required, sheets)
            for need in required:
                k = f"map_{cat}_{_norm(need)}"
                v = base.get(need, "")
                if v not in [""] + list(sheets):
                    v = ""
                st.session_state[k] = v
            s.cat_map[cat] = { r: st.session_state.get(f"map_{cat}_{_norm(r)}", "") for r in required }
    finally:
        s["_mapping_bootstrap_active"] = False

def _render_cat_mapping(cat: str, s) -> bool:
    label, src, name = _get_filelike_and_label_for_cat(cat, s)

    required = REQUIRED_SHEETS[cat]
    if not src:
        for need in required:
            friendly_name = _get_friendly_display_name(cat, need)
            st.subheader(friendly_name)
            st.info("No workbook uploaded for this section.")
        s.cat_ready[cat] = False
        return False
    if name and not _is_excel_name(name):
        for need in required:
            friendly_name = _get_friendly_display_name(cat, need)
            display_name = f"{friendly_name}: {name}" if name else friendly_name
            st.subheader(display_name)
            st.error("Please upload an Excel workbook (.xlsx / .xls).")
        s.cat_ready[cat] = False
        return False

    sheets = _sheet_names(src, name)
    if len(sheets) < len(required):
        for need in required:
            friendly_name = _get_friendly_display_name(cat, need)
            display_name = f"{friendly_name}: {name}" if name else friendly_name
            st.subheader(display_name)
            st.error("Workbook has fewer sheets than required.")
        s.cat_ready[cat] = False
        return False

    snap = {}
    for need in required:
        friendly_name = _get_friendly_display_name(cat, need)
        # Add filename to the display
        display_name = f"{friendly_name}: {name}" if name else friendly_name
        st.subheader(display_name)   # Use friendly display name with filename

        key_name = f"map_{cat}_{_norm(need)}"
        options = [""] + list(sheets)

        # Auto-select if only one sheet is available
        if len(sheets) == 1:
            st.session_state[key_name] = sheets[0]

        mapped_val = st.session_state.get(key_name, "")
        st.selectbox(
            f"Select sheet for {need}",
            options=options,
            key=key_name,
            on_change=_make_on_change(cat, need, required)
        )
        snap[need] = st.session_state.get(key_name, "")

        # Show caption with friendly name once mapped
        if snap[need]:
            display_name = f"{friendly_name}: {name}" if name else friendly_name
            st.caption(f"{display_name} → {snap[need]}")

    s.cat_map[cat] = snap

    if not all(bool(v) for v in snap.values()):
        s.cat_ready[cat] = False
        st.info("Map all required sheets to continue for this file.")
        return False

    s.cat_ready[cat] = True
    st.success("All required sheets mapped.")
    return True




# ============================== Main Renderer ==============================
def render_bank_process():
    s = st.session_state
    
    # Ensure page starts at top
    st.components.v1.html(
        """
        <script>
        // Force scroll to top immediately on page load
        window.parent.scrollTo(0, 0);
        setTimeout(() => window.parent.scrollTo(0, 0), 100);
        </script>
        """,
        height=0
    )

    # ---------- Header / styles ----------
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
                <div class="ab-subtitle">Banking Sheet Mapping</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    if "cat_map" not in s:   s.cat_map = {}
    if "cat_ready" not in s: s.cat_ready = {}

    # Normalize session to *_bytes + *_name if needed
    _bootstrap_bytes(s)

    # ---------- Determine which mapping sections to show (based on uploads in b1.py) ----------
    ccis_present = bool(s.get("u_ccis_bytes") or s.get("u_ccis"))
    bl_present   = bool(s.get("u_blacklist_bytes") or s.get("u_blacklist"))
    mar_present  = bool(s.get("u_loan_mar_bytes") or s.get("u_loan_mar"))
    jun_present  = bool(s.get("u_loan_jun_bytes") or s.get("u_loan_jun"))

    scope = []
    if ccis_present: scope.append("Banking")
    if mar_present:  scope.append("Loan Book (31.03.2025)")
    if jun_present:  scope.append("Loan Book (30.06.2025)")
    if bl_present:   scope.append("Blacklisted PIN CODE")  # optional

    if not scope:
        st.warning("No files detected from the previous step. Please upload in the Banking Home page and then proceed.")
        st.stop()

    # Bootstrap default mappings for the detected scope
    if not s.get("_mapping_prepared", False):
        with st.spinner("Preparing mapping…"):
            _prepare_scope_defaults(scope, s)
        s["_mapping_prepared"] = True
        st.rerun()

    # ---------- Render mapping cards ----------
    ready_flags = {}
    for i, c in enumerate(scope):
        if i > 0: st.markdown("---")
        ready_flags[c] = _render_cat_mapping(c, s)

    # ---------- Enable proceed: CCIS OR (Mar & Jun). Blacklist is optional ----------
    ccis_ready = ready_flags.get("Banking", False) if ccis_present else False
    mar_ready  = ready_flags.get("Loan Book (31.03.2025)", False) if mar_present else False
    jun_ready  = ready_flags.get("Loan Book (30.06.2025)", False) if jun_present else False
    proceed_enabled = bool(ccis_ready or (mar_ready and jun_ready))

    st.markdown("---")
    proceed = st.button("Proceed ➜", type="primary", disabled=not proceed_enabled, use_container_width=True)
    if proceed:
        # Save only for the scope shown
        s.sheet_mapping_list  = { c: [ s.cat_map.get(c, {}).get(r, "") for r in REQUIRED_SHEETS[c] ] for c in scope }
        s.sheet_mapping_pairs = { c: { r: s.cat_map.get(c, {}).get(r, "") for r in REQUIRED_SHEETS[c] } for c in scope }
        s.sheet_sources       = { c: _get_filelike_and_label_for_cat(c, s)[0] for c in scope }

        # Optional: store which flow is chosen for downstream logic in b3+
        s["data_mode"] = "ccis" if ccis_ready else ("loan_book_dual" if (mar_ready and jun_ready) else "unknown")

        s.sheet_mapping_saved = True
        s["_force_scroll_top"] = True
        s.page = "bank3"   # redirect to b3.py
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
