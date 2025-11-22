# ============================== secondpage.py — Sheet Mapping ==============================
import hashlib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
import re
import base64
from pathlib import Path

LEFT_LOGO_PATH = "logo.png"

# ------------------------------------------------------------------------------------------
# UPDATED REQUIRED SHEETS (P2P now includes "Authority Matrix")
# ------------------------------------------------------------------------------------------
if "REQUIRED_SHEETS" not in globals():
    REQUIRED_SHEETS = {
        "P2P": [
            "P2P Sample",
            "Vendor Master",
            "Employee Master",
            # "Authority Matrix",     # 🔵 NEW ADDITION FOR YOUR NEW 5 BOTS
        ],
        "O2C": ["O2C Sample", "Customer Master"],
        "H2R": ["Employee Master", "Attendance Register"],
    }

if "ALL_CATEGORIES_LABEL" not in globals():
    ALL_CATEGORIES_LABEL = "All Categories"

# ---------- helper: normalize --------------------------------------------------------------
if "_norm" not in globals():
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s.strip().lower())

# ---------- helper: excel filename ---------------------------------------------------------
if "_is_excel_name" not in globals():
    def _is_excel_name(name: str) -> bool:
        return name.lower().endswith((".xlsx", ".xls"))

# ------------------------------------------------------------------------------------------
# FILE SOURCE RESOLUTION (same logic but supports Authority Matrix naturally)
# ------------------------------------------------------------------------------------------
if "_get_filelike_and_label_for_cat" not in globals():
    def _get_filelike_and_label_for_cat(cat: str, s):
        if cat == "P2P":
            if s.get("u_p2p_bytes"):
                return ("P2P_Master file", BytesIO(s["u_p2p_bytes"]), s.get("u_p2p_name", ""))
            if s.get("u_p2p"):
                return ("P2P_Master file", s["u_p2p"], getattr(s["u_p2p"], "name", ""))
        if cat == "O2C":
            if s.get("u_o2c_bytes"):
                return ("O2C_Master file", BytesIO(s["u_o2c_bytes"]), s.get("u_o2c_name", ""))
            if s.get("u_o2c"):
                return ("O2C_Master file", s["u_o2c"], getattr(s["u_o2c"], "name", ""))
        if cat == "H2R":
            if s.get("u_h2r_bytes"):
                return ("H2R_Master file", BytesIO(s["u_h2r_bytes"]), s.get("u_h2r_name", ""))
            if s.get("u_h2r"):
                return ("H2R_Master file", s["u_h2r"], getattr(s["u_h2r"], "name", ""))

        # fallback → Master file
        if s.get("u_master_bytes"):
            return ("Master_Data file", BytesIO(s["u_master_bytes"]), s.get("u_master_name", ""))
        if s.get("u_master"):
            return ("Master_Data file", s["u_master"], getattr(s["u_master"], "name", ""))
        return ("", None, "")

# ------------------------------------------------------------------------------------------
# AVAILABLE CATEGORIES LOGIC
# ------------------------------------------------------------------------------------------
if "_categories_available" not in globals():
    def _categories_available(s):
        cats = set()
        if s.get("u_master") or s.get("u_master_bytes"):
            cats.update(["P2P", "O2C", "H2R"])
        if s.get("u_p2p") or s.get("u_p2p_bytes"):
            cats.add("P2P")
        if s.get("u_o2c") or s.get("u_o2c_bytes"):
            cats.add("O2C")
        if s.get("u_h2r") or s.get("u_h2r_bytes"):
            cats.add("H2R")

        order = {"P2P": 0, "O2C": 1, "H2R": 2}
        return sorted(list(cats), key=lambda x: order.get(x, 99))

# ---------- simple auto mapper -------------------------------------------------------------
if "_auto_map" not in globals():
    def _auto_map(required, sheets):
        idx = {_norm(s): s for s in sheets}
        return {r: idx.get(_norm(r), "") for r in required}

# ---------- on_change generator -------------------------------------------------------------
if "_make_on_change" not in globals():
    def _make_on_change(cat: str, need: str, required: list[str]):
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
            if new_val and new_val != prev_current_val:
                for other in required:
                    if other == need:
                        continue
                    other_key = f"map_{cat}_{_norm(other)}"
                    if s.get(other_key, "") == new_val:
                        s[other_key] = ""
            s.cat_map[cat] = {r: s.get(f"map_{cat}_{_norm(r)}", "") for r in required}

        return _cb

# ---------- file content cache -------------------------------------------------------------
def _file_cache_key(src, name: str) -> str:
    try:
        src.seek(0)
        data = src.read()
        src.seek(0)
        h = hashlib.md5(data).hexdigest()
        return f"{name}:{h}"
    except Exception:
        return f"{name}:nohash"

@st.cache_data(show_spinner=False)
def _sheet_names_cached(cache_key: str, file_bytes: bytes):
    with BytesIO(file_bytes) as f:
        xl = pd.ExcelFile(f)
        return xl.sheet_names

def _sheet_names(filelike, name_hint=""):
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

# ---------- convert uploaded Streamlit file to bytes ----------------------------------------
def _bootstrap_bytes(s):
    for key in ("u_master", "u_p2p", "u_o2c", "u_h2r"):
        f = s.get(key)
        bkey = f"{key}_bytes"
        nkey = f"{key}_name"
        if f is not None:
            try:
                s[bkey] = f.getvalue()
                s[nkey] = getattr(f, "name", "")
            except Exception:
                try:
                    s[bkey] = f.read()
                    s[nkey] = getattr(f, "name", "")
                    f.seek(0)
                except Exception:
                    pass

# ---------- header logo helper --------------------------------------------------------------
def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    else:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ---------- scroll to header ---------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------
# INITIALIZE DEFAULT MAPPINGS FOR THE SELECTED CATEGORIES
# ------------------------------------------------------------------------------------------
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

            s.cat_map[cat] = {r: st.session_state.get(f"map_{cat}_{_norm(r)}", "") for r in required}

    finally:
        s["_mapping_bootstrap_active"] = False

# ------------------------------------------------------------------------------------------
# RENDER CATEGORY-SPECIFIC SHEET MAPPING UI
# ------------------------------------------------------------------------------------------
def _render_cat_mapping(cat: str, s) -> bool:
    label, src, name = _get_filelike_and_label_for_cat(cat, s)

    st.subheader(f"{cat} • Source: {label or '—'}{f' ({name})' if name else ''}")

    if not src:
        s.cat_ready[cat] = False
        st.warning("No workbook available for this category.")
        return False

    if name and not _is_excel_name(name):
        s.cat_ready[cat] = False
        st.error("Please upload an Excel workbook (.xlsx / .xls).")
        return False

    sheets = _sheet_names(src, name)
    required = REQUIRED_SHEETS[cat]

    if len(sheets) < len(required):
        s.cat_ready[cat] = False
        st.error("Workbook has fewer sheets than required for this category.")
        return False

    cols = st.columns(2) if len(required) > 1 else [st]
    for i, need in enumerate(required):
        with cols[i % len(cols)]:
            key_name = f"map_{cat}_{_norm(need)}"
            st.selectbox(
                f"Map “{need}”",
                options=[""] + list(sheets),
                key=key_name,
                on_change=_make_on_change(cat, need, required),
            )

    snap = {r: st.session_state.get(f"map_{cat}_{_norm(r)}", "") for r in required}
    s.cat_map[cat] = snap

    if not all(bool(v) for v in snap.values()):
        s.cat_ready[cat] = False
        st.info("Map all required sheets to continue.")
        return False

    s.cat_ready[cat] = True
    st.success("All required sheets mapped.")
    return True

# ==========================================================================================
# MAIN RENDER FUNCTION FOR secondpage.py
# ==========================================================================================
def render_process():
    s = st.session_state

    # -------------- HEADER (matches your other pages) -----------------
    st.markdown(
        """
        <style>
          .ab-wrap * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
          .ab-title { font-weight:860; letter-spacing:.02em;
                      background:linear-gradient(90deg,#1e3a8a 0%,#60a5fa 100%);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                      background-clip:text; color:transparent;
                      font-size:clamp(30px,3.2vw,38px); line-height:1.05; }
          .ab-underline { height:8px; border-radius:9999px; background:#ff7636;
                          margin-top:8px; width:320px; max-width:30vw; }
          .ab-subtitle { margin-top:10px; font-weight:700; color:#0f172a;
                         font-size:clamp(16px,1.6vw,20px); }
          .ab-spacer { height: 1.2em; }
          .hdr-grid { display:grid; grid-template-columns:268px 1fr;
                      align-items:center; column-gap:16px; margin-bottom:8px; }
          .hdr-brand { display:flex; justify-content:flex-end; align-items:center;
                       height:168px; position:relative; }
          #__ab_header__ { scroll-margin-top: 96px; }
          @media (max-width: 640px){
              #__ab_header__{ scroll-margin-top: 120px; }
          }
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
              {'<img src="'+logo_src+'" width="154" height="100" alt="Logo" style="display:block;" />' if logo_src else ''}
            </div>
            <div class="hdr-brand" style="text-align:right;">
              <div>
                <div class="ab-title">Audit&nbsp;Bots</div>
                <div class="ab-underline" style="margin-left:auto;"></div>
                <div class="ab-subtitle">Sheet Mapping</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)

    _scroll_to_brand_if_needed(s)

    # -------------------- State initialization ------------------------
    if "cat_map" not in s:
        s.cat_map = {}
    if "cat_ready" not in s:
        s.cat_ready = {}

    _bootstrap_bytes(s)

    cats = _categories_available(s)
    if not cats:
        st.info("Upload at least one workbook on the first page to proceed.")
        if st.button("⟵ Back"):
            s["_force_scroll_top"] = True
            s.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # -------------------- Sidebar Category Selection -------------------
    with st.sidebar:
        st.header("Categories")
        all_mode = st.toggle(ALL_CATEGORIES_LABEL, value=True, key="all_categories_toggle")
        selected_categories = (
            cats if all_mode else st.multiselect("Choose categories", options=cats, default=cats)
        )

    scope = selected_categories if not s.all_categories_toggle else cats

    # -------------------- Bootstrap sheet mapping ----------------------
    last_scope = s.get("_last_scope", ())
    scope_tuple = tuple(scope)
    if scope_tuple != last_scope:
        s["_mapping_prepared"] = False
        s["_last_scope"] = scope_tuple

    if not s.get("_mapping_prepared", False):
        with st.spinner("Preparing mapping…"):
            _prepare_scope_defaults(scope, s)
        s["_mapping_prepared"] = True
        st.rerun()

    # -------------------- Render mapping UI ----------------------------
    if s.all_categories_toggle:
        flags = []
        for i, c in enumerate(cats):
            if i > 0:
                st.markdown("---")
            flags.append(_render_cat_mapping(c, s))
        ready_scope = all(flags) if flags else False
    else:
        if not scope:
            st.info("Select at least one category to continue.")
            ready_scope = False
        else:
            flags = []
            for i, c in enumerate(scope):
                if i > 0:
                    st.markdown("---")
                flags.append(_render_cat_mapping(c, s))
            ready_scope = all(flags) if flags else False

    st.markdown("---")

    # -------------------- Proceed Button -------------------------------
    proceed = st.button("Proceed", type="primary", disabled=not ready_scope, use_container_width=True)

    if proceed:
        s.sheet_mapping_list = {
            c: [s.cat_map[c][r] for r in REQUIRED_SHEETS[c]]
            for c in scope
        }
        s.sheet_mapping_pairs = {
            c: {r: s.cat_map[c][r] for r in REQUIRED_SHEETS[c]}
            for c in scope
        }
        s.sheet_sources = {
            c: _get_filelike_and_label_for_cat(c, s)[0]
            for c in scope
        }
        s.sheet_mapping_saved = True
        s["_force_scroll_top"] = True
        s.page = "next"   # go to Third Page (Column Mapping)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
