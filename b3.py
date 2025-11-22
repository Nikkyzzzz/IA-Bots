# ============================== b3.py — Banking Column Mapping ==============================
import streamlit as st
import pandas as pd
import re
from io import BytesIO
import base64
from pathlib import Path
import streamlit.components.v1 as components

LEFT_LOGO_PATH = "logo.png"

# ---------------- Field Requirements ----------------
FIELD_REQUIREMENTS = {
    "Banking": {
        "Loan Dump": [
            'ASSET', 'INT_RATE', 'URI', 'CUST_CATEGORY', 'PROVISION', 'AMT_OS',
            'RESTRUCTURED_FLG', 'RESTR_DATE', 'FB_NFB_FLG', 'OUT_ORD_DT',
            'DRAW_LMT', 'SECTOR', 'SANC_LMT'
            # "PIN CODE" will be added dynamically if blacklist uploaded
        ]
    },
    "Blacklisted PIN CODE": {
        "Blacklisted PIN CODE": ["PIN CODE"]
    },
    "Loan Book (31.03.2025)": {
        "Loan Book (31.03.2025)": ["PROJECT NO", "LOAN OUTSTANDING (Rs.)", "Asset classification"]
    },
    "Loan Book (30.06.2025)": {
        "Loan Book (30.06.2025)": ["PROJECT NO", "LOAN OUTSTANDING (Rs.)", "Asset classification"]
    },
}

# ---------------- Helpers ----------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

def _get_friendly_display_name(cat: str, req_sheet: str) -> str:
    """Get user-friendly display name for category and sheet"""
    if cat == "Banking" and req_sheet == "Loan Dump":
        return "Loan Dump file"
    elif cat == "Blacklisted PIN CODE" and req_sheet == "Blacklisted PIN CODE":
        return "Blacklisted PIN code file"
    elif cat == "Loan Book (31.03.2025)" and req_sheet == "Loan Book (31.03.2025)":
        return "Loan Book Base Period"
    elif cat == "Loan Book (30.06.2025)" and req_sheet == "Loan Book (30.06.2025)":
        return "Loan Book Comparison Period"
    else:
        return f"{cat} • {req_sheet}"  # fallback

def _bytes_for_cat(cat, s):
    if cat == "Banking":
        return s.get("u_ccis_bytes"), s.get("u_ccis_name","")
    elif cat == "Blacklisted PIN CODE":
        return s.get("u_blacklist_bytes"), s.get("u_blacklist_name","")
    elif cat == "Loan Book (31.03.2025)":
        return s.get("u_loan_mar_bytes"), s.get("u_loan_mar_name","")
    elif cat == "Loan Book (30.06.2025)":
        return s.get("u_loan_jun_bytes"), s.get("u_loan_jun_name","")
    return None, ""

def _columns_for_sheet(xls_bytes: bytes, sheet_name: str, header_row: int = 0):
    try:
        bio = BytesIO(xls_bytes)
        xl = pd.ExcelFile(bio)
        df = xl.parse(sheet_name=sheet_name, header=header_row, nrows=0)
        return list(map(str, df.columns.tolist()))
    except Exception:
        return []

def _auto_map_fields(required_fields, actual_cols):
    idx = { _norm(c): c for c in actual_cols }
    return { f: idx.get(_norm(f), "") for f in required_fields }

def _make_field_on_change(cat: str, req_sheet: str, field_name: str, all_fields: list[str]):
    key_root_norm = _norm(f"{cat}::{req_sheet}")
    key_current = f"fieldmap_{key_root_norm}_{_norm(field_name)}"
    def _cb():
        s = st.session_state
        if s.get("_field_bootstrap_active", False): return
        new_val = s.get(key_current, "")
        if new_val:
            for other in all_fields:
                if other == field_name: continue
                other_key = f"fieldmap_{key_root_norm}_{_norm(other)}"
                if s.get(other_key, "") == new_val:
                    s[other_key] = ""
        s.field_map.setdefault(cat, {})[req_sheet] = {
            f: s.get(f"fieldmap_{key_root_norm}_{_norm(f)}", "") for f in all_fields
        }
    return _cb

def _prepare_field_defaults(s):
    # Skip re-preparation if column mapping is already saved and complete
    if s.get("column_mapping_saved", False) and s.get("column_mapping_pairs"):
        return
    
    s["_field_bootstrap_active"] = True
    try:
        if "field_map" not in s: s.field_map = {}
        for cat, pair_map in s.sheet_mapping_pairs.items():
            for req_sheet, mapped_sheet in pair_map.items():
                xbytes, _ = _bytes_for_cat(cat, s)
                if not xbytes or not mapped_sheet: continue
                # Header row choice
                header_row = 1 if cat == "Loan Book (31.03.2025)" else (2 if cat == "Loan Book (30.06.2025)" else 0)
                actual_cols = _columns_for_sheet(xbytes, mapped_sheet, header_row)
                need_fields = FIELD_REQUIREMENTS.get(cat, {}).get(req_sheet, [])
                # Dynamically add PIN CODE to Loan Dump if blacklist uploaded
                if cat == "Banking" and s.get("u_blacklist_bytes") and "PIN CODE" not in need_fields:
                    need_fields = need_fields + ["PIN CODE"]
                    FIELD_REQUIREMENTS["Banking"]["Loan Dump"] = need_fields
                if not actual_cols or not need_fields: continue
                prev = s.field_map.get(cat, {}).get(req_sheet, {})
                base = prev if prev else _auto_map_fields(need_fields, actual_cols)
                key_root_norm = _norm(f"{cat}::{req_sheet}")
                options = [""] + list(actual_cols)
                for f in need_fields:
                    k = f"fieldmap_{key_root_norm}_{_norm(f)}"
                    v = base.get(f, "")
                    if v not in options: v = ""
                    st.session_state[k] = v
                s.field_map.setdefault(cat, {})[req_sheet] = {
                    f: st.session_state.get(f"fieldmap_{key_root_norm}_{_norm(f)}", "")
                    for f in need_fields
                }
    finally:
        s["_field_bootstrap_active"] = False

def _render_sheet_field_mapping(cat: str, req_sheet: str, mapped_sheet: str, s) -> bool:
    xbytes, filename = _bytes_for_cat(cat, s)
    friendly_name = _get_friendly_display_name(cat, req_sheet)
    display_name = f"{friendly_name}: {filename}" if filename else friendly_name
    st.caption(f"{display_name} → {mapped_sheet}")
    if not xbytes or not mapped_sheet:
        s.field_map.setdefault(cat, {})[req_sheet] = {}
        st.warning("Missing sheet mapping or source file."); return False

    header_row = 1 if cat == "Loan Book (31.03.2025)" else (2 if cat == "Loan Book (30.06.2025)" else 0)
    actual_cols = _columns_for_sheet(xbytes, mapped_sheet, header_row)
    if not actual_cols:
        s.field_map.setdefault(cat, {})[req_sheet] = {}
        st.error("Could not read columns from the selected sheet."); return False

    need_fields = FIELD_REQUIREMENTS.get(cat, {}).get(req_sheet, [])
    if not need_fields:
        s.field_map.setdefault(cat, {})[req_sheet] = {}
        st.info("No field requirements configured."); return False

    key_root_norm = _norm(f"{cat}::{req_sheet}")
    options = [""] + list(actual_cols)

    # ✅ FIX: always return containers, never st module
    if len(need_fields) > 1:
        cols = st.columns(2)
    else:
        cols = [st.container()]

    for i, f in enumerate(need_fields):
        with cols[i % len(cols)]:
            k = f"fieldmap_{key_root_norm}_{_norm(f)}"
            st.selectbox(f"{f}", options=options, key=k,
                         on_change=_make_field_on_change(cat, req_sheet, f, need_fields))

    snap = { f: st.session_state.get(f"fieldmap_{key_root_norm}_{_norm(f)}", "") for f in need_fields }
    s.field_map.setdefault(cat, {})[req_sheet] = snap

    if all(bool(v) for v in snap.values()):
        st.success("All fields mapped."); return True
    st.info("Map all fields to continue."); return False


def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists(): return ""
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

# ---------------- Main Renderer ----------------
def render_bank3():
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
                <div class="ab-subtitle">Banking Column Mapping</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    if "sheet_mapping_pairs" not in s or not s.sheet_mapping_pairs:
        st.warning("No sheet mapping found. Go back and complete sheet mapping.")
        if st.button("⟵ Back"):
            s["_force_scroll_top"] = True
            s.page = "bankprocess"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if "field_map" not in s: s.field_map = {}

    # ---- Prepare defaults (skip if already saved) ----
    if not (s.get("_field_prepared", False) or (s.get("column_mapping_saved", False) and s.get("column_mapping_pairs"))):
        with st.spinner("Preparing column mapping…"):
            _prepare_field_defaults(s)
        s["_field_prepared"] = True
        st.rerun()

    # ---- Render per category ----
    flags = []
    
    # Check if column mapping already saved to avoid re-prompting
    if s.get("column_mapping_saved", False) and s.get("column_mapping_pairs"):
        st.info("✅ Column mapping already completed. Click 'Proceed' to review and confirm.")
        selection_ready = True
    else:
        for cat, pair_map in s.sheet_mapping_pairs.items():
            for req_sheet, mapped_sheet in pair_map.items():
                friendly_name = _get_friendly_display_name(cat, req_sheet)
                _, filename = _bytes_for_cat(cat, s)
                display_name = f"{friendly_name}: {filename}" if filename else friendly_name
                st.subheader(display_name)
                flags.append(_render_sheet_field_mapping(cat, req_sheet, mapped_sheet, s))
        selection_ready = all(flags) if flags else False

    st.markdown("---")
    proceed = st.button("Proceed ➜", type="primary", disabled=not selection_ready, use_container_width=True)
    if proceed:
        s.column_mapping_pairs = {}
        s.column_mapping_list = {}
        for cat, pair_map in s.sheet_mapping_pairs.items():
            s.column_mapping_pairs[cat] = {}
            s.column_mapping_list[cat] = {}
            for req_sheet, _ in pair_map.items():
                fm = s.field_map.get(cat, {}).get(req_sheet, {})
                req_fields = FIELD_REQUIREMENTS.get(cat, {}).get(req_sheet, list(fm.keys()))
                s.column_mapping_pairs[cat][req_sheet] = fm
                s.column_mapping_list[cat][req_sheet] = [fm.get(f, "") for f in req_fields]
        s.column_mapping_saved = True
        s["_force_scroll_top"] = True
        s.page = "bank4"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
