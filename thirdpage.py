import streamlit as st
import pandas as pd
import re
from io import BytesIO
import base64
from pathlib import Path
import streamlit.components.v1 as components

LEFT_LOGO_PATH = "logo.png"
ALL_CATEGORIES_LABEL = "All Categories"

FIELD_REQUIREMENTS = {
    "P2P": {
        "P2P Sample": [
            "Vendor Name", "PO No", "PO Date", "PO Quantity", "PO Amount", "PO Approved By",
            "GRN No", "GRN Date", "GRN Quantity",
            "Invoice Date", "Invoice Quantity", "Invoice Amount", "Creator ID"
        ],
        "Vendor Master": ["Vendor Name", "GST", "PAN", "Bank Account", "Creator ID"],
        "Employee Master": ["Employee ID", "Employee Name", "Department", "Creator ID", "Designation"],  # NEW
        # "Authority Matrix": [   # NEW SHEET
        #     "Department",
        #     "Role",
        #     "Creation Authority",
        #     "Creation Amount Limit",
        #     "Approval Authority",
        #     "Approval Amount Limit"
        # ]
    },

    "O2C": {
        "O2C Sample": ["SO Date","Delivery Date","Invoice No", # add these 3 required for new bots
    "SO No",
    "Invoice Amount",
    "Taxable Amount"],
        "Customer Master": ["GST No","PAN No","Credit Limit"],
    },
    "H2R": {
        "Employee Master": [
            "Employee ID",
            "Employee Name",
            "Exit Date",
            "Status",
            "Bank Account",   # NEW – needed for H2R44
            "PAN No",         # NEW – needed for H2R44
        ],
        "Attendance Register": ["Employee ID", "Employee Name", "Month"],
    },

}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

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
                pass

def _bytes_for_cat(cat: str, s):
    if cat == "P2P":
        if s.get("u_p2p_bytes"): return s["u_p2p_bytes"], s.get("u_p2p_name","")
    if cat == "O2C":
        if s.get("u_o2c_bytes"): return s["u_o2c_bytes"], s.get("u_o2c_name","")
    if cat == "H2R":
        if s.get("u_h2r_bytes"): return s["u_h2r_bytes"], s.get("u_h2r_name","")
    if s.get("u_master_bytes"):  return s["u_master_bytes"], s.get("u_master_name","")
    return None, ""

def _columns_for_sheet(xls_bytes: bytes, sheet_name: str):
    try:
        bio = BytesIO(xls_bytes)
        xl = pd.ExcelFile(bio)
        df = xl.parse(sheet_name=sheet_name, nrows=0)
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

def _prepare_field_defaults(scope_cats, s):
    s["_field_bootstrap_active"] = True
    try:
        if "field_map" not in s: s.field_map = {}
        for c in scope_cats:
            pair_map = s.sheet_mapping_pairs.get(c, {})
            for req_sheet, mapped_sheet in pair_map.items():
                xbytes, _ = _bytes_for_cat(c, s)
                if not xbytes or not mapped_sheet: continue
                actual_cols = _columns_for_sheet(xbytes, mapped_sheet)
                need_fields = FIELD_REQUIREMENTS.get(c, {}).get(req_sheet, [])
                if not actual_cols or not need_fields: continue
                prev = s.field_map.get(c, {}).get(req_sheet, {})
                base = prev if prev else _auto_map_fields(need_fields, actual_cols)
                key_root_norm = _norm(f"{c}::{req_sheet}")
                options = [""] + list(actual_cols)
                for f in need_fields:
                    k = f"fieldmap_{key_root_norm}_{_norm(f)}"
                    v = base.get(f, "")
                    if v not in options: v = ""
                    st.session_state[k] = v
                s.field_map.setdefault(c, {})[req_sheet] = {
                    f: st.session_state.get(f"fieldmap_{key_root_norm}_{_norm(f)}", "")
                    for f in need_fields
                }
    finally:
        s["_field_bootstrap_active"] = False

def _render_sheet_field_mapping(cat: str, req_sheet: str, mapped_sheet: str, s) -> bool:
    xbytes, _ = _bytes_for_cat(cat, s)
    st.caption(f"{cat} • {req_sheet} → {mapped_sheet}")
    if not xbytes or not mapped_sheet:
        s.field_map.setdefault(cat, {})[req_sheet] = {}
        st.warning("Missing sheet mapping or source file."); return False
    actual_cols = _columns_for_sheet(xbytes, mapped_sheet)
    if not actual_cols:
        s.field_map.setdefault(cat, {})[req_sheet] = {}
        st.error("Could not read columns from the selected sheet."); return False
    need_fields = FIELD_REQUIREMENTS.get(cat, {}).get(req_sheet, [])
    if not need_fields:
        s.field_map.setdefault(cat, {})[req_sheet] = {}
        st.info("No field requirements configured."); return False
    key_root_norm = _norm(f"{cat}::{req_sheet}")
    options = [""] + list(actual_cols)
    cols = st.columns(2) if len(need_fields) > 1 else [st]
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

def render_next():
    s = st.session_state
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
                <div class="ab-subtitle">Column Mapping</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    _bootstrap_bytes(s)
    if "sheet_mapping_pairs" not in s or not s.sheet_mapping_pairs:
        st.warning("No sheet mapping found. Go back and complete sheet mapping.")
        if st.button("⟵ Back"):
            s["_force_scroll_top"] = True
            s.page = "process"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if "field_map" not in s: s.field_map = {}
    cats = list(s.sheet_mapping_pairs.keys())

    with st.sidebar:
        st.header("Categories")
        all_mode = st.toggle(ALL_CATEGORIES_LABEL, value=True, key="all_categories_toggle_fields")
        selected_categories = cats if all_mode else st.multiselect("Choose categories", options=cats, default=cats, key="multi_category_select_fields")

    scope = selected_categories if not st.session_state.get("all_categories_toggle_fields", True) else cats

    last_scope = s.get("_last_field_scope", ())
    scope_tuple = tuple(scope)
    if scope_tuple != last_scope:
        s["_field_prepared"] = False
        s["_last_field_scope"] = scope_tuple

    if not s.get("_field_prepared", False):
        with st.spinner("Preparing column mapping…"):
            _prepare_field_defaults(scope, s)
        s["_field_prepared"] = True
        st.rerun()

    if not scope:
        st.info("Select at least one category to continue.")
        selection_ready = False
    else:
        flags = []
        for ci, c in enumerate(scope):
            if ci > 0: st.markdown("---")
            for req_sheet, mapped_sheet in s.sheet_mapping_pairs[c].items():
                st.subheader(f"{c} • {req_sheet}")
                flags.append(_render_sheet_field_mapping(c, req_sheet, mapped_sheet, s))
        selection_ready = all(flags) if flags else False

    st.markdown("---")
    proceed = st.button("Proceed ➜", type="primary", disabled=not selection_ready, use_container_width=True)
    if proceed:
        s.column_mapping_pairs = {}
        s.column_mapping_list = {}
        for c in scope:
            s.column_mapping_pairs[c] = {}
            s.column_mapping_list[c] = {}
            for req_sheet, _ in s.sheet_mapping_pairs[c].items():
                fm = s.field_map.get(c, {}).get(req_sheet, {})
                req_fields = FIELD_REQUIREMENTS.get(c, {}).get(req_sheet, list(fm.keys()))
                s.column_mapping_pairs[c][req_sheet] = fm
                s.column_mapping_list[c][req_sheet] = [fm.get(f, "") for f in req_fields]
        s.column_mapping_saved = True
        s["_force_scroll_top"] = True
        s.page = "fourth"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
