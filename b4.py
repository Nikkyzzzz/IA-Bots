# ============================== b4.py — Loan Dump Review & Confirm ==============================
import streamlit as st
import pandas as pd
import base64
from pathlib import Path
import streamlit.components.v1 as components

LEFT_LOGO_PATH = "logo.png"  # shared asset

# Field requirements (must match b3.py)
FIELD_REQUIREMENTS = {
    "Banking": {
        "Loan Dump": [
            'ASSET','INT_RATE','URI','CUST_CATEGORY','PROVISION','AMT_OS',
            'RESTRUCTURED_FLG','RESTR_DATE','FB_NFB_FLG','OUT_ORD_DT',
            'DRAW_LMT','SECTOR','SANC_LMT'
            # "PIN CODE" added dynamically if blacklist uploaded
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

# Friendly display names for categories
CATEGORY_DISPLAY_NAMES = {
    "Banking": "Loan Dump file: Sample Loan Dump.xlsx",
    "Blacklisted PIN CODE": "Blacklisted PIN code file: Blacklisted PIN CODES.xlsx",
    "Loan Book (31.03.2025)": "Loan Book Base Period: Loan Book (31.03.2025).xlsx",
    "Loan Book (30.06.2025)": "Loan Book Comparison Period: Loan Book (30.06.2025).xlsx"
}

def _get_friendly_display_name(cat):
    return CATEGORY_DISPLAY_NAMES.get(cat, cat)

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
                  if (el && el.scrollIntoView) {
                    el.scrollIntoView({block:"start", inline:"nearest"});
                  } else {
                    (window.parent||window).scrollTo(0,0);
                  }
                }catch(e){ (window.parent||window).scrollTo(0,0); }
              }
              go(); setTimeout(go,50); setTimeout(go,150); setTimeout(go,300); setTimeout(go,600);
            })();
            </script>
            """,
            height=0,
        )

def render_bank4():
    s = st.session_state

    # ===== Header =====
    st.markdown(
        """
        <style>
          .ab-wrap * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
          .ab-title {
            font-weight:860; letter-spacing:.02em;
            background: linear-gradient(90deg, #1e3a8a 0%, #60a5fa 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; color: transparent;
            font-size: clamp(30px, 3.2vw, 38px); line-height: 1.05;
          }
          .ab-underline { height:8px; border-radius:9999px; background:#ff7636; margin-top:8px; width:320px; max-width:30vw; }
          .ab-subtitle { margin-top:10px; font-weight:700; color:#0f172a; font-size: clamp(16px, 1.6vw, 20px); }
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
                <div class="ab-subtitle">Loan Dump Sheet Mapping</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    # ===== Guards =====
    if "sheet_mapping_pairs" not in s or not s.sheet_mapping_pairs:
        st.warning("No sheet mappings found. Go back and complete sheet mapping.")
        if st.button("⟵ Back to Sheet Mapping"):
            s.page = "bank2"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if "column_mapping_pairs" not in s or not s.column_mapping_pairs:
        st.warning("No column mappings found. Go back and complete column mapping.")
        if st.button("⟵ Back to Column Mapping"):
            s.page = "bank3"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ===== Review content =====
    for cat, pair_map in s.sheet_mapping_pairs.items():
        st.header(_get_friendly_display_name(cat))

        st.subheader("Sheet Mapping")
        sheet_rows = [{"Required Sheet": req, "Mapped Sheet": sel} for req, sel in pair_map.items()]
        st.dataframe(pd.DataFrame(sheet_rows), use_container_width=True)

        st.subheader("Column Mapping")
        for req_sheet, mapped_sheet in pair_map.items():
            st.markdown(f"**{req_sheet}** → _{mapped_sheet}_")
            fm = (s.column_mapping_pairs.get(cat, {}) or {}).get(req_sheet, {})
            if not fm:
                st.info("No field mapping available for this sheet.")
                continue
            req_fields = FIELD_REQUIREMENTS.get(cat, {}).get(req_sheet, list(fm.keys()))
            # dynamically add PIN CODE for CCIS if blacklist uploaded
            if cat == "Banking" and s.get("u_blacklist_bytes") and "PIN CODE" not in req_fields:
                req_fields = req_fields + ["PIN CODE"]
            rows = [{"Required Field": f, "Mapped Column": fm.get(f, "")} for f in req_fields]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")

    # ===== Navigation =====
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⟵ Back to Column Mapping"):
            s.page = "bank3"; st.rerun()
    with col2:
        if st.button("Confirm & Continue ➜", type="primary", use_container_width=True):
            s.review_confirmed = True
            s["_force_scroll_top"] = True
            s.page = "bank5"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
