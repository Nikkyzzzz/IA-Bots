# ============================== fourthpage.py — Review & Confirm ==============================
import streamlit as st
import pandas as pd
import base64
from pathlib import Path
import streamlit.components.v1 as components

LEFT_LOGO_PATH = "logo.png"  # shared asset

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
    import base64 as _b64
    b64 = _b64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# Same robust scroll-to-brand used on second/third/process pages
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

def render_fourth():
    s = st.session_state

    # Header styles (match third/process pages)
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

    # Header
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
                <div class="ab-subtitle">Review &amp; Confirm Mappings</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    # Guards
    if "sheet_mapping_pairs" not in s or not s.sheet_mapping_pairs:
        st.warning("No sheet mappings found. Go back and complete sheet mapping.")
        if st.button("⟵ Back to Sheet Mapping"):
            s["_force_scroll_top"] = True
            s.page = "process"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if "column_mapping_pairs" not in s or not s.column_mapping_pairs:
        st.warning("No field mappings found. Go back and complete field mapping.")
        if st.button("⟵ Back to Field Mapping"):
            s["_force_scroll_top"] = True
            s.page = "next"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Review content
    for cat, sheets_map in s.sheet_mapping_pairs.items():
        st.header(cat)

        st.subheader("Sheet Mapping")
        sheet_rows = [{"Required Sheet": req, "Mapped Sheet": sel} for req, sel in sheets_map.items()]
        st.dataframe(pd.DataFrame(sheet_rows), use_container_width=True)

        st.subheader("Field Mapping")
        for req_sheet, mapped_sheet in sheets_map.items():
            st.markdown(f"**{req_sheet}** → _{mapped_sheet}_")
            fm = (s.column_mapping_pairs.get(cat, {}) or {}).get(req_sheet, {})
            if not fm:
                st.info("No field mapping available for this sheet.")
                continue
            req_fields = FIELD_REQUIREMENTS.get(cat, {}).get(req_sheet, list(fm.keys()))
            rows = [{"Required Field": f, "Mapped Column": fm.get(f, "")} for f in req_fields]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⟵ Back to Field Mapping"):
            s["_force_scroll_top"] = True
            s.page = "next"; st.rerun()
    with col2:
        if st.button("Confirm & Continue"):
            s.review_confirmed = True
            s["_force_scroll_top"] = True
            s.page = "selection"   # ✅ FIXED
            st.rerun()

