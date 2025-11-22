# ============================== b5.py — Banking Ready ==============================
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64

LEFT_LOGO_PATH = "logo.png"

BASE_BANKING_BOTS = [
    "zero_or_null_roi_loans",
    "standard_accounts_with_uri_zero",
    "provision_verification_substandard_npa",
    "restructured_standard_accounts",
    "provision_verification_doubtful3_npa",
    "npa_fb_accounts_overdue",
    "negative_amt_outstanding",
    "standard_accounts_overdue_details",
    "standard_accounts_with_odd_interest",
    "agri0_sector_over_limit",
    "misaligned_scheme_for_facilities",  # ✅ new
]

EXTRA_BOT_BLACKLIST = "Loans & Advances to Blacklisted Areas"
EXTRA_BOT_LOANBOOKS = "Blank Asset Classification"

# ---------- header helpers ----------
def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists(): return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
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
              go(); setTimeout(go,50); setTimeout(go,150); setTimeout(go,300); setTimeout(go,600);
            })();
            </script>
            """,
            height=0,
        )

def _render_header():
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
          .hint { color:#475569; font-size: 0.92rem; margin: .25rem 0 .75rem 0; }
          .box { border:1px solid rgba(15,23,42,.06); border-radius:14px; padding:12px; background:#fff; }
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
                <div class="ab-subtitle">Banking • Select Bots</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)

# ---------- page ----------
def render_bank5():
    s = st.session_state
    st.set_page_config(layout="wide")

    _render_header()
    _scroll_to_brand_if_needed(s)

    if "selected_bots" not in s or not isinstance(s.selected_bots, list):
        s.selected_bots = []

    st.markdown('<div class="hint">Select the bots you want to run for Banking dataset.</div>', unsafe_allow_html=True)

    # ===== Determine bots dynamically =====
    bots = BASE_BANKING_BOTS.copy()

    ccis_present = bool(s.get("u_ccis_bytes") or s.get("u_ccis"))
    bl_present   = bool(s.get("u_blacklist_bytes") or s.get("u_blacklist"))
    mar_present  = bool(s.get("u_loan_mar_bytes") or s.get("u_loan_mar"))
    jun_present  = bool(s.get("u_loan_jun_bytes") or s.get("u_loan_jun"))

    # Add blacklist bot if both CCIS + Blacklist present
    if ccis_present and bl_present:
        bots.append(EXTRA_BOT_BLACKLIST)

    # Add loan book bot if both Loan Books present
    if mar_present and jun_present:
        bots.append(EXTRA_BOT_LOANBOOKS)

    # ===== Select All handling =====
    all_selected = set(s.selected_bots) == set(bots)
    select_all = st.checkbox("Select All Bots", value=all_selected)

    if select_all and not all_selected:
        s.selected_bots = bots.copy()
        st.rerun()
    elif not select_all and all_selected:
        s.selected_bots = []
        st.rerun()

    # ===== Bot checkboxes =====
    BOT_DESCRIPTIONS = {
        "zero_or_null_roi_loans": "Filter loans with asset codes '11' or '12' where the interest rate is 0 or null ('-').",
        "standard_accounts_with_uri_zero": "Filter accounts with asset codes '11' or '12' where the Uniform Risk Index (URI) is 0.",
        "provision_verification_substandard_npa": "Verify provisions for NPA accounts with asset codes '21' or '22' and check whether the provision percentage is less than 15%.",
        "restructured_standard_accounts": "Identify restructured standard accounts (ASSET '11' or '12') with outstanding balance, provision < 15%, and restructuring date within the last 2 years.",
        "provision_verification_doubtful3_npa": "Check Doubtful-3 NPA accounts (ASSET '31', '32', '33') and verify if the provision amount equals the outstanding amount.",
        "npa_fb_accounts_overdue": "Find fund-based (FB) NPA accounts where the overdue date is more than three months in the past.",
        "negative_amt_outstanding": "Identify accounts where the outstanding amount is a negative value.",
        "standard_accounts_overdue_details": "Filter accounts where the outstanding amount exceeds the drawing limit by more than 10%.",
        "standard_accounts_with_odd_interest": "Identify standard accounts (ASSET '11' or '12') with interest rates that are not divisible by 0.05.",
        "agri0_sector_over_limit": "Find standard accounts in the 'Agri0' sector where the outstanding amount exceeds 134% of the sanctioned limit.",
        "misaligned_scheme_for_facilities": "Flag facilities where the scheme code differs from the most common scheme for that facility.",
        "Loans & Advances to Blacklisted Areas": "Match CCIS accounts against a Blacklisted PIN list and flag all loans where the account PIN code appears in the blacklist.",
        "Blank Asset Classification": "Merge two Loan Book files (31.03.2025 & 30.06.2025), compute differences in outstanding balances, and blank out 'Asset classification' fields where balances match and classifications are identical.",
    }
        # ===== Bot checkboxes =====
    BOT_LOGICS = {
        "zero_or_null_roi_loans": "**Loans with Zero/Null ROI**:  Filter loans with asset codes '11' or '12' where the interest rate is 0 or null.",
        "standard_accounts_with_uri_zero": "**Standard Accounts having Zero URI**:  Filter accounts with asset codes '11' or '12' where the Uniform Risk Index (URI) is 0.",
        "provision_verification_substandard_npa": "**Provision Status – Sub-Standard NPA**:  Verify provisions for NPA accounts with asset codes '21' or '22' and check whether the provision percentage is less than 15%.",
        "restructured_standard_accounts": "**Restructured Standard Accounts**:  Identify restructured standard accounts (ASSET '11' or '12') with outstanding balance, provision < 15%, and restructuring date within the last 2 years.",
        "provision_verification_doubtful3_npa": "**Provision Status – Doubtful-3 NPA**:  Check Doubtful-3 NPA accounts (ASSET '31', '32', '33') and verify if the provision amount equals the outstanding amount.",
        "npa_fb_accounts_overdue": "**NPA FB Accounts with Overdues**:  Find fund-based (FB) NPA accounts where the overdue date is more than three months in the past.",
        "negative_amt_outstanding": "**Accounts with Negative Outstanding**:  Identify accounts where the outstanding amount is a negative value.",
        "standard_accounts_overdue_details": "**Overdue Standard Accounts**:  Filter accounts where the outstanding amount exceeds the drawing limit by more than 10%.",
        "standard_accounts_with_odd_interest": "**Standard Accounts with Irregular Interest**:  Identify standard accounts (ASSET '11' or '12') with interest rates that are not divisible by 0.05.",
        "agri0_sector_over_limit": "**Agriculture Sector – Limit Breach Cases**:  Find standard accounts in the 'Agri0' sector where the outstanding amount exceeds 134% of the sanctioned limit.",
        "misaligned_scheme_for_facilities": "**Facilities with Scheme Mismatch**:  Flag facilities where the scheme code differs from the most common scheme for that facility.",
        "Loans & Advances to Blacklisted Areas": "**Loans in Blacklisted Areas**:  Match CCIS accounts against a Blacklisted PIN list and flag all loans where the account PIN code appears in the blacklist.",
        "Blank Asset Classification": "**Accounts with Blank Asset Classification**:  Merge two Loan Book files, compute differences in outstanding balances, and blank out 'Asset classification' fields where balances match and classifications are identical.",
    }

    with st.container():
        st.markdown(
            """
            <style>
              .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
                margin-left: 6px;
              }
              .tooltip .tooltiptext {
                visibility: hidden;
                width: 280px;
                background-color: #1e293b;
                color: #f8fafc;
                text-align: left;
                border-radius: 8px;
                padding: 8px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -140px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 0.85rem;
                line-height: 1.3;
              }
              .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="box">', unsafe_allow_html=True)
        new_selected = []

        for bot in bots:
            bot_name = bot.replace("_", " ").title() if "_" in bot else bot
            logic_text = BOT_LOGICS.get(bot, f"{bot}")

            col1, col2 = st.columns([12, 1])
            with col1:
                checked = st.checkbox(bot_name, key=f"sel_{bot}", value=(bot in s.selected_bots))
            with col2:
                st.markdown(
                    f"""
                    <div class="tooltip">ℹ️
                      <span class="tooltiptext">{logic_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if checked:
                new_selected.append(bot)

        s.selected_bots = new_selected
        st.markdown('</div>', unsafe_allow_html=True)



    # ===== Navigation =====
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⟵ Back to Review"):
            s["_force_scroll_top"] = True
            s.page = "bank4"; st.rerun()
    with col2:
        if st.button("Continue ➜", type="primary", disabled=not s.selected_bots, use_container_width=True):
            s["_force_scroll_top"] = True
            s.page = "bank6"; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
