# ============================== b7.py — Banking Results Dashboard ==============================
import streamlit as st
import streamlit_antd_components as sac
import pandas as pd
import altair as alt
from io import BytesIO
from pathlib import Path
import base64
import streamlit.components.v1 as components
import re
import charts

import blogic6

LEFT_LOGO_PATH = "logo.png"

import plotly.express as px
import streamlit as st


# ============================== PDF CHECKING HELPERS ==============================

def extract_amount(val):
    if pd.isna(val): 
        return None
    s = str(val).replace(",", "").strip()

    # Match crore / cr
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:cr|crore)\b', s, flags=re.I)
    if m:
        return float(m.group(1)) * 1e7  # 1 crore = 10,000,000

    # Match lakh / l
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:l|lakh)\b', s, flags=re.I)
    if m:
        return float(m.group(1)) * 1e5  # 1 lakh = 100,000

    # Match thousand / k
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|thousand)\b', s, flags=re.I)
    if m:
        return float(m.group(1)) * 1e3  # 1 thousand = 1,000

    # Match bare number (assume in rupees, so keep as-is)
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m:
        return float(m.group(1))

    return None


def extract_percent(val):
    if pd.isna(val):
        return None
    s = str(val)
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', s, flags=re.I)
    if m: return float(m.group(1))
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:percent|pct)\b', s, flags=re.I)
    if m: return float(m.group(1))
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m:
        n = float(m.group(1))
        return n * 100 if n <= 1.5 else n
    return None

def extract_ratio_or_number(val):
    if pd.isna(val):
        return None
    s = str(val)
    m = re.search(r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return None if b == 0 else a / b
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:x|times)\b', s, flags=re.I)
    if m: return float(m.group(1))
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m: return float(m.group(1))
    return None

def extract_months_window(val):
    if pd.isna(val): return None
    s = str(val)
    m = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:months?|mos?|mths?)\b', s, flags=re.I)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:months?|mos?|mths?)\b', s, flags=re.I)
    if m:
        n = float(m.group(1)); return n, n
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)\b', s, flags=re.I)
    if m:
        n = float(m.group(1)) * 12; return n, n
    return None

def normalize_sno(df):
    candidates = ["S.No", "S. No", "S No", "S. No.", "SNO", "S_NO", "Serial Number", "S. no", "S no"]
    for col in df.columns:
        if col.strip() in candidates:
            if col != "S.No":
                df = df.rename(columns={col: "S.No"})
            return df
    if "S.No" not in df.columns:
        df.insert(0, "S.No", range(1, len(df) + 1))
    return df

def run_all_checks(df):
    # Resolve the "Loan amount" source row once (by Particulars)
    ser = df.get("Particulars", pd.Series(index=df.index, data="")).astype(str).str.strip().str.lower()
    la_hits = df[ser.str.startswith("loan amount")].index.tolist()
    loan_row_idx = la_hits[0] if la_hits else None
    
    checks = {
        8:  lambda v: (extract_percent(v) is not None) and (extract_percent(v) >= 30),
        9:  lambda v: (extract_ratio_or_number(v) is not None) and (extract_ratio_or_number(v) <= 3),
        10: lambda v: (extract_ratio_or_number(v) is not None) and (extract_ratio_or_number(v) >= 1.2),
        11: lambda v: (extract_ratio_or_number(v) is not None) and (extract_ratio_or_number(v) >= 1.25),
        12: lambda v: (extract_ratio_or_number(v) is not None) and (extract_ratio_or_number(v) >= 1.25),
        13: lambda v: (extract_ratio_or_number(v) is not None) and (extract_ratio_or_number(v) >= 3),
        15: lambda v: (extract_months_window(v) is not None) and (
                        extract_months_window(v)[0] >= 6 and extract_months_window(v)[1] <= 18)
    }
    results = {}
    for idx, row in df.iterrows():
        check_num = idx + 1
        if (check_num in checks) or (check_num == 14):
            row_results = {}
            for col in df.columns[3:]:
                if check_num == 14:
                    if loan_row_idx is not None:
                        v = df.loc[loan_row_idx, col]
                        row_results[col] = (extract_amount(v) is not None) and (extract_amount(v) >= 50 * 1e5)
                    else:
                        row_results[col] = False
                else:
                    row_results[col] = checks[check_num](row[col])
            results[idx] = row_results
    return results

def highlight_results(val, row_idx, col_name, results):
    if row_idx in results and col_name in results[row_idx]:
        return 'background-color: #DAF2D0;' if results[row_idx][col_name] else 'background-color: salmon;'
    return ''

def apply_highlighting(df, results):
    return df.style.apply(lambda row: [
        highlight_results(row[col], row.name, col, results) for col in df.columns
    ], axis=1)

# ---------- Fee Calculation Helpers ----------
def extract_amount(val):
    if pd.isna(val):
        return None
    s = str(val).replace(",", "").strip()

    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:cr|crore)', s, flags=re.I)
    if m: return float(m.group(1)) * 1e7

    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:l|lakh)', s, flags=re.I)
    if m: return float(m.group(1)) * 1e5

    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|thousand)', s, flags=re.I)
    if m: return float(m.group(1)) * 1e3

    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m: return float(m.group(1))

    return None

def format_amount(n):
    if n is None:
        return "-"
    if n >= 1e7:
        return f"{n/1e7:.2f}Cr"
    elif n >= 1e5:
        return f"{n/1e5:.2f}L"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(round(n, 2))

def calculate_registration_fee(loan_amt):
    if loan_amt <= 20e7:   # 20 Cr
        return 1e5
    elif loan_amt <= 125e7:
        return 2.5e5
    elif loan_amt <= 250e7:
        return 5e5
    else:
        return 10e5

def calculate_frontend_fee(loan_amt):
    if loan_amt <= 100e7:
        return 0.01 * loan_amt
    else:
        return (0.01 * 100e7) + (0.0025 * (loan_amt - 100e7))

def build_fee_table(df):
    loan_row = df[df["Particulars"].str.contains("Loan", case=False, na=False)]
    if loan_row.empty:
        return pd.DataFrame()

    results = []
    for col in df.columns[2:]:
        loan_val = loan_row.iloc[0][col]
        loan_amt = extract_amount(loan_val)
        if loan_amt is None:    
            continue

        reg_fee = calculate_registration_fee(loan_amt)
        front_fee = calculate_frontend_fee(loan_amt)

        results.append({
            "Company Name": col,
            "Loan Amount": format_amount(loan_amt),
            "Registration Fee": format_amount(reg_fee),
            "Front-end Fee": format_amount(front_fee)
        })

    return pd.DataFrame(results)
#___________________2
# ---------------- Utility helpers ---------------- #
def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else ("image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png")
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

# ---------------- Sidebar menu helpers ---------------- #
def _dynamic_menu_items_selected(selected_codes: set[str]):
    BOT_DISPLAY_NAMES = {
        "zero_or_null_roi_loans": "Loans with Zero/Null ROI",
        "standard_accounts_with_uri_zero": "Standard Accounts having Zero URI",
        "provision_verification_substandard_npa": "Provision Status – Sub-Standard NPA",
        "restructured_standard_accounts": "Restructured Standard Accounts",
        "provision_verification_doubtful3_npa": "Provision Status – Doubtful-3 NPA",
        "npa_fb_accounts_overdue": "NPA FB Accounts with Overdues",
        "negative_amt_outstanding": "Accounts with Negative Outstanding",
        "standard_accounts_overdue_details": "Overdue Standard Accounts",
        "standard_accounts_with_odd_interest": "Standard Accounts with Irregular Interest",
        "agri0_sector_over_limit": "Agriculture Sector – Limit Breach Cases",
        "misaligned_scheme_for_facilities": "Facilities with Scheme Mismatch",
        "Loans & Advances to Blacklisted Areas": "Loans in Blacklisted Areas",
        "Blank Asset Classification": "Accounts with Blank Asset Classification"
    }
    items = [sac.MenuItem("Select All", icon="grid")]
    for code, (_, pname) in blogic6.PROCESS_TITLES.items():
        if selected_codes and code not in selected_codes:
            continue
        display_name = BOT_DISPLAY_NAMES.get(code, pname)
        items.append(sac.MenuItem(display_name))
    return items

def _parse_choice(choice: str):
    if choice in (None, "", "Select All"):
        return ("all", None)
    BOT_DISPLAY_NAMES = {
        "zero_or_null_roi_loans": "Loans with Zero/Null ROI",
        "standard_accounts_with_uri_zero": "Standard Accounts having Zero URI",
        "provision_verification_substandard_npa": "Provision Status – Sub-Standard NPA",
        "restructured_standard_accounts": "Restructured Standard Accounts",
        "provision_verification_doubtful3_npa": "Provision Status – Doubtful-3 NPA",
        "npa_fb_accounts_overdue": "NPA FB Accounts with Overdues",
        "negative_amt_outstanding": "Accounts with Negative Outstanding",
        "standard_accounts_overdue_details": "Overdue Standard Accounts",
        "standard_accounts_with_odd_interest": "Standard Accounts with Irregular Interest",
        "agri0_sector_over_limit": "Agriculture Sector – Limit Breach Cases",
        "misaligned_scheme_for_facilities": "Facilities with Scheme Mismatch",
        "Loans & Advances to Blacklisted Areas": "Loans in Blacklisted Areas",
        "Blank Asset Classification": "Accounts with Blank Asset Classification"
    }
    code_map = {v: k for k, v in BOT_DISPLAY_NAMES.items()}
    code = code_map.get(choice)
    if code:
        return ("bot", code)
    code = {v[1]: k for k, v in blogic6.PROCESS_TITLES.items()}.get(choice)
    if code:
        return ("bot", code)
    return ("all", None)

def _ensure_bot_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Bot" in df.columns:
        return df
    if "Bots" in df.columns:
        return df.rename(columns={"Bots": "Bot"})
    return df

# ---------------- Enriched Summary Builder ---------------- #
def _build_enriched_summary_bank(proc_status, results, only_codes=None):
    s = st.session_state

    BOT_DISPLAY_NAMES = {
        "zero_or_null_roi_loans": "Loans with Zero/Null ROI",
        "standard_accounts_with_uri_zero": "Standard Accounts having Zero URI",
        "provision_verification_substandard_npa": "Provision Status – Sub-Standard NPA",
        "restructured_standard_accounts": "Restructured Standard Accounts",
        "provision_verification_doubtful3_npa": "Provision Status – Doubtful-3 NPA",
        "npa_fb_accounts_overdue": "NPA FB Accounts with Overdues",
        "negative_amt_outstanding": "Accounts with Negative Outstanding",
        "standard_accounts_overdue_details": "Overdue Standard Accounts",
        "standard_accounts_with_odd_interest": "Standard Accounts with Irregular Interest",
        "agri0_sector_over_limit": "Agriculture Sector – Limit Breach Cases",
        "misaligned_scheme_for_facilities": "Facilities with Scheme Mismatch",
        "Loans & Advances to Blacklisted Areas": "Loans in Blacklisted Areas",
        "Blank Asset Classification": "Accounts with Blank Asset Classification",
    }

    BOT_DETAILS = {
        "zero_or_null_roi_loans": "Filtered loans with asset codes '11' or '12' where the interest rate was 0 or null ('-').",
        "standard_accounts_with_uri_zero": "Filtered accounts with asset codes '11' or '12' where the Uniform Risk Index (URI) was 0.",
        "provision_verification_substandard_npa": "Verified provisions for NPA accounts with asset codes '21' or '22' and checked whether the provision percentage was less than 15%.",
        "restructured_standard_accounts": "Identified restructured standard accounts (ASSET '11' or '12') with outstanding balance, provision < 15%, and restructuring date within the last 2 years.",
        "provision_verification_doubtful3_npa": "Checked Doubtful-3 NPA accounts (ASSET '31', '32', '33') and verified if the provision amount equaled the outstanding amount.",
        "npa_fb_accounts_overdue": "Found fund-based (FB) NPA accounts where the overdue date was more than three months in the past.",
        "negative_amt_outstanding": "Identified accounts where the outstanding amount was a negative value.",
        "standard_accounts_overdue_details": "Filtered accounts where the outstanding amount exceeded the drawing limit by more than 10%.",
        "standard_accounts_with_odd_interest": "Identified standard accounts (ASSET '11' or '12') with interest rates that were not divisible by 0.05.",
        "agri0_sector_over_limit": "Found standard accounts in the 'Agri0' sector where the outstanding amount exceeded 134% of the sanctioned limit.",
        "misaligned_scheme_for_facilities": "Flagged facilities where the scheme code differed from the most common scheme for that facility.",
        "Loans & Advances to Blacklisted Areas": "Matched CCIS accounts against a Blacklisted PIN list and flagged all loans where the account PIN code appeared in the blacklist.",
        "Blank Asset Classification": "Merged Loan Book (31.03.2025 & 30.06.2025), computed differences, and blanked out 'Asset classification' if balances and classifications matched.",
    }
    BOT_FILES = {
        "zero_or_null_roi_loans": ["u_ccis_name"],
        "standard_accounts_with_uri_zero": ["u_ccis_name"],
        "provision_verification_substandard_npa": ["u_ccis_name"],
        "restructured_standard_accounts": ["u_ccis_name"],
        "provision_verification_doubtful3_npa": ["u_ccis_name"],
        "npa_fb_accounts_overdue": ["u_ccis_name"],
        "negative_amt_outstanding": ["u_ccis_name"],
        "standard_accounts_overdue_details": ["u_ccis_name"],
        "standard_accounts_with_odd_interest": ["u_ccis_name"],
        "agri0_sector_over_limit": ["u_ccis_name"],
        "misaligned_scheme_for_facilities": ["u_ccis_name"],
        "Loans & Advances to Blacklisted Areas": ["u_ccis_name", "u_blacklist_name"],
        "Blank Asset Classification": ["u_loan_mar_name", "u_loan_jun_name"],
    }

    BOT_SHEETS = {
        "zero_or_null_roi_loans": [("Banking", "Loan Dump")],
        "standard_accounts_with_uri_zero": [("Banking", "Loan Dump")],
        "provision_verification_substandard_npa": [("Banking", "Loan Dump")],
        "restructured_standard_accounts": [("Banking", "Loan Dump")],
        "provision_verification_doubtful3_npa": [("Banking", "Loan Dump")],
        "npa_fb_accounts_overdue": [("Banking", "Loan Dump")],
        "negative_amt_outstanding": [("Banking", "Loan Dump")],
        "standard_accounts_overdue_details": [("Banking", "Loan Dump")],
        "standard_accounts_with_odd_interest": [("Banking", "Loan Dump")],
        "agri0_sector_over_limit": [("Banking", "Loan Dump")],
        "misaligned_scheme_for_facilities": [("Banking", "Loan Dump")],
        "Loans & Advances to Blacklisted Areas": [
            ("Banking", "Loan Dump"),
            ("Blacklisted PIN CODE", "Blacklisted PIN CODE"),
        ],
        "Blank Asset Classification": [
            ("Loan Book (31.03.2025)", "Loan Book (31.03.2025)"),
            ("Loan Book (30.06.2025)", "Loan Book (30.06.2025)"),
        ],
    }

    def _get_mapped_sheet(cat, need):
        pairs = s.get("sheet_mapping_pairs", {})
        val = pairs.get(cat, {}).get(need, "")
        if not val:
            cat_map = s.get("cat_map", {})
            val = cat_map.get(cat, {}).get(need, "")
        return val or ""

    rows = []
    for code, status in proc_status.items():
        if only_codes and code not in only_codes:
            continue

        display_name = BOT_DISPLAY_NAMES.get(code, code)
        logic_desc = BOT_DETAILS.get(code, "")

        files_used = [s.get(k, "") for k in BOT_FILES.get(code, []) if s.get(k, "")]
        file_used_str = " / ".join(files_used)

        sheets_used = []
        for cat, need in BOT_SHEETS.get(code, []):
            sheet_val = _get_mapped_sheet(cat, need)
            if sheet_val:
                display_cat = "Sample Loan Dump" if cat == "Banking" else cat
                sheets_used.append(f"{display_cat}: {sheet_val}")
        sheet_used_str = " | ".join(sheets_used)

        issues_found = 0
        if code in results and results[code] is not None:
            try:
                issues_found = len(results[code])
            except Exception:
                issues_found = 0

        rows.append({
            "Bot": display_name,
            "Logic Description": logic_desc,
            "File Used": file_used_str,
            "Sheet Used": sheet_used_str,
            "Status": status,
            "Issues Found": issues_found,
            "_code": code,
        })

    return pd.DataFrame(rows)
#____________3
# ---------------- Excel Report Builder ---------------- #
def _build_detailed_report_excel_bank(proc_status: dict, results: dict, only_codes: set[str] | None) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Summary sheet
        summary_df = _build_enriched_summary_bank(proc_status, results, only_codes)
        if summary_df is None or summary_df.empty:
            summary_df = pd.DataFrame()
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Each bot’s output
        codes = list(blogic6.PROCESS_TITLES.keys())
        if only_codes:
            codes = [c for c in codes if c in only_codes]

        for code in codes:
            df = results.get(code)
            if df is not None and not df.empty:
                _, pname = blogic6.PROCESS_TITLES[code]
                sheet_name = pname[:28]  # Excel sheet name ≤ 31 chars
                pd.DataFrame([{"Total Records": len(df)}]).to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=0
                )
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)

    output.seek(0)
    return output.getvalue()


# ---------------- Main Render ---------------- #
def render_bank7():
    s = st.session_state
    st.set_page_config(layout="wide")

    # ---------- Branding ----------
    st.markdown(
        """
        <style>
          .ab-wrap * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
          .ab-title { font-weight:860; letter-spacing:.02em; background:linear-gradient(90deg,#1e3a8a 0%,#60a5fa 100%);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; color:transparent;
                      font-size:clamp(30px,3.2vw,38px); line-height:1.05; }
          .ab-underline { height:8px; border-radius:9999px; background:#ff7636; margin-top:8px; width:320px; max-width:30vw; }
          .ab-subtitle { margin-top:10px; font-weight:700; color:#0f172a; font-size:clamp(16px,1.6vw,20px); }
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
                <div class="ab-subtitle">Banking Dashboard</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    if not s.get("processing_done"):
        st.warning("No processed results found. Please run processing first.")
        if st.button("⟵ Back to Processing", key="go_processing_btn"):
            s.page = "b6"; st.rerun()
        return

    results     = s.get("results", {})
    proc_status = s.get("proc_status", {})

    selected_codes = list(s.get("selected_bots", []) or [])
    selected_set   = set(selected_codes)

    # Sidebar with Select All + Bots
    menu_items = _dynamic_menu_items_selected(selected_set)
    with st.sidebar:
        choice = sac.menu(items=menu_items, open_all=True, indent=16)
    sel_mode, sel_value = _parse_choice(choice)

    # Tabs
    tabs = st.tabs(["Analysis", "Output", "Report", "Data Extraction"])

    # ---------- Analysis ----------
    with tabs[0]:
        st.subheader("Summary Overview")
        summary = _build_enriched_summary_bank(
            proc_status, results, only_codes=selected_set if selected_set else None
        )
        if sel_mode == "bot" and sel_value:
            summary = summary[summary["_code"] == sel_value]
        cols = [c for c in summary.columns if c != "_code"]
        st.dataframe(summary[cols].reset_index(drop=True), use_container_width=True)

        st.subheader("Issues Per Bot")
        if not summary.empty:
            chart = (
                alt.Chart(summary)
                .mark_bar()
                .encode(
                    x=alt.X("Issues Found:Q", title="Issues Found"),
                    y=alt.Y("Bot:N", sort=None, title="Bots", axis=alt.Axis(labelAngle=0)),
                    color="Status:N",
                    tooltip=["Bot", "Issues Found", "Status"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data to plot for the current selection.")

        # 🔹 Loan Book Charts
        loan_file1 = s.get("u_loan_mar_bytes")
        loan_file2 = s.get("u_loan_jun_bytes")
        loan_name1 = s.get("u_loan_mar_name", "Loan Book 1")
        loan_name2 = s.get("u_loan_jun_name", "Loan Book 2")

        if loan_file1 and loan_file2:
            st.subheader("Asset Classification (Base Period Vs Comparative Period)")

            col1, col2 = st.columns(2)
            with col1:
                charts.compare_project_counts_plotly(BytesIO(loan_file1), BytesIO(loan_file2), key="asset_count_analysis")
            with col2:
                charts.compare_loan_outstanding_plotly(BytesIO(loan_file1), BytesIO(loan_file2), key="asset_outstanding_analysis")

            col3, col4 = st.columns(2)
            with col3:
                charts.compare_project_counts_sma(BytesIO(loan_file1), BytesIO(loan_file2), key="sma_count_analysis")
            with col4:
                charts.compare_loan_outstanding_sma(BytesIO(loan_file1), BytesIO(loan_file2), key="sma_outstanding_analysis")
        else:
            st.info("Upload both Loan Book 1 and Loan Book 2 in Banking Home to see comparisons.")


#__________________4

    # ---------- Output ----------
    BOT_DISPLAY_NAMES = {
        "zero_or_null_roi_loans": "Loans with Zero/Null ROI",
        "standard_accounts_with_uri_zero": "Standard Accounts having Zero URI",
        "provision_verification_substandard_npa": "Provision Status – Sub-Standard NPA",
        "restructured_standard_accounts": "Restructured Standard Accounts",
        "provision_verification_doubtful3_npa": "Provision Status – Doubtful-3 NPA",
        "npa_fb_accounts_overdue": "NPA FB Accounts with Overdues",
        "negative_amt_outstanding": "Accounts with Negative Outstanding",
        "standard_accounts_overdue_details": "Overdue Standard Accounts",
        "standard_accounts_with_odd_interest": "Standard Accounts with Irregular Interest",
        "agri0_sector_over_limit": "Agriculture Sector – Limit Breach Cases",
        "misaligned_scheme_for_facilities": "Facilities with Scheme Mismatch",
        "Loans & Advances to Blacklisted Areas": "Loans in Blacklisted Areas",
        "Blank Asset Classification": "Accounts with Blank Asset Classification"
    }


    # List of first 11 bots (using the same input sheet)
    FIRST_11_BOTS = [
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
        "misaligned_scheme_for_facilities",
        "Loans & Advances to Blacklisted Areas"
    ]

    def _render_bot_output(code: str):
        _, orig_pname = blogic6.PROCESS_TITLES[code]
        pname = BOT_DISPLAY_NAMES.get(code, orig_pname)
        status = proc_status.get(code, "Pending")
        st.markdown(f"### {pname}")
        if status == "Failed":
            st.warning("Process Failed")
            return

        df = results.get(code, pd.DataFrame())

        # For first 11 bots, always use the total input row count from session state
        total_input_rows = None
        if code in FIRST_11_BOTS:
            total_input_rows = s.get("input_row_count")
        # For other bots, fallback to previous logic
        if total_input_rows is None:
            if df is not None and hasattr(df, 'input_row_count'):
                total_input_rows = df.input_row_count
            elif df is not None and hasattr(df, 'attrs') and 'input_row_count' in getattr(df, 'attrs', {}):
                total_input_rows = df.attrs['input_row_count']
            elif df is not None and len(df) > 0 and 'input_row_count' in df.columns:
                total_input_rows = df['input_row_count'].iloc[0]
            elif df is not None and not df.empty:
                total_input_rows = df.index.max() + 1

        total_exceptions = len(df) if df is not None else 0

        st.markdown(
            f"<div style='font-size:1.05rem;font-weight:600;margin-bottom:0.2em;'>"
            f"Total Input Rows: <span style='color:#2563eb'>{total_input_rows if total_input_rows is not None else 'N/A'}</span> &nbsp; | &nbsp; "
            f"Total Exceptions Found: <span style='color:#dc2626'>{total_exceptions}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        if df is None or df.empty:
            st.info("No issues found.")
        else:
            st.dataframe(df, use_container_width=True)

    with tabs[1]:
        st.subheader("Output")
        codes = [c for c in blogic6.PROCESS_TITLES.keys() if not selected_set or c in selected_set]
        if sel_mode == "bot" and sel_value:
            codes = [sel_value]
        for code in codes:
            _render_bot_output(code)

    # ---------- Report ----------
    with tabs[2]:
        st.subheader("Report")

        summary_for_report = _build_enriched_summary_bank(
            proc_status, results, only_codes=(selected_set if selected_set else None)
        )
        if sel_mode == "bot" and sel_value:
            summary_for_report = summary_for_report[summary_for_report["_code"] == sel_value]

        if summary_for_report.empty:
            st.info("No report data available for current selection.")
        else:
            for _, row in summary_for_report.iterrows():
                pname = row["Bot"]
                status = row["Status"]
                cnt = int(row["Issues Found"])
                if status == "Failed":
                    st.markdown(f"- **{pname}** — Process Failed")
                else:
                    st.markdown(f"- **{pname}** — {'No issues found' if cnt == 0 else f'Issues Found: {cnt}'}")

            with st.expander("📊 Full Report Data"):
                st.dataframe(
                    summary_for_report.drop(columns=["_code"]).reset_index(drop=True),
                    use_container_width=True
                )

        excel_bytes = _build_detailed_report_excel_bank(
            proc_status=proc_status,
            results=results,
            only_codes=selected_set if selected_set else None,
        )
        st.download_button(
            "Download Detailed Report (Excel)",
            data=excel_bytes,
            file_name="banking_detailed_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_detailed_report_excel_bank",
        )

    # ---------- Data Extraction Tab ----------
    with tabs[3]:
        st.subheader("📄 Data Extraction")
        
        # Check if PDFs are uploaded
        uploaded_pdfs = s.get("u_pdfs_cached", [])
        
        if not uploaded_pdfs:
            st.info("📄 No PDF files uploaded. Please go to Banking Home and upload PDF files in the 'PDF Data Extraction' tab.")
            return
        
        # Show uploaded files info
        st.markdown("### Uploaded PDF Files")
        for i, pdf_file in enumerate(uploaded_pdfs, 1):
            st.markdown(f"{i}. **{pdf_file.name}** ({pdf_file.size / 1024:.1f} KB)")
        
        # Import PDF extraction functionality
        try:
            from pdf_extraction import run_pdf_extraction, check_pdf_extraction_requirements
        except ImportError:
            st.error("❌ PDF extraction module not found. Please ensure pdf_extraction.py is in the project directory.")
            return
        
        # Check requirements
        missing_reqs = check_pdf_extraction_requirements()
        if missing_reqs:
            st.error(f"❌ Missing requirements: {', '.join(missing_reqs)}")
            st.info("Please install the required packages and ensure your .env file contains the COHERE_API_KEY.")
            return
        
        # Initialize processing state
        if "pdf_extraction_results" not in s:
            s["pdf_extraction_results"] = None
        if "pdf_processing_progress" not in s:
            s["pdf_processing_progress"] = 0
        if "pdf_processing_status" not in s:
            s["pdf_processing_status"] = "ready"
        
        # Check if processing should start
        if s.get("pdf_processing_started", False) and s["pdf_processing_status"] == "processing":
            # Show processing status
            st.markdown("### 🔄 Processing Status")
            
            # Create UI references for progress tracking
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            ui_refs = {
                "data_extraction": {
                    "status": status_placeholder,
                    "prog": progress_placeholder
                }
            }
            
            status_placeholder.info("🔄 Processing PDF files...")
            progress_bar = progress_placeholder.progress(0)
            
            # Process PDFs
            try:
                results = run_pdf_extraction(uploaded_pdfs, ui_refs)
                s["pdf_extraction_results"] = results
                s["pdf_processing_status"] = "completed"
                s["pdf_processing_progress"] = 100
                
                status_placeholder.success("✅ PDF processing completed successfully!")
                progress_bar.progress(100)
                
            except Exception as e:
                s["pdf_processing_status"] = "failed"
                status_placeholder.error(f"❌ PDF processing failed: {str(e)}")
                return
        
        # Show results if available
        if s["pdf_processing_status"] == "completed" and s["pdf_extraction_results"]:
            st.markdown("---")
            st.markdown("### 📊 Extraction Results")
            
            results = s["pdf_extraction_results"]
            
            if results and "consolidated_data" in results:
                df = results["consolidated_data"]
                
                # Normalize the S.No column
                df = normalize_sno(df)
                
                # Add standard values column if not present
                if "Standard Values" not in df.columns:
                    standard_values = [
                        "Project No/ID",  # Project Number
                        ">=Rs. 50 Lakh",  # Loan amount
                        "Renewable Energy/Manufacturing",  # Project Type & Sector
                        "A/AA/B/BB or equivalent",  # Grade
                        "Market Rate %",  # Interest
                        "Total Investment Amount",  # Project Cost
                        ">=25% of Project Cost",  # Promotor Contribution
                        ">=25%",  # Minimum promoter contribution
                        "<=75:25 (Debt:Equity)",  # Debt Equity Ratio
                        ">=1.2",  # Average DSCR (new clients)
                        ">=1.25",  # Average DSCR requirement
                        ">=1.25",  # Average Asset Coverage ratio
                        "<=3% of Net Worth",  # Contingent Liability
                        "6-18 months from COD/SCOD"  # Moratorium/grace period
                    ]
                    
                    # Insert Standard Values column after Particulars
                    particulars_idx = list(df.columns).index("Particulars")
                    df.insert(particulars_idx + 1, "Standard Values", standard_values[:len(df)])
                
                # Insert minimum loan row
                def insert_minimum_loan_row(df):
                    idx = df[df["Particulars"].str.contains("morator", case=False, na=False)].index
                    insert_at = idx[0] if len(idx) > 0 else len(df)
                    new_row = {col: "" for col in df.columns}
                    new_row["Particulars"] = "The minimum loan eligibility from IREDA will be Rs. 50 Lakh unless specified otherwise"
                    new_row["Standard Values"] = "Loan Amount should be >=50 lakh"
                    new_row["S.No"] = insert_at + 1
                    
                    df1 = df.iloc[:insert_at].copy()
                    df2 = df.iloc[insert_at:].copy()
                    new_df = pd.concat([df1, pd.DataFrame([new_row]), df2], ignore_index=True)
                    new_df["S.No"] = range(1, len(new_df) + 1)
                    return new_df
                
                df = insert_minimum_loan_row(df)
                
                # Run compliance checks
                check_results = run_all_checks(df)
                
                # Apply highlighting
                styled_df = apply_highlighting(df, check_results)
                
                # Display the table
                st.markdown("#### Extracted Data with Compliance Checking")
                st.dataframe(styled_df, use_container_width=True)
                
                # Show compliance summary
                if check_results:
                    st.markdown("#### 📋 Compliance Summary")
                    total_checks = sum(len(row_checks) for row_checks in check_results.values())
                    passed_checks = sum(sum(row_checks.values()) for row_checks in check_results.values())
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Checks", total_checks)
                    with col2:
                        st.metric("Passed", passed_checks, delta=None)
                    with col3:
                        st.metric("Failed", total_checks - passed_checks, delta=None)
                
                # Fee calculation
                st.markdown("#### 💰 Fee Calculation")
                fee_df = build_fee_table(df)
                if not fee_df.empty:
                    st.dataframe(fee_df, use_container_width=True)
                else:
                    st.info("No fee calculation data available.")
                
                # Download button
                st.markdown("#### 📥 Download Results")
                
                # Create Excel file for download
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="Extracted Data", index=False)
                    if not fee_df.empty:
                        fee_df.to_excel(writer, sheet_name="Fee Calculation", index=False)
                
                st.download_button(
                    "📥 Download Extraction Results (Excel)",
                    data=output.getvalue(),
                    file_name="pdf_extraction_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_pdf_extraction_results"
                )
            else:
                st.warning("No data extracted from the uploaded PDFs.")
        
        elif s["pdf_processing_status"] == "ready":
            st.info("📄 PDF files are ready for processing. Processing will start automatically when you navigate to this tab.")
            if st.button("🔄 Start Processing Now", key="start_pdf_processing"):
                s["pdf_processing_started"] = True
                s["pdf_processing_status"] = "processing"
                st.rerun()
        
        elif s["pdf_processing_status"] == "failed":
            st.error("❌ PDF processing failed. Please try uploading the files again.")
            if st.button("🔄 Retry Processing", key="retry_pdf_processing"):
                s["pdf_processing_started"] = True
                s["pdf_processing_status"] = "processing"
                s.pop("pdf_extraction_results", None)
                st.rerun()

    st.markdown("---")
    if st.button("⟵ Back to Processing", key="back_to_processing_btn"):
        s.page = "b6"; st.rerun()
