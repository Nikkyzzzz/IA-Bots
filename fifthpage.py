import streamlit as st
import streamlit_antd_components as sac
import pandas as pd
import altair as alt
from io import BytesIO
import time
import datetime as dt
import numpy as np
from openai import OpenAI

import logic6

LEFT_LOGO_PATH = "logo.png"
CATEGORIES_ORDER = ("P2P", "O2C", "H2R")

def codes_for_category(category: str):
    return [code for code, (cat, _) in logic6.PROCESS_TITLES.items() if cat == category]

def parse_selection(sel: str):
    if sel in (None, "", "All"):
        return ("all", None)
    if sel == "All (P2P)":
        return ("category", "P2P")
    if sel == "All (O2C)":
        return ("category", "O2C")
    if sel == "All (H2R)":
        return ("category", "H2R")
    if sel in CATEGORIES_ORDER:
        return ("category", sel)
    if sel in logic6.BOT_NAMES:
        return ("bot", logic6.PROC_BY_NAME.get(sel))
    return ("all", None)

def _dynamic_menu_items(available_categories: list[str]):
    items = [sac.MenuItem('All', icon='grid')]
    if "P2P" in available_categories:
        items.append(
            sac.MenuItem('P2P', icon='layers', children=[
                sac.MenuItem('All (P2P)'),

                # Existing P2P bots
                sac.MenuItem('Validate Vendor KYC'),
                sac.MenuItem('PO-GRN-Invoice Match'),
                sac.MenuItem('Post-Invoice POs'),
                sac.MenuItem('Split Orders'),
                sac.MenuItem('Duplicate Vendors'),
                # sac.MenuItem('PO Approval Bypass'),   # already added

                # 🔵 NEW 5 P2P BOTS (names must match logic6.BOT_NAMES)
                sac.MenuItem('Excessive Emergency Purchases'),   # P2P10
                sac.MenuItem('Invoice vs GRN Date Gap'),                    # P2P13
                sac.MenuItem('GSTIN Validation'),                            # P2P18 (GST bot)
                sac.MenuItem('Invoices to Inactive Vendors'),                # P2P17
                sac.MenuItem('Round Sum Invoices (Rate/Qty Missing)'),       # P2P20
            ])
        )
    if "O2C" in available_categories:
        items.append(
            sac.MenuItem('O2C', icon='truck', children=[
                sac.MenuItem('All (O2C)'),
                sac.MenuItem('Overdue Delivery'),
                sac.MenuItem('Dispatch Without Invoice'),
                sac.MenuItem('Missing Customer Master Data'),
            ])
        )
    if "H2R" in available_categories:
        items.append(
            sac.MenuItem('H2R', icon='user-check', children=[
                sac.MenuItem('All (H2R)'),
                sac.MenuItem('Ghost employee detection'),
                sac.MenuItem('Inactive Employees In Payroll'),
            ])
        )
    return items


def _codes_for_present_categories(cats_present: list[str]):
    codes = []
    for cat in cats_present:
        codes.extend(codes_for_category(cat))
    return codes

LOGIC_DESCRIPTIONS = {
    "P2P1": "Identified rows where vendor fields (PAN, GST, Bank Account) were missing and marked them in an Exception Noted column.",
    "P2P2": "Detected quantity and amount mismatches between PO, GRN, and Invoice records and computed financial impact per PO.",
    "P2P3": "Extracted POs where PO Date was later than Invoice Date, marking them as invalid.",
    "P2P4": "Flagged cases where a vendor had multiple POs for the same item on the same date with combined invoice value above the threshold.",
    "P2P5": "Detected duplicate vendor records by PAN, GST, Name, or Bank Account and listed row pairs with exceptions.",
    # Updated 17-nov start
    # "P2P8": "Checked whether purchase orders were created or approved beyond the Authority Matrix limits or by unauthorised employees.",
    # Updated 17-nov completed    

    # 🔵 NEW P2P BOTS (5 bots you shared)
    # p2p 10 df4 – Emergency in percentage %
    "P2P10": "Measured what percentage of total POs were marked as Emergency and flagged when the emergency share breached the configured threshold.",
    # p2p 13 df4 – threshold ±10 days (invoice date vs GRN date)
    "P2P13": "Compared Invoice Date and GRN Date, calculated the days gap, and flagged invoices that were too early or too late beyond the allowed day threshold.",
    # GSTIN validation (state codes 01–28)
    "P2P18": "Validated GSTIN formats against structure rules and state codes 01–28 and flagged vendors with invalid GST registration numbers.",
    # p2p 17 df4, df5 – invoice issued to inactive vendors
    "P2P17": "Identified invoices raised to vendors whose status was inactive and marked such cases as exceptions.",
    # p2p 20 df4 – round sum / missing rate or qty
    "P2P20": "Flagged round-sum invoices where rate and/or quantity were missing or inconsistent with the invoice amount.",
    # ---- end new P2P bots ----

    # 🟦 P2P new bots
    "P2P11": "Summed invoice amount per vendor, calculated each vendor’s percentage of total spend, and flagged vendors exceeding the concentration threshold.",
    "P2P14": "Compared PO quantity with GRN quantity row-wise and flagged over-receipt, shortage, or value errors for quantity fields.",
    "P2P15": "Compared invoice date and payment date against payment terms to classify each payment as early, late, or pre-payment and flagged non-adherences.",
    "P2P16": "Detected duplicate invoices by vendor, invoice date, and invoice amount, marking all matching rows as potential duplicates.",

    # 🟧 O2C new bots
    "O2C24": "Identified invoices where SO number was missing but invoice number existed and tagged them as sales without SO reference.",
    "O2C29": "Counted invoices per sales order and flagged orders with more than one invoice as multiple-invoice cases.",
    "O2C32": "Flagged invoices with zero or missing invoice amount as potentially erroneous or incomplete billing.",
    "O2C39": "Measured the share of low-value sales and flagged small-value invoices when the overall percentage crossed the configured risk threshold.",

    # 🟩 H2R new bot
    "H2R44": "Detected duplicate employees based on repeated bank accounts and/or PAN numbers and grouped them by duplicate keys.",

    "O2C3": "Identified customers with missing GST, PAN, or Credit Limit and flagged them.",
    "O2C1": "Flagged sales orders where delivery was delayed beyond the allowed threshold.",
    "O2C2": "Detected cases where goods were dispatched but no invoice was issued.",
    "H2R1": "Identified employees with attendance records but no entry in the Employee Master, i.e., ghost employees.",
    "H2R2": "Flagged employees who continued to appear as present after their recorded exit date.",
}


DATA_USED = {
    "P2P1": "Vendor Master",
    "P2P2": "P2P Sample",
    "P2P3": "P2P Sample",
    "P2P4": "P2P Sample",
    "P2P5": "Vendor Master",
    # Updated 17-nov start
    # "P2P8": "P2P Sample, Authority Matrix, Employee Master",
    # Updated 17-nov completed

    # 🔵 NEW P2P BOTS
    # Emergency PO % is from P2P sheet
    "P2P10": "P2P Sample",
    # Invoice vs GRN Date gap -> P2P sheet
    "P2P13": "P2P Sample",
    # GSTIN Validation on Vendor Master (GST_No)
    "P2P18": "Vendor Master",
    # Invoices to inactive vendors -> P2P + Vendor Master
    "P2P17": "P2P Sample, Vendor Master",
    # Round sum invoices -> P2P sheet
    "P2P20": "P2P Sample",
    # ---- end new P2P bots ----

    # 🟦 P2P
    "P2P11": "P2P Sample",
    "P2P14": "P2P Sample",
    "P2P15": "P2P Sample",
    "P2P16": "P2P Sample",

    # 🟧 O2C
    "O2C24": "O2C Sample",
    "O2C29": "O2C Sample",
    "O2C32": "O2C Sample",
    "O2C39": "O2C Sample",

    # 🟩 H2R
    "H2R44": "Employee Master",

    "O2C1": "O2C Sample",
    "O2C2": "O2C Sample",
    "O2C3": "Customer Master",
    "H2R1": "Employee Master, Attendance Register",
    "H2R2": "Employee Master, Attendance Register",
}


def _build_enriched_summary(proc_status: dict, results: dict, cats_present: list[str]) -> pd.DataFrame:
    rows = []
    for code, (cat, bot_name) in logic6.PROCESS_TITLES.items():
        if cat not in cats_present:
            continue
        cnt = logic6.issues_count_for(code, results.get(code))
        rows.append({
            "Bot": bot_name,
            "Category": cat,
            "Data Used": DATA_USED.get(code, ""),
            "Logic Description": LOGIC_DESCRIPTIONS.get(code, ""),
            "Issues Found": int(cnt),
            "Status": proc_status.get(code, "Pending"),
            "_code": code,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["Bot", "Category", "Data Used", "Logic Description", "Issues Found", "Status", "_code"]]
    return df

def _build_detailed_report_excel(
    cats_present: list[str],
    proc_status: dict,
    results: dict,
    vendor_raw: pd.DataFrame | None,
    p2p_raw: pd.DataFrame | None,
    emp_raw: pd.DataFrame | None,
    file_bytes: bytes | None
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df = logic6.build_summary_df(proc_status, results)
        if not summary_df.empty:
            summary_df = summary_df[summary_df["Category"].isin(cats_present)]
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for cat in cats_present:
            codes = codes_for_category(cat)
            for code in codes:
                df = results.get(code)
                if df is not None and not df.empty:
                    _, pname = logic6.PROCESS_TITLES[code]
                    sheet_name = f"{cat}_{pname[:20]}"
                    total_df = pd.DataFrame([{"Total Records": len(df)}])
                    total_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)

                    if code == "P2P1" and vendor_raw is not None and p2p_raw is not None:
                        logic6.anomalies_by_creator(vendor_raw).to_excel(writer, sheet_name="P2P1_Anomalies", index=False)
                        logic6.merge_missing_with_duplicates(vendor_raw, p2p_raw).to_excel(writer, sheet_name="P2P1_MissingDup", index=False)

                    if code == "P2P2" and results.get("P2P2") is not None and p2p_raw is not None and emp_raw is not None:
                        item_sum, dept_sum = logic6.summarize_mismatches(results["P2P2"], p2p_raw, emp_raw)
                        item_sum.to_excel(writer, sheet_name="P2P2_ItemSummary", index=False)
                        dept_sum.to_excel(writer, sheet_name="P2P2_DeptSummary", index=False)
                        try:
                            logic6.calculate_financial_impact_df(results["P2P2"]).to_excel(writer, sheet_name="P2P2_FinImpact", index=False)
                        except Exception:
                            pass

                    if code == "P2P3" and results.get("P2P3") is not None:
                        item_counts, creator_counts = logic6.next_level_analytics(results["P2P3"])
                        item_counts.to_excel(writer, sheet_name="P2P3_ItemIssues", index=False)
                        creator_counts.to_excel(writer, sheet_name="P2P3_CreatorIssues", index=False)
                        logic6.financial_impact(results["P2P3"]).to_excel(writer, sheet_name="P2P3_FinImpact", index=False)

                    if code == "P2P5" and file_bytes is not None:
                        try:
                            detailed = df
                            fy_sum, fy_detail = logic6.vendor_year_threshold_alerts(
                                detailed,
                                BytesIO(file_bytes),
                                sheet_name="P2P_Sample (Bots 1-20)",
                                threshold=50_000
                            )
                            day_sum, day_detail = logic6.vendor_daily_threshold_alerts(
                                detailed,
                                BytesIO(file_bytes),
                                sheet_name="P2P_Sample (Bots 1-20)",
                                threshold=10_000
                            )
                            fy_sum.to_excel(writer, sheet_name="P2P5_FY_Summary", index=False)
                            if fy_detail is not None and not fy_detail.empty:
                                fy_detail.to_excel(writer, sheet_name="P2P5_FY_Detail", index=False)
                            day_sum.to_excel(writer, sheet_name="P2P5_Daily_Summary", index=False)
                            if day_detail is not None and not day_detail.empty:
                                day_detail.to_excel(writer, sheet_name="P2P5_Daily_Detail", index=False)
                        except Exception:
                            pass

    output.seek(0)
    return output.getvalue()

def _ensure_keys(keys: list[str]):
    s = st.session_state
    for k in keys:
        if k not in s:
            s[k] = None

def _render_persisted_md(key: str):
    val = st.session_state.get(key)
    if isinstance(val, str) and val.strip():
        st.markdown(val)

def _progress_task(task_name: str, stages: list[tuple[int, str]]):
    pg = st.progress(0, text=f"{task_name}...")
    last = 0
    for pct, label in stages:
        pct = max(min(pct, 100), 0)
        if pct > last:
            pg.progress(pct, text=label)
            last = pct
            time.sleep(0.1)
    if last < 100:
        pg.progress(100, text=f"{task_name} complete")

def _df_preview(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 14) -> str:
    if df is None or df.empty:
        return "(no rows)"
    cols = list(df.columns)[:max_cols]
    slim = df[cols].head(max_rows)
    return slim.to_csv(index=False)

def _df_schema(df: pd.DataFrame, max_cols: int = 24) -> str:
    if df is None or df.empty:
        return "(no columns)"
    cols = list(df.columns)[:max_cols]
    dtypes = [str(df[c].dtype) for c in cols]
    return "\n".join(f"- {c}: {t}" for c, t in zip(cols, dtypes))

def _logic_lines_for_category(cat: str) -> list[str]:
    return [
        f"- {LOGIC_DESCRIPTIONS[code]}"
        for code, (c, _) in logic6.PROCESS_TITLES.items()
        if c == cat and code in LOGIC_DESCRIPTIONS
    ]

def _sample_blocks_for_category(
    cat: str,
    vendor_raw: pd.DataFrame | None,
    p2p_raw: pd.DataFrame | None,
    o2c_raw: pd.DataFrame | None,
    cust_raw: pd.DataFrame | None,
    emp_raw: pd.DataFrame | None,
    att_raw: pd.DataFrame | None,
    include_rows: bool = True,
) -> str:
    blocks = []
    if cat == "P2P":
        if vendor_raw is not None:
            blocks.append("• Vendor Master — Schema:\n" + _df_schema(vendor_raw))
            if include_rows: blocks.append("Vendor Master — Sample (CSV):\n" + _df_preview(vendor_raw))
        if p2p_raw is not None:
            blocks.append("• P2P Sample — Schema:\n" + _df_schema(p2p_raw))
            if include_rows: blocks.append("P2P Sample — Sample (CSV):\n" + _df_preview(p2p_raw))
    elif cat == "O2C":
        if o2c_raw is not None:
            blocks.append("• O2C Sample — Schema:\n" + _df_schema(o2c_raw))
            if include_rows: blocks.append("O2C Sample — Sample (CSV):\n" + _df_preview(o2c_raw))
        if cust_raw is not None:
            blocks.append("• Customer Master — Schema:\n" + _df_schema(cust_raw))
            if include_rows: blocks.append("Customer Master — Sample (CSV):\n" + _df_preview(cust_raw))
    elif cat == "H2R":
        if emp_raw is not None:
            blocks.append("• Employee Master — Schema:\n" + _df_schema(emp_raw))
            if include_rows: blocks.append("Employee Master — Sample (CSV):\n" + _df_preview(emp_raw))
        if att_raw is not None:
            blocks.append("• Attendance Register — Schema:\n" + _df_schema(att_raw))
            if include_rows: blocks.append("Attendance Register — Sample (CSV):\n" + _df_preview(att_raw))
    return "\n\n".join(blocks).strip() or "(no sample available)"

# --------- PROMPTS tuned for 6–7 bullets per heading ----------
LAYMAN_BLOCK_SPEC = (
    "Append a 'Quick Read' block with EXACTLY these headings and format.\n"
    "For EACH heading, produce AT LEAST 6–7 bullet points.\n"
    "- Each bullet must be 2–3 sentences with this micro-structure: Context — Numeric Evidence — Business Impact.\n"
    "- Always quantify with numbers (counts, percentages, ₹ amounts, thresholds, days, variances). Avoid vague phrases.\n"
    "- Do NOT fabricate values. If a field is missing, explicitly state the limitation.\n"
    "Headings (use these exact titles):\n"
    "Observation\n"
    "Risk (with root cause & consequence)\n"
    "Recommendation for Improvement (specific, actionable)\n"
    "Risk Category with Rationale\n"
)

def _deep_prompt_for_categories(
    cats: list[str],
    vendor_raw: pd.DataFrame | None,
    p2p_raw: pd.DataFrame | None,
    o2c_raw: pd.DataFrame | None,
    cust_raw: pd.DataFrame | None,
    emp_raw: pd.DataFrame | None,
    att_raw: pd.DataFrame | None,
    include_rows: bool = True,
) -> str:
    header = (
        "You are an **Audit Analytics Assistant**. Generate a deep, elaborated analysis for the selected category(ies).\n"
        "- Use ONLY the provided schemas/samples; if a required column is missing, say so; NEVER fabricate.\n"
        "- For EACH logic, write 6–7 bullets. Each bullet MUST:\n"
        "  • Follow: Context — Numeric Evidence — Business Impact\n"
        "  • Include quantitative details (e.g., '10/150 POs = 6.7%', '₹3.2L variance', 'TAT +4 days vs SLA 2 days').\n"
        "  • Be 2–3 sentences, concise and decision-oriented.\n"
        "- Structure output with clear headings & subheadings per category and per logic; prefer bullet points.\n"
        "- End EACH category with a one-line **Summary Insight** linking issues to strategic risks/compliance exposure.\n\n"
        + LAYMAN_BLOCK_SPEC
    )
    sections = []
    for cat in cats:
        logic_lines = _logic_lines_for_category(cat)
        logic_text = "\n".join(logic_lines) if logic_lines else "- No explicit logic registered."
        samples = _sample_blocks_for_category(
            cat, vendor_raw, p2p_raw, o2c_raw, cust_raw, emp_raw, att_raw, include_rows=include_rows
        )
        sec = [
            f"## Category: {cat}",
            "### Logics to Cover:",
            logic_text,
            "### Schemas & Small Samples:",
            samples,
            "### Analysis:"
        ]
        sections.append("\n".join(sec))
    return header + "\n\n" + "\n\n".join(sections) + "\n\nBegin the analysis now."

def render_fifth():
    s = st.session_state
    st.set_page_config(layout="wide")

    top_l, _, _ = st.columns([1, 6, 1])
    with top_l:
        st.image(LEFT_LOGO_PATH, width=160)

    st.markdown("""
    <style>
    .ant-menu-dark { background:#1f2937 !important; }
    .ant-menu-item-selected,.ant-menu-submenu-title:hover { background:#22c55e !important; }
    .ant-menu-title-content,.ant-menu-dark .ant-menu-item,.ant-menu-dark .ant-menu-submenu-title { color:#e5e7eb !important; }
    </style>
    """, unsafe_allow_html=True)

    if not s.get("processing_done"):
        st.warning("No processed results found. Please run processing first.")
        if st.button("⟵ Back to Processing", key="go_processing_btn"):
            s.page = "processpage"; st.rerun()
        return
        # --- AI visibility toggle coming from first page ---
    show_ai = bool(s.get("use_ai", False))

    # Optional notice if AI disabled
    if not show_ai:
        st.info(
            "AI features are disabled. To view **Deep Analysis**, **AI Generated Reports**, "
            "**Numerical Data Analysis**, and **Control Descriptions**, enable the AI option on the first page."
        )


    _ensure_keys([
        "ai_report_P2P1_done", "ai_report_P2P1_md",
        "num_analysis_P2P1_done", "num_analysis_P2P1_md",
        "ctrl_desc_P2P1_done", "ctrl_desc_P2P1_md",
        "ai_report_P2P2_done", "ai_report_P2P2_md",
        "num_analysis_P2P2_done", "num_analysis_P2P2_md",
        "ctrl_desc_P2P2_done", "ctrl_desc_P2P2_md",
        "ai_report_P2P3_done", "ai_report_P2P3_md",
        "num_analysis_P2P3_done", "num_analysis_P2P3_md",
        "ctrl_desc_P2P3_done", "ctrl_desc_P2P3_md",
        "ai_report_P2P4_done", "ai_report_P2P4_md",
        "num_analysis_P2P4_done", "num_analysis_P2P4_md",
        "ctrl_desc_P2P4_done", "ctrl_desc_P2P4_md",
        "ai_report_P2P5_done", "ai_report_P2P5_md",
        "num_analysis_P2P5_done", "num_analysis_P2P5_md",
        "ctrl_desc_P2P5_done", "ctrl_desc_P2P5_md",
    ])

    results     = s.get("results", {})
    proc_status = s.get("proc_status", {})
    statuses    = s.get("statuses", {})

    vendor_raw = (s.raw_dfs.get("VENDOR_RAW") if s.get("raw_dfs") else None)
    p2p_raw    = (s.raw_dfs.get("P2P_RAW")    if s.get("raw_dfs") else None)
    emp_raw    = (s.raw_dfs.get("EMP_RAW")    if s.get("raw_dfs") else None)
    o2c_raw    = (s.raw_dfs.get("O2C_RAW")    if s.get("raw_dfs") else None)
    cust_raw   = (s.raw_dfs.get("CUST_RAW")   if s.get("raw_dfs") else None)
    att_raw    = (s.raw_dfs.get("ATT_RAW")    if s.get("raw_dfs") else None)

    cats_present = [c for c in CATEGORIES_ORDER if c in (s.sheet_mapping_pairs or {})]
    if not cats_present:
        cats_from_proc = set()
        for code, (cat, _) in logic6.PROCESS_TITLES.items():
            if proc_status.get(code) in ("Complete", "Failed"):
                cats_from_proc.add(cat)
        cats_present = [c for c in CATEGORIES_ORDER if c in cats_from_proc] or list(CATEGORIES_ORDER)

    with st.sidebar:
        choice = sac.menu(
            items=_dynamic_menu_items(cats_present),
            open_all=True,
            indent=16
        )

    sel_mode, sel_value = parse_selection(choice)
    tabs = st.tabs(["Analysis", "Output", "Report"])

    with tabs[0]:
        # st.markdown("**Analysis**")
        summary = _build_enriched_summary(proc_status, results, cats_present)

        if sel_mode == "bot":
            bot_name = next((name for name, code in logic6.PROC_BY_NAME.items() if code == sel_value), None)
            if bot_name:
                summary = summary[summary["Bot"] == bot_name]
        elif sel_mode == "category":
            summary = summary[summary["Category"] == sel_value]

        summary_to_show = summary.drop(columns=["_code"]) if "_code" in summary.columns else summary
        
        def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
            """
            Make a DataFrame safe for Streamlit's Arrow serialization.
            - Normalizes datetime-like columns
            - Decodes bytes
            - Stringifies mixed-type object columns
            """
            if df is None or df.empty:
                return df

            df = df.copy()

            def is_dt_like_series(s: pd.Series) -> bool:
                # object column containing any datetime/date
                if s.dtype == "object":
                    try:
                        return s.map(lambda x: isinstance(x, (pd.Timestamp, dt.datetime, dt.date)) if x is not None else False).any()
                    except Exception:
                        return False
                # already datetime64/period
                return np.issubdtype(s.dtype, np.datetime64)

            for col in df.columns:
                s = df[col]

                # 1) If column is datetime-like (or contains datetimes), coerce to pandas datetime64
                if is_dt_like_series(s):
                    df[col] = pd.to_datetime(s, errors="coerce")

                # 2) If still object & contains bytes, decode to UTF-8
                if df[col].dtype == "object":
                    try:
                        if s.map(lambda x: isinstance(x, (bytes, bytearray))).any():
                            df[col] = s.map(lambda b: b.decode("utf-8", "replace") if isinstance(b, (bytes, bytearray)) else b)
                    except Exception:
                        pass

                # 3) If still object & mixed types remain, stringify as a last resort
                if df[col].dtype == "object":
                    try:
                        unique_types = s.map(lambda x: type(x).__name__ if x is not None else "None").unique()
                        if len(unique_types) > 1:
                            df[col] = s.astype(str)
                    except Exception:
                        df[col] = s.astype(str)

            return df

        st.subheader("Summary Overview")
        st.dataframe(summary_to_show.reset_index(drop=True), use_container_width=True)

        st.subheader("Issues Per Bot")
        if not summary_to_show.empty:
            chart = (
                alt.Chart(summary_to_show)
                .mark_bar()
                .encode(
                    # x=alt.X('Bot:N', sort=None, title='Bots', axis=alt.Axis(labelAngle=-35)),
                    # y=alt.Y('Issues Found:Q', title='Issues Found', stack='zero'),
                    x=alt.X('Issues Found:Q', title='Issues Found', stack='zero'),
                    y=alt.Y('Bot:N', sort=None, title='Bots', axis=alt.Axis(labelAngle=0)),
                    color='Category:N',
                    tooltip=['Bot', 'Category', 'Data Used', 'Logic Description', 'Issues Found', 'Status']
                )
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data to plot for the current selection.")

        if show_ai:
            if st.button("Deep Analysis"):
                cats = []
                if sel_mode == "category" and sel_value in ("P2P", "O2C", "H2R"):
                    cats = [sel_value]
                elif sel_mode == "all":
                    cats = cats_present[:]
                elif sel_mode == "bot" and sel_value:
                    cat_for_bot, _ = logic6.PROCESS_TITLES.get(sel_value, (None, None))
                    if cat_for_bot:
                        cats = [cat_for_bot]
                if not cats:
                    cats = cats_present[:] if cats_present else ["P2P", "O2C", "H2R"]

                _progress_task("Generating deep analysis", [
                    (15, "Preparing category briefs..."),
                    (55, "Compiling sample data..."),
                    (85, "Calling AI model..."),
                    (100, "Formatting results...")
                ])
                try:
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    prompt = _deep_prompt_for_categories(
                        cats=cats,
                        vendor_raw=vendor_raw,
                        p2p_raw=p2p_raw,
                        o2c_raw=o2c_raw,
                        cust_raw=cust_raw,
                        emp_raw=emp_raw,
                        att_raw=att_raw,
                        include_rows=True,
                    )
                    if len(prompt) > 18000:
                        prompt = _deep_prompt_for_categories(
                            cats=cats,
                            vendor_raw=vendor_raw,
                            p2p_raw=p2p_raw,
                            o2c_raw=o2c_raw,
                            cust_raw=cust_raw,
                            emp_raw=emp_raw,
                            att_raw=att_raw,
                            include_rows=False,
                        )
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=2000
                    )
                    st.markdown(response.choices[0].message.content.strip())
                    st.success("Deep Analysis generated successfully.")
                except Exception as e:
                    st.error(f"Deep Analysis failed: {str(e)}")

    def _gen_ai_report_md(df_for_llm: pd.DataFrame) -> str:
        df_text = df_for_llm.head(120).to_string()
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = (
            "You are an analytics assistant. Using ONLY the data sample below, "
            "produce a professional summary of issues, anomalies, compliance insights, root causes, and business impact.\n"
            "- Do NOT invent columns or values; if needed fields are missing, say so succinctly and continue.\n"
            "- Write 6–7 bullets, each 2–3 sentences, strictly following: Context — Numeric Evidence — Business Impact.\n"
            "- Always quantify (counts, %, ₹ amounts, thresholds/SLA deltas, date gaps).\n"
            "- After the bullets, add a 'Top 3 Critical Anomalies' list ordered by estimated impact and likelihood.\n\n"
            f"DATA SAMPLE (first 120 rows, plain text):\n{df_text}\n\n"
            "OUTPUT FORMAT:\n"
            "1) Professional Summary (6–7 bullets)\n"
            + LAYMAN_BLOCK_SPEC
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()

    def _gen_numerical_md(df_for_llm: pd.DataFrame) -> str:
        df_text = df_for_llm.head(120).to_string()
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = (
            "You are an analytics assistant. Using ONLY the numerical fields visible in the sample below, "
            "summarize distributions, outliers, variance, missingness, apparent correlations, and threshold breaches.\n"
            "- If numeric fields are absent, state that clearly. Do NOT fabricate numbers.\n"
            "- Write 6–7 bullets, each 2–3 sentences, following: Context — Numeric Evidence — Business Impact.\n"
            "- Quantify with descriptive statistics or counts from the sample (e.g., mean/median deltas, % missing, IQR/±3σ hints, min–max ranges).\n"
            "- Close with a short 'What to Investigate Next' bullet (one line) identifying 2–3 priority checks.\n\n"
            f"DATA SAMPLE (first 120 rows, plain text):\n{df_text}\n\n"
            "OUTPUT FORMAT:\n"
            "1) Numeric Insights (6–7 bullets)\n"
            + LAYMAN_BLOCK_SPEC
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    def _gen_chat_md(user_message: str, df_for_llm: pd.DataFrame) -> str:
        # Show columns and a sample of the data, and instruct the model to only use this data
        if df_for_llm is None or df_for_llm.empty:
            return "No data available to answer your question."
        df_text = df_for_llm.head(120).to_string()
        columns = ', '.join([str(c) for c in df_for_llm.columns])
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = (
            "You are an expert analytics assistant. Answer the user's question using ONLY the data sample below.\n"
            "- Do NOT use any external knowledge or assumptions.\n"
            "- If the answer is not present in the data, say so clearly.\n"
            "- Use the column names and values exactly as shown.\n"
            "- If the user asks about missing values (e.g., NaN, null), count them using the data sample.\n"
            "- If the question cannot be answered, reply: 'Cannot answer based on the data above.'\n"
            f"\nUSER QUESTION: {user_message}\n"
            f"\nDATA COLUMNS: {columns}\n"
            f"\nDATA SAMPLE (first 120 rows):\n{df_text}\n"
            "\nYour response must include:\n"
            "- A direct answer to the user's question, using only the data above.\n"
            "- Numeric evidence or counts from the data sample.\n"
            "- If the data is insufficient, state the limitation.\n"
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

    def _gen_control_md(csv_df: pd.DataFrame) -> str:
        keep_cols = [
            "PO_No", "Vendor_Name",
            "PO_Qty_PO", "PO_Amt_PO",
            "GRN_Qty_Sum", "Invoice_Qty_Sum", "Invoice_Amount_Sum",
            "Exception Noted (Qty)", "Exception Noted (Amt)", "Financial Impact"
        ]
        cols = [c for c in keep_cols if c in csv_df.columns]
        slim = csv_df[cols].head(40) if cols else csv_df.head(40)
        df_text = slim.to_csv(index=False)
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = (
            "You are an audit analytics assistant. Use ONLY the CSV sample below.\n"
            "- Provide detailed bullets under two headings: 'PO after Invoice' and 'Item-wise Count (desc)'.\n"
            "- Each bullet MUST follow: Context — Numeric Evidence — Business Impact (2–3 sentences).\n"
            "- Then propose 6–7 **practical controls** mapped to COSO components (Control Activities, Information & Communication, Monitoring), "
            "and mark each control with: Control Owner (Role), Control Frequency (e.g., per PO / daily / monthly), Control Type (Preventive/Detective), Nature (Automated/Manual), and 1–2 Test Steps.\n"
            "- If required columns are missing, call it out explicitly; do NOT fabricate.\n\n"
            f"DATA (CSV sample):\n{df_text}\n\n"
            + LAYMAN_BLOCK_SPEC
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()

    def show_process(code):
        cat, pname = logic6.PROCESS_TITLES[code]
        status = proc_status.get(code, "Pending")
        st.markdown(f"### {cat} — {pname}")
        if status == "Failed":
            st.warning("Process Failed")
            return
        df = results.get(code, pd.DataFrame())
        if df is None or df.empty:
            st.info("No issues found.")
        else:
            st.markdown(f"**Total records:** {len(df)}")
            st.dataframe(_arrow_safe(df), use_container_width=True)

        def _render_llm_panel(scope_code: str, ai_df_builder, num_df_builder, ctrl_df_builder):
            ai_done_k = f"ai_report_{scope_code}_done"
            ai_md_k   = f"ai_report_{scope_code}_md"
            num_done_k = f"num_analysis_{scope_code}_done"
            num_md_k   = f"num_analysis_{scope_code}_md"
            ctrl_done_k = f"ctrl_desc_{scope_code}_done"
            ctrl_md_k   = f"ctrl_desc_{scope_code}_md"
            if show_ai:
                st.markdown("#### AI Reports")
                _render_persisted_md(ai_md_k)
                ai_disabled = bool(st.session_state.get(ai_done_k))
                if st.button("Generate AI Report", key=f"ai_report_btn_{scope_code}", disabled=ai_disabled):
                    _progress_task("Generating AI report", [
                        (20, "Preparing sample data..."),
                        (65, "Calling AI model..."),
                        (90, "Post-processing output..."),
                        (100, "AI report ready"),
                    ])
                    try:
                        md = _gen_ai_report_md(ai_df_builder())
                        st.session_state[ai_md_k] = md
                        st.session_state[ai_done_k] = True
                        st.markdown(md)
                        st.success("AI report generated.")
                    except Exception as e:
                        st.error(f"AI report generation failed: {str(e)}")
            if show_ai:
                st.markdown("#### Numerical Analysis")
                _render_persisted_md(num_md_k)
                num_disabled = bool(st.session_state.get(num_done_k))
                if st.button("Run Numerical Analysis", key=f"num_btn_{scope_code}", disabled=num_disabled):
                    _progress_task("Running numerical analysis", [
                        (15, "Scanning numeric columns..."),
                        (55, "Computing stats & outliers..."),
                        (85, "Summarizing insights..."),
                        (100, "Numerical analysis ready"),
                    ])
                    try:
                        md = _gen_numerical_md(num_df_builder())
                        st.session_state[num_md_k] = md
                        st.session_state[num_done_k] = True
                        st.markdown(md)
                        st.success("Numerical analysis generated.")
                    except Exception as e:
                        st.error(f"Numerical analysis failed: {str(e)}")
            if show_ai:
                st.markdown("#### Control Description through AI")
                _render_persisted_md(ctrl_md_k)
                ctrl_disabled = bool(st.session_state.get(ctrl_done_k))
                if st.button("Generate Control Description", key=f"ctrl_btn_{scope_code}", disabled=ctrl_disabled):
                    _progress_task("Generating control description", [
                        (20, "Preparing CSV sample..."),
                        (60, "Calling AI model..."),
                        (90, "Drafting control suggestions..."),
                        (100, "Control description ready"),
                    ])
                    try:
                        md = _gen_control_md(ctrl_df_builder())
                        st.session_state[ctrl_md_k] = md
                        st.session_state[ctrl_done_k] = True
                        st.markdown(md)
                        st.success("Control description generated.")
                    except Exception as e:
                        st.error(f"Control Description generation failed: {str(e)}")

        if code == "P2P1":
            sub_tabs = st.tabs([
                "Anomalies by Creator",
                "Missing Vendors × Duplicate Invoices",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Anomalies by Creator**")
                if vendor_raw is not None:
                    a1 = logic6.anomalies_by_creator(vendor_raw)
                    st.session_state['output_df_P2P1_anomalies_by_creator'] = a1
                    st.dataframe(a1 if not a1.empty else pd.DataFrame({"Info":["No anomalies by Creator_ID."]}), use_container_width=True)
                else:
                    st.info("Raw Vendor sheet not available for analysis.")
            with sub_tabs[1]:
                st.markdown("**Missing Vendors × Duplicate Invoices**")
                if vendor_raw is not None and p2p_raw is not None:
                    a2 = logic6.merge_missing_with_duplicates(vendor_raw, p2p_raw)
                    st.session_state['output_df_P2P1_missing_vendors_duplicates'] = a2
                    st.dataframe(a2 if not a2.empty else pd.DataFrame({"Info":["No intersection between missing vendors and duplicate invoices."]}), use_container_width=True)
                else:
                    st.info("Required raw sheets not available for analysis.")
            with sub_tabs[2]:
                df = results.get(code, pd.DataFrame())
                st.session_state['output_df_P2P1_main'] = df
                if df is not None and not df.empty:
                    _render_llm_panel(
                        "P2P1",
                        ai_df_builder=lambda: logic6.find_missing_vendor_fields(df),
                        num_df_builder=lambda: logic6.find_missing_vendor_fields(df),
                        ctrl_df_builder=lambda: (p2p_raw if p2p_raw is not None else df)
                    )
                else:
                    st.warning("Required data not available for AI/Numerical/Control sections.")
            with sub_tabs[3]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                # Use the most recent output table for chatbot context
                df_for_llm = st.session_state.get('output_df_P2P1_main', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.rerun()

        if code == "P2P2":
            sub_tabs = st.tabs([
                "Item & Department Summary",
                "Financial Impact",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Item & Department Summary**")
                if results.get("P2P2") is not None and not results["P2P2"].empty and p2p_raw is not None and emp_raw is not None:
                    item_sum, dept_sum = logic6.summarize_mismatches(results["P2P2"], p2p_raw, emp_raw)
                    st.session_state['output_df_P2P2_item_summary'] = item_sum
                    st.session_state['output_df_P2P2_dept_summary'] = dept_sum
                    st.write("Item-wise Summary")
                    st.dataframe(item_sum if not item_sum.empty else pd.DataFrame({"Info":["No item-wise mismatches."]}), use_container_width=True)
                    st.write("Department-wise Summary")
                    st.dataframe(dept_sum if not dept_sum.empty else pd.DataFrame({"Info":["No department-wise mismatches or missing Department mapping."]}), use_container_width=True)
                else:
                    st.info("Mismatch result or required raw sheets (P2P, Employee Master) not available.")
            with sub_tabs[1]:
                st.markdown("**Financial Impact**")
                if results.get("P2P2") is not None and not results["P2P2"].empty:
                    try:
                        fi = logic6.calculate_financial_impact_df(results["P2P2"])
                        st.session_state['output_df_P2P2_financial_impact'] = fi
                        st.dataframe(fi, use_container_width=True)
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.info("Mismatch result not available.")
            with sub_tabs[2]:
                _render_llm_panel(
                    "P2P2",
                    ai_df_builder=lambda: (logic6.find_po_grn_invoice_mismatches(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    num_df_builder=lambda: (logic6.find_po_grn_invoice_mismatches(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    ctrl_df_builder=lambda: (logic6.get_invalid_rows(p2p_raw) if p2p_raw is not None else pd.DataFrame())
                )
            with sub_tabs[3]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                # Use the main P2P2 result for chatbot context
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P3":
            inv = results.get("P2P3")
            sub_tabs = st.tabs([
                "Item-wise & Creator-wise Issues",
                "Total Financial Impact",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Item-wise & Creator-wise Issues**")
                if inv is not None and not inv.empty:
                    item_counts, creator_counts = logic6.next_level_analytics(inv)
                    st.session_state['output_df_P2P3_item_issues'] = item_counts
                    st.session_state['output_df_P2P3_creator_issues'] = creator_counts
                    st.write("Item-wise Issues")
                    st.dataframe(item_counts if not item_counts.empty else pd.DataFrame({"Info":["No item-wise issues."]}), use_container_width=True)
                    st.write("Creator-wise Issues")
                    st.dataframe(creator_counts if not creator_counts.empty else pd.DataFrame({"Info":["No creator-wise issues."]}), use_container_width=True)
                else:
                    st.info("No invalid rows available for analysis.")
            with sub_tabs[1]:
                st.markdown("**Total Financial Impact**")
                if inv is not None and not inv.empty:
                    fin_df = logic6.financial_impact(inv)
                    st.session_state['output_df_P2P3_financial_impact'] = fin_df
                    st.dataframe(fin_df, use_container_width=True)
                else:
                    st.info("No invalid rows available for analysis.")
            with sub_tabs[2]:
                _render_llm_panel(
                    "P2P3",
                    ai_df_builder=lambda: (logic6.get_invalid_rows(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    num_df_builder=lambda: (logic6.get_invalid_rows(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    ctrl_df_builder=lambda: (logic6.get_invalid_rows(p2p_raw) if p2p_raw is not None else pd.DataFrame())
                )
            with sub_tabs[3]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                # Use the main P2P3 result for chatbot context
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P4":
            sub_tabs = st.tabs([
                "Self-Approved Over Threshold",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Self-Approved Over Threshold**")
                if p2p_raw is not None and not p2p_raw.empty:
                    res = logic6.generate_self_approved_over_threshold(p2p_raw)
                    st.session_state['output_df_P2P4_self_approved'] = res
                    st.dataframe(res if not res.empty else pd.DataFrame({"Info":["No self-approved POs over threshold."]}), use_container_width=True)
                else:
                    st.info("Raw P2P sheet not available for analysis.")
            with sub_tabs[1]:
                _render_llm_panel(
                    "P2P4",
                    ai_df_builder=lambda: (logic6.generate_result(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    num_df_builder=lambda: (logic6.generate_result(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    ctrl_df_builder=lambda: (logic6.get_invalid_rows(p2p_raw) if p2p_raw is not None else pd.DataFrame())
                )
            with sub_tabs[2]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                # Use the self-approved table for chatbot context
                df_for_llm = st.session_state.get('output_df_P2P4_self_approved', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P5":
            sub_tabs = st.tabs([
                "FY Threshold Alerts (50k)",
                "Daily Threshold Alerts (10k)",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**FY Threshold Alerts (50k)**")
                detailed = results.get("P2P5", pd.DataFrame())
                if detailed is None or detailed.empty or s.get('file_bytes') is None:
                    st.info("Duplicate pairs or original workbook not available.")
                else:
                    try:
                        fy_sum, fy_detail = logic6.vendor_year_threshold_alerts(
                            detailed,
                            BytesIO(s.file_bytes),
                            sheet_name="P2P_Sample (Bots 1-20)",
                            threshold=50_000
                        )
                        st.session_state['output_df_P2P5_fy_summary'] = fy_sum
                        st.session_state['output_df_P2P5_fy_detail'] = fy_detail
                        st.write("Alerts Summary")
                        st.dataframe(fy_sum if not fy_sum.empty else pd.DataFrame({"Info":["No FY alerts over threshold."]}), use_container_width=True)
                        if fy_detail is not None and not fy_detail.empty:
                            st.write("Alerts Detail")
                            st.dataframe(fy_detail, use_container_width=True)
                    except Exception as e:
                        st.error(str(e))
            with sub_tabs[1]:
                st.markdown("**Daily Threshold Alerts (10k)**")
                detailed = results.get("P2P5", pd.DataFrame())
                if detailed is None or detailed.empty or s.get('file_bytes') is None:
                    st.info("Duplicate pairs or original workbook not available.")
                else:
                    try:
                        day_sum, day_detail = logic6.vendor_daily_threshold_alerts(
                            detailed,
                            BytesIO(s.file_bytes),
                            sheet_name="P2P_Sample (Bots 1-20)",
                            threshold=10_000
                        )
                        st.session_state['output_df_P2P5_day_summary'] = day_sum
                        st.session_state['output_df_P2P5_day_detail'] = day_detail
                        st.write("Alerts Summary")
                        st.dataframe(day_sum if not day_sum.empty else pd.DataFrame({"Info":["No daily alerts over threshold."]}), use_container_width=True)
                        if day_detail is not None and not day_detail.empty:
                            st.write("Alerts Detail")
                            st.dataframe(day_detail, use_container_width=True)
                    except Exception as e:
                        st.error(str(e))
            with sub_tabs[2]:
                _render_llm_panel(
                    "P2P5",
                    ai_df_builder=lambda: (logic6.generate_result(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    num_df_builder=lambda: (logic6.generate_result(p2p_raw) if p2p_raw is not None else pd.DataFrame()),
                    ctrl_df_builder=lambda: (logic6.get_invalid_rows(p2p_raw) if p2p_raw is not None else pd.DataFrame())
                )
            with sub_tabs[3]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                # Use the FY summary table for chatbot context
                df_for_llm = st.session_state.get('output_df_P2P5_fy_summary', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P6":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "P2P6",
                    ai_df_builder=lambda: (results.get("P2P6", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P6", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P6", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P10":
            sub_tabs = st.tabs([
                "Emergency Purchase Analysis",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Emergency Purchase Analysis**")
                if p2p_raw is not None and not p2p_raw.empty:
                    res = results.get("P2P10", pd.DataFrame())
                    st.session_state['output_df_P2P10_analysis'] = res
                    st.dataframe(res if not res.empty else pd.DataFrame({"Info":["No excessive emergency purchases detected."]}), use_container_width=True)
                else:
                    st.info("Raw P2P sheet not available for analysis.")
            with sub_tabs[1]:
                _render_llm_panel(
                    "P2P10",
                    ai_df_builder=lambda: (results.get("P2P10", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P10", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P10", pd.DataFrame()))
                )
            with sub_tabs[2]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = st.session_state.get('output_df_P2P10_analysis', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P11":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "P2P11",
                    ai_df_builder=lambda: (results.get("P2P11", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P11", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P11", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P13":
            sub_tabs = st.tabs([
                "Invoice-GRN Date Gap Analysis",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Invoice-GRN Date Gap Analysis**")
                if p2p_raw is not None and not p2p_raw.empty:
                    res = results.get("P2P13", pd.DataFrame())
                    st.session_state['output_df_P2P13_analysis'] = res
                    st.dataframe(res if not res.empty else pd.DataFrame({"Info":["No date gaps beyond threshold detected."]}), use_container_width=True)
                else:
                    st.info("Raw P2P sheet not available for analysis.")
            with sub_tabs[1]:
                _render_llm_panel(
                    "P2P13",
                    ai_df_builder=lambda: (results.get("P2P13", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P13", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P13", pd.DataFrame()))
                )
            with sub_tabs[2]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = st.session_state.get('output_df_P2P13_analysis', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P14":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "P2P14",
                    ai_df_builder=lambda: (results.get("P2P14", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P14", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P14", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P15":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "P2P15",
                    ai_df_builder=lambda: (results.get("P2P15", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P15", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P15", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P16":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "P2P16",
                    ai_df_builder=lambda: (results.get("P2P16", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P16", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P16", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P17":
            sub_tabs = st.tabs([
                "Invoices to Inactive Vendors",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Invoices to Inactive Vendors**")
                if p2p_raw is not None and not p2p_raw.empty:
                    res = results.get("P2P17", pd.DataFrame())
                    st.session_state['output_df_P2P17_analysis'] = res
                    st.dataframe(res if not res.empty else pd.DataFrame({"Info":["No invoices to inactive vendors detected."]}), use_container_width=True)
                else:
                    st.info("Raw P2P sheet not available for analysis.")
            with sub_tabs[1]:
                _render_llm_panel(
                    "P2P17",
                    ai_df_builder=lambda: (results.get("P2P17", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P17", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P17", pd.DataFrame()))
                )
            with sub_tabs[2]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = st.session_state.get('output_df_P2P17_analysis', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P18":
            sub_tabs = st.tabs([
                "GSTIN Validation Results",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**GSTIN Validation Results**")
                if p2p_raw is not None and not p2p_raw.empty:
                    res = results.get("P2P18", pd.DataFrame())
                    st.session_state['output_df_P2P18_analysis'] = res
                    st.dataframe(res if not res.empty else pd.DataFrame({"Info":["No GSTIN validation issues detected."]}), use_container_width=True)
                else:
                    st.info("Raw P2P sheet not available for analysis.")
            with sub_tabs[1]:
                _render_llm_panel(
                    "P2P18",
                    ai_df_builder=lambda: (results.get("P2P18", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P18", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P18", pd.DataFrame()))
                )
            with sub_tabs[2]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = st.session_state.get('output_df_P2P18_analysis', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "P2P20":
            sub_tabs = st.tabs([
                "Round-Sum Invoices Analysis",
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                st.markdown("**Round-Sum Invoices Analysis**")
                if p2p_raw is not None and not p2p_raw.empty:
                    res = results.get("P2P20", pd.DataFrame())
                    st.session_state['output_df_P2P20_analysis'] = res
                    st.dataframe(res if not res.empty else pd.DataFrame({"Info":["No round-sum invoices with missing rate/quantity detected."]}), use_container_width=True)
                else:
                    st.info("Raw P2P sheet not available for analysis.")
            with sub_tabs[1]:
                _render_llm_panel(
                    "P2P20",
                    ai_df_builder=lambda: (results.get("P2P20", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("P2P20", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("P2P20", pd.DataFrame()))
                )
            with sub_tabs[2]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = st.session_state.get('output_df_P2P20_analysis', pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "O2C1":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "O2C1",
                    ai_df_builder=lambda: (results.get("O2C1", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("O2C1", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("O2C1", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "O2C2":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "O2C2",
                    ai_df_builder=lambda: (results.get("O2C2", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("O2C2", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("O2C2", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "O2C3":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "O2C3",
                    ai_df_builder=lambda: (results.get("O2C3", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("O2C3", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("O2C3", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "O2C24":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "O2C24",
                    ai_df_builder=lambda: (results.get("O2C24", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("O2C24", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("O2C24", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "O2C29":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "O2C29",
                    ai_df_builder=lambda: (results.get("O2C29", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("O2C29", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("O2C29", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "O2C32":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "O2C32",
                    ai_df_builder=lambda: (results.get("O2C32", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("O2C32", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("O2C32", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "O2C39":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "O2C39",
                    ai_df_builder=lambda: (results.get("O2C39", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("O2C39", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("O2C39", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "H2R1":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "H2R1",
                    ai_df_builder=lambda: (results.get("H2R1", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("H2R1", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("H2R1", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "H2R2":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "H2R2",
                    ai_df_builder=lambda: (results.get("H2R2", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("H2R2", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("H2R2", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

        if code == "H2R44":
            sub_tabs = st.tabs([
                "AI Reports / Numerical / Control",
                "Chatbot"
            ])
            with sub_tabs[0]:
                _render_llm_panel(
                    "H2R44",
                    ai_df_builder=lambda: (results.get("H2R44", pd.DataFrame())),
                    num_df_builder=lambda: (results.get("H2R44", pd.DataFrame())),
                    ctrl_df_builder=lambda: (results.get("H2R44", pd.DataFrame()))
                )
            with sub_tabs[1]:
                st.markdown("**Chatbot: Ask about the data above**")
                chat_key = f"chatbot_history_{code}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                chat_history = st.session_state[chat_key]
                for msg in chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align:right; color:#22c55e'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:left; color:#3b82f6'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
                user_input = st.text_input(f"Ask a question about this data", key=f"chatbot_input_{code}")
                df_for_llm = results.get(code, pd.DataFrame())
                if st.button(f"Send (Chatbot)", key=f"chatbot_btn_{code}") and user_input:
                    chat_history.append({'role': 'user', 'content': user_input})
                    try:
                        response = _gen_chat_md(user_input, df_for_llm)
                    except Exception as e:
                        response = f"[Error generating response: {e}]"
                    chat_history.append({'role': 'bot', 'content': response})
                    st.session_state[chat_key] = chat_history
                    st.experimental_rerun()

    with tabs[1]:
        # st.markdown("**Output**")

        def render_selection():
            if sel_mode == "all":
                codes = _codes_for_present_categories(cats_present)
                any_complete = False
                for code in codes:
                    if proc_status.get(code) == "Complete":
                        any_complete = True
                        show_process(code)
                if not any_complete:
                    st.info("All processes failed for the current selection.")
            elif sel_mode == "category":
                codes = codes_for_category(sel_value)
                any_complete = False
                for code in codes:
                    if proc_status.get(code) == "Complete":
                        any_complete = True
                        show_process(code)
                if not any_complete:
                    st.info("All processes failed for the current selection.")
            else:
                code = sel_value
                if code:
                    show_process(code)

        render_selection()

    with tabs[2]:
        # st.markdown("**Report**")

        def line_for(code):
            cat, pname = logic6.PROCESS_TITLES[code]
            status = proc_status.get(code, "Pending")
            if status == "Failed":
                return f"- **{cat} / {pname}** — Process Failed"
            cnt = logic6.issues_count_for(code, results.get(code))
            msg = "No issues found" if cnt == 0 else f"Issues Found: {cnt}"
            return f"- **{cat} / {pname}** — {msg}"

        report_rows = []
        if sel_mode == "all":
            codes = _codes_for_present_categories(cats_present)
            for code in codes:
                st.markdown(line_for(code))
                cat, pname = logic6.PROCESS_TITLES[code]
                status = proc_status.get(code, "Pending")
                issues = logic6.issues_count_for(code, results.get(code)) if status != "Failed" else None
                report_rows.append({"Category": cat, "Process": pname, "Status": status, "Issues Found": issues})
        elif sel_mode == "category":
            codes = codes_for_category(sel_value)
            for code in codes:
                st.markdown(line_for(code))
                cat, pname = logic6.PROCESS_TITLES[code]
                status = proc_status.get(code, "Pending")
                issues = logic6.issues_count_for(code, results.get(code)) if status != "Failed" else None
                report_rows.append({"Category": cat, "Process": pname, "Status": status, "Issues Found": issues})
        else:
            code = sel_value
            if code:
                st.markdown(line_for(code))
                cat, pname = logic6.PROCESS_TITLES[code]
                status = proc_status.get(code, "Pending")
                issues = logic6.issues_count_for(code, results.get(code)) if status != "Failed" else None
                report_rows.append({"Category": cat, "Process": pname, "Status": status, "Issues Found": issues})

        excel_bytes = _build_detailed_report_excel(
            cats_present=cats_present,
            proc_status=proc_status,
            results=results,
            vendor_raw=vendor_raw,
            p2p_raw=p2p_raw,
            emp_raw=emp_raw,
            file_bytes=s.get('file_bytes')
        )
        st.download_button(
            label="Download Detailed Report (Excel)",
            data=excel_bytes,
            file_name="detailed_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_detailed_report_excel",
        )

    st.markdown("---")
    if st.button("⟵ Back to Processing", key="back_to_processing_btn"):
        s.page = "processpage"; st.rerun()
