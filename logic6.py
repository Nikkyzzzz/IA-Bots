import pandas as pd
import numpy as np
import os, json, re
from io import BytesIO
from itertools import combinations

RESULTS_DIR = "results_cache"
os.makedirs(RESULTS_DIR, exist_ok=True)

PROCESS_TITLES = {
    # ---------------- P2P ----------------
    "P2P1":  ("P2P", "Validate Vendor KYC"),
    "P2P2":  ("P2P", "PO-GRN-Invoice Match"),
    "P2P3":  ("P2P", "Post-Invoice POs"),
    "P2P4":  ("P2P", "Split Orders"),
    "P2P5":  ("P2P", "Duplicate Vendors"),
    "P2P11": ("P2P", "Vendor Concentration Risk"),
    "P2P14": ("P2P", "Over Receipt Check"),
    "P2P15": ("P2P", "Payment Term Adherence"),
    "P2P16": ("P2P", "Duplicate Invoice Detection"),
    "P2P18": ("P2P", "Non-PO Invoices"),


    # NEW
    "P2P6":  ("P2P", "Vendor GSTIN Validation"),
    # "P2P8":  ("P2P", "PO Approval Bypass"),
    "P2P10": ("P2P", "Excessive Emergency Purchases"),
    "P2P13": ("P2P", "Invoice vs GRN Gap"),
    "P2P17": ("P2P", "Invoices to Inactive Vendors"),
    "P2P20": ("P2P", "Round-Sum Invoices"),

    # ---------------- O2C ----------------
    "O2C1": ("O2C", "Overdue Delivery"),
    "O2C2": ("O2C", "Dispatch Without Invoice"),
    "O2C3": ("O2C", "Missing Customer Master Data"),

    "O2C24": ("O2C", "Sales without SO"),
    "O2C29": ("O2C", "Multiple Invoices per SO"),
    "O2C32": ("O2C", "Zero/Missing Invoice Amount"),
    "O2C39": ("O2C", "Excessive Small-Value Sales"),

    


    # ---------------- H2R ----------------
    "H2R44": ("H2R", "Duplicate Employees"),
    "H2R1": ("H2R", "Ghost employee detection"),
    "H2R2": ("H2R", "Inactive Employees In Payroll"),
}

BOT_NAMES = [v[1] for v in PROCESS_TITLES.values()]
PROC_BY_NAME = {v[1]: k for k, v in PROCESS_TITLES.items()}

def proc_symbol(status):
    return "✅" if status == "Complete" else ("❌" if status == "Failed" else "⏳")

def compute_category_status(proc_status, codes):
    if not codes:
        return 'Pending'
    vals = [proc_status.get(c, 'Pending') for c in codes]
    if all(v == 'Complete' for v in vals): return 'Complete'
    if all(v == 'Failed' for v in vals): return 'Failed'
    if any(v == 'Failed' for v in vals) and any(v == 'Complete' for v in vals): return 'Exceptions'
    return 'Pending'

def badge_color(s):
    return {'Complete': '#22c55e', 'Exceptions': '#ef4444', 'Failed': '#ef4444'}.get(s, '#f59e0b')

def issues_count_for(code, df):
    if df is None or df.empty: return 0
    if code == "P2P2":
        qty_col = "Exception Noted (Qty)"
        amt_col = "Exception Noted (Amt)"
        for c in [qty_col, amt_col]:
            if c not in df.columns:
                df[c] = ""
        return int(((df[qty_col] != "") | (df[amt_col] != "")).sum())
    # Default for all other bots
    return len(df)

def build_summary_df(proc_status, dfs):
    rows = []
    for code, (cat, pname) in PROCESS_TITLES.items():
        cnt = issues_count_for(code, dfs.get(code))
        rows.append({"Bots": pname, "Category": cat, "Issues Found": int(cnt), "Status": proc_status.get(code, "Pending")})
    return pd.DataFrame(rows)


def _resolve_vendor_column(df):
    """Return the best vendor column name in df."""
    cols = {c.lower(): c for c in df.columns}

    possible = [
        "vendor", "vendor_name", "vendorname",
        "vendor code", "vendor_code"
    ]

    for p in possible:
        if p in cols:
            return cols[p]

    # fallback to first column containing 'vendor'
    for c in df.columns:
        if "vendor" in c.lower():
            return c

    return None


# ---------------- P2P logic ----------------
def find_missing_vendor_fields(df):
    pan_re = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$')
    gst_re = re.compile(r'^(0[1-9]|1[0-9]|2[0-9]|3[0-7])[A-Z]{5}[0-9]{4}[A-Z][0-9]Z[0-9A-Z]$')

    def val(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        return s

    out = []
    for i in range(len(df)):
        row_src = df.iloc[i]
        pan = val(row_src["PAN_No"]) if "PAN_No" in df.columns else ""
        gst = val(row_src["GST_No"]) if "GST_No" in df.columns else ""
        bank = val(row_src["Bank_Account"]) if "Bank_Account" in df.columns else ""

        pan_exc = ""
        gst_exc = ""
        bank_exc = ""

        pan_u = pan.upper()
        gst_u = gst.upper()

        if pan == "":
            pan_exc = "Missing"
        elif not pan_re.match(pan_u):
            pan_exc = "Invalid"

        if gst == "":
            gst_exc = "Missing"
        elif not gst_re.match(gst_u):
            gst_exc = "Invalid"

        if bank == "":
            bank_exc = "Missing"

        if pan_exc or gst_exc or bank_exc:
            row = row_src.to_dict()
            row["PAN_Exception_Noted"] = pan_exc
            row["GST_Exception_Noted"] = gst_exc
            row["Bank_Exception_Noted"] = bank_exc
            out.append(row)
    # first_cols = ["GST_Exception_Noted", "PAN_Exception_Noted", "Bank_Exception_Noted"]
    # remaining = [c for c in out.columns if c not in first_cols]
    # out = out[first_cols + remaining]

    return pd.DataFrame(out)

def find_po_grn_invoice_mismatches(po_df: pd.DataFrame) -> pd.DataFrame:
    df = po_df.copy()
    for c in ["PO_No", "PO_Qty", "PO_Amt", "GRN_Qty", "Invoice_Qty", "Invoice_Amount"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df[df["PO_No"].notna() & (df["PO_No"].astype(str).str.strip() != "")]
    if df.empty:
        return pd.DataFrame(columns=list(po_df.columns) + [
            "Exception Noted (Qty)", "Exception Noted (Amt)", "Financial Impact",
            "PO_Qty_PO", "PO_Amt_PO", "GRN_Qty_Sum", "Invoice_Qty_Sum", "Invoice_Amount_Sum"
        ])
    summary = (
        df.groupby("PO_No", dropna=False)
          .agg(
              PO_Qty=("PO_Qty", "first"),
              PO_Amt=("PO_Amt", "first"),
              GRN_Qty=("GRN_Qty", "sum"),
              Invoice_Qty=("Invoice_Qty", "sum"),
              Invoice_Amount=("Invoice_Amount", "sum"),
          )
          .reset_index()
    )
    summary["Exception Noted (Qty)"] = np.where(
        (summary["PO_Qty"] != summary["GRN_Qty"]) | (summary["PO_Qty"] != summary["Invoice_Qty"]),
        "Quantity Mismatch", ""
    )
    summary["Exception Noted (Amt)"] = np.where(
        summary["PO_Amt"] != summary["Invoice_Amount"],
        "Amount Mismatch", ""
    )
    mismatched = summary[
        (summary["Exception Noted (Qty)"] != "") | (summary["Exception Noted (Amt)"] != "")
    ].copy()
    if mismatched.empty:
        return pd.DataFrame(columns=list(po_df.columns) + [
            "Exception Noted (Qty)", "Exception Noted (Amt)", "Financial Impact",
            "PO_Qty_PO", "PO_Amt_PO", "GRN_Qty_Sum", "Invoice_Qty_Sum", "Invoice_Amount_Sum"
        ])
    mismatched = mismatched.rename(columns={
        "PO_Qty": "PO_Qty_PO",
        "PO_Amt": "PO_Amt_PO",
        "GRN_Qty": "GRN_Qty_Sum",
        "Invoice_Qty": "Invoice_Qty_Sum",
        "Invoice_Amount": "Invoice_Amount_Sum",
    })
    out = df.merge(
        mismatched[
            [
                "PO_No",
                "PO_Qty_PO", "PO_Amt_PO", "GRN_Qty_Sum", "Invoice_Qty_Sum", "Invoice_Amount_Sum",
                "Exception Noted (Qty)", "Exception Noted (Amt)"
            ]
        ],
        on="PO_No",
        how="inner",
    )
    out["Financial Impact"] = out["PO_Amt"] - out["Invoice_Amount"]
    front = [
        "PO_No", "Exception Noted (Qty)", "Exception Noted (Amt)", "Financial Impact",
        "PO_Qty_PO", "PO_Amt_PO", "GRN_Qty_Sum", "Invoice_Qty_Sum", "Invoice_Amount_Sum",
    ]
    cols = front + [c for c in out.columns if c not in front]
    return out[cols].reset_index(drop=True)


def anomalies_by_creator(df):
    missing_table = find_missing_vendor_fields(df)
    if "Creator_ID" not in missing_table.columns:
        return pd.DataFrame()
    return (
        missing_table.groupby('Creator_ID')
        .size()
        .reset_index(name='Count')
        .sort_values(by='Count', ascending=False)
        .reset_index(drop=True)
    )

def merge_missing_with_duplicates(vendor_df, invoice_df):
    missing_table = find_missing_vendor_fields(vendor_df)
    req = ['Vendor_Name','Invoice_Date','Invoice_Amount']
    for c in req:
        if c not in invoice_df.columns:
            invoice_df[c] = np.nan
    dup_mask = invoice_df[req].duplicated(keep=False)
    dup_invoices = invoice_df[dup_mask].copy().reset_index(drop=True)
    dup_pairs, seen = [], set()
    for i in range(len(dup_invoices)):
        for j in range(i+1, len(dup_invoices)):
            if (dup_invoices.loc[i, 'Vendor_Name'] == dup_invoices.loc[j, 'Vendor_Name'] and
                dup_invoices.loc[i, 'Invoice_Date'] == dup_invoices.loc[j, 'Invoice_Date'] and
                dup_invoices.loc[i, 'Invoice_Amount'] == dup_invoices.loc[j, 'Invoice_Amount']):
                key = tuple(sorted([i, j]))
                if key not in seen:
                    seen.add(key)
                    left = dup_invoices.loc[i].to_dict()
                    right = dup_invoices.loc[j].to_dict()
                    pair = {}
                    for k, v in left.items():
                        pair[f"{k}_1"] = v
                    for k, v in right.items():
                        pair[f"{k}_2"] = v
                    dup_pairs.append(pair)
    dup_pairs_df = pd.DataFrame(dup_pairs)
    if not dup_pairs_df.empty:
        if 'Vendor_Name_1' in dup_pairs_df.columns:
            dup_pairs_df['Vendor_Name'] = dup_pairs_df['Vendor_Name_1']
        merged = pd.merge(missing_table, dup_pairs_df, on='Vendor_Name', how='inner')
    else:
        merged = pd.DataFrame()
    return merged.reset_index(drop=True)

def summarize_mismatches(mismatch_df: pd.DataFrame, po_df: pd.DataFrame, employee_master: pd.DataFrame):
    if mismatch_df is None or mismatch_df.empty:
        return (
            pd.DataFrame(columns=["Item_Code", "Issue_Count"]),
            pd.DataFrame(columns=["Department", "Issue_Count"]),
        )
    if "Item_Code" in mismatch_df.columns:
        item_summary = (
            mismatch_df.groupby("Item_Code", dropna=False)
                       .size()
                       .reset_index(name="Issue_Count")
                       .sort_values("Issue_Count", ascending=False)
                       .reset_index(drop=True)
        )
    else:
        item_summary = pd.DataFrame(columns=["Item_Code", "Issue_Count"])
    if "Creator_ID" in mismatch_df.columns and not mismatch_df["Creator_ID"].isna().all():
        creators = mismatch_df[["Creator_ID"]].copy()
    else:
        key = "PO_No" if "PO_No" in mismatch_df.columns else ("PO_Number" if "PO_Number" in mismatch_df.columns else None)
        if not key or po_df is None or key not in po_df.columns:
            return item_summary, pd.DataFrame(columns=["Department", "Issue_Count"])
        creators = (
            mismatch_df[[key]].drop_duplicates()
            .merge(po_df[[key, "Creator_ID"]], on=key, how="left")
            [["Creator_ID"]]
        )
    if creators.empty:
        return item_summary, pd.DataFrame(columns=["Department", "Issue_Count"])
    def norm_str(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip()
    def norm_digits(s: pd.Series) -> pd.Series:
        d = s.astype(str).str.replace(r"\D+", "", regex=True)
        return d.str.lstrip("0").replace({"": np.nan})
    creators = creators.copy()
    creators["Creator_ID_norm"] = norm_str(creators["Creator_ID"])
    creators["Creator_ID_digits"] = norm_digits(creators["Creator_ID"])
    if employee_master is None or employee_master.empty:
        return item_summary, pd.DataFrame(columns=["Department", "Issue_Count"])
    emp = employee_master.copy()
    emp_id_col = next((c for c in ["Employee_ID", "Employee Id", "EmployeeID", "EmployeeId"] if c in emp.columns), None)
    if emp_id_col is None:
        emp_id_col = "Employee_ID"
        emp[emp_id_col] = np.nan
    dept_col = next((c for c in ["Department", "Dept", "Department Name", "Department_Name"] if c in emp.columns), None)
    if dept_col is None:
        dept_col = "Department"
        emp[dept_col] = np.nan
    emp["Employee_ID_norm"] = norm_str(emp[emp_id_col])
    emp["Employee_ID_digits"] = norm_digits(emp[emp_id_col])
    emp["Department_norm"] = norm_str(emp[dept_col])
    mapped = creators.merge(
        emp[["Employee_ID_norm", "Department_norm"]],
        left_on="Creator_ID_norm",
        right_on="Employee_ID_norm",
        how="left",
    )
    still_unmapped = mapped["Department_norm"].isna() | (mapped["Department_norm"] == "")
    if still_unmapped.any():
        fallback = creators.loc[still_unmapped, ["Creator_ID_digits"]].merge(
            emp[["Employee_ID_digits", "Department_norm"]],
            left_on="Creator_ID_digits",
            right_on="Employee_ID_digits",
            how="left",
        )
        mapped.loc[still_unmapped, "Department_norm"] = fallback["Department_norm"].values
    mapped["Department"] = mapped["Department_norm"].replace({"": np.nan}).fillna("Unknown / Not Mapped")
    dept_summary = (
        mapped.groupby("Department", dropna=False)
              .size()
              .reset_index(name="Issue_Count")
              .sort_values("Issue_Count", ascending=False)
              .reset_index(drop=True)
    )
    return item_summary, dept_summary

def calculate_financial_impact_df(mismatch_df):
    if "Financial Impact" not in mismatch_df.columns:
        raise ValueError("Mismatch DataFrame must contain 'Financial Impact' column")
    positive_impact = mismatch_df.loc[mismatch_df["Financial Impact"] > 0, "Financial Impact"].sum()
    negative_impact = mismatch_df.loc[mismatch_df["Financial Impact"] < 0, "Financial Impact"].sum()
    return pd.DataFrame({
        "Total Positive Impact": [positive_impact],
        "Total Negative Impact": [negative_impact]
    })

def get_invalid_rows(src, sheet_name="P2P_Sample (Bots 1-20)"):
    if isinstance(src, (str, bytes, os.PathLike, BytesIO)):
        df = pd.read_excel(src, sheet_name=sheet_name)
    else:
        df = src.copy()
    for c in ["PO_Date", "Invoice_Date"]:
        if c not in df.columns:
            df[c] = pd.NaT
    df['PO_Date'] = pd.to_datetime(df['PO_Date'], errors='coerce')
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'], errors='coerce')
    invalid_rows = df[df['PO_Date'] > df['Invoice_Date']].copy()
    first_cols = ["PO_Date", "Invoice_Date", "PO_No"]
    remaining = [c for c in invalid_rows.columns if c not in first_cols + ["_approved_clean", "_created_clean"]]
    invalid_rows = invalid_rows[first_cols + remaining].reset_index(drop=True)
    return invalid_rows

def next_level_analytics(invalid_rows: pd.DataFrame):
    df = invalid_rows.copy()
    if "Item_Code" not in df.columns:
        df["Item_Code"] = np.nan
    if "PO_Created_By" not in df.columns:
        df["PO_Created_By"] = np.nan
    item_counts = (
        df.groupby('Item_Code', dropna=False)
          .size()
          .reset_index(name="Issues Found")
          .sort_values(by="Issues Found", ascending=False)
          .reset_index(drop=True)
    )
    creator_counts = (
        df.groupby('PO_Created_By', dropna=False)
          .size()
          .reset_index(name="Issues Found")
          .sort_values(by="Issues Found", ascending=False)
          .reset_index(drop=True)
    )
    return item_counts, creator_counts

def financial_impact(invalid_rows: pd.DataFrame):
    df = invalid_rows.copy()
    if "Invoice_Amount" not in df.columns:
        df["Invoice_Amount"] = 0
    total = pd.to_numeric(df['Invoice_Amount'], errors='coerce').fillna(0).sum()
    return pd.DataFrame({"Total_Financial_Impact": [total]})

# ---------------- Split Orders (P2P4) ----------------
def generate_result(df: pd.DataFrame, threshold: float = 10_000) -> pd.DataFrame:
    d = df.copy()
    d = d.replace(r'^\s*$', np.nan, regex=True)
    d['PO_Date'] = pd.to_datetime(d['PO_Date'], errors='coerce')
    d['Invoice_Amount'] = pd.to_numeric(d['Invoice_Amount'], errors='coerce')
    req = ['PO_No', 'PO_Date', 'Vendor_Name', 'Item_Code']
    d = d.dropna(subset=req)
    g = (
        d.groupby(['Vendor_Name', 'Item_Code', 'PO_Date'], dropna=False)
         .agg(Distinct_PO_Count=('PO_No', lambda s: s.dropna().nunique()),
              Group_Total_Invoice=('Invoice_Amount', 'sum'))
         .reset_index()
    )
    g = g[(g['Distinct_PO_Count'] > 1) & (g['Group_Total_Invoice'] > threshold)]
    out = (
        d.merge(g[['Vendor_Name','Item_Code','PO_Date','Group_Total_Invoice']],
                on=['Vendor_Name', 'Item_Code', 'PO_Date'], how='inner')
         .sort_values(['Vendor_Name', 'Item_Code', 'PO_Date', 'PO_No'])
         .reset_index(drop=True)
    )
    keys = pd.Series(list(zip(out['Vendor_Name'], out['Item_Code'], out['PO_Date'])))
    group_numbers = pd.factorize(keys, sort=True)[0] + 1
    out.insert(0, 'Group', pd.Series(group_numbers, index=out.index).map(lambda x: f'Group_{x}'))
    first_cols = ['Group', 'Vendor_Name', 'Item_Code', 'PO_Date', 'PO_No', 'Invoice_Amount', 'Group_Total_Invoice']
    remaining = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + remaining]
    return out

def generate_self_approved_over_threshold(df: pd.DataFrame, threshold: float = 10_000) -> pd.DataFrame:
    d = df.copy()
    for col in ["PO_Approved_By", "PO_Created_By"]:
        if col in d.columns:
            d[col] = d[col].astype(str)
            d[col] = d[col].replace(r'^\s*$', np.nan, regex=True)
            d[col] = d[col].str.replace(r'\s+', ' ', regex=True).str.strip()
        else:
            d[col] = np.nan
    d["Invoice_Amount"] = pd.to_numeric(d.get("Invoice_Amount", np.nan), errors="coerce")
    d = d.dropna(subset=["PO_Approved_By", "PO_Created_By"])
    d["_approved_clean"] = d["PO_Approved_By"].astype(str).str.lower().str.strip()
    d["_created_clean"]  = d["PO_Created_By"].astype(str).str.lower().str.strip()
    mask = np.where(
        (d["_approved_clean"] == d["_created_clean"]) & (d["Invoice_Amount"] > threshold),
        True, False
    )
    out = d[mask].copy()
    if out.empty:
        return out.drop(columns=["_approved_clean", "_created_clean"], errors="ignore")
    first_cols = ["PO_Created_By", "PO_Approved_By", "PO_No", "PO_Date", "Invoice_Amount"]
    remaining = [c for c in out.columns if c not in first_cols + ["_approved_clean", "_created_clean"]]
    out = out[first_cols + remaining].reset_index(drop=True)
    out = out.drop(columns=["_approved_clean", "_created_clean"], errors="ignore")
    return out

# ---------------- Duplicate Vendors (P2P5) ----------------
def find_matching_rows(file_path: str, sheet_name: str = ""):
    df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df = df.reset_index(drop=True)
    df.insert(0, "RowNo", df.index + 1)

    keys = {
        "PAN_No": "PAN Match",
        "GST_No": "GST Match",
        "Vendor_Name": "Vendor Name Match",
        "Bank_Account": "Bank Account Match",
    }

    pairs = {}
    for col, label in keys.items():
        if col in df.columns:
            tmp = df.dropna(subset=[col])
            groups = tmp.groupby(col, dropna=True)["RowNo"].apply(list)
            for rows in groups:
                if len(rows) >= 2:
                    for a, b in combinations(sorted(rows), 2):
                        pairs.setdefault((a, b), set()).add(label)

    rows_info = df.set_index("RowNo")
    records = []
    for (a, b), labels in sorted(pairs.items()):
        records.append({
            "Row_A": a,
            "Row_B": b,
            "Exception_Noted": ", ".join(sorted(labels))
        })
    matches_df = pd.DataFrame(records)

    detailed_rows = []
    for rec in records:
        a, b = rec["Row_A"], rec["Row_B"]
        note = rec["Exception_Noted"]
        row_a = rows_info.loc[a].to_dict()
        row_b = rows_info.loc[b].to_dict()
        flat = {f"A_{k}": v for k, v in row_a.items()}
        flat.update({f"B_{k}": v for k, v in row_b.items()})
        flat.update({"Row_A": a, "Row_B": b, "Exception_Noted": note})
        detailed_rows.append(flat)
    detailed_df = pd.DataFrame(detailed_rows)

    return detailed_df

def vendor_year_threshold_alerts(
    matches_detailed_df: pd.DataFrame,
    file_path,
    sheet_name: str = "P2P_Sample (Bots 1-20)",
    threshold: float = 50_000,
):
    def fiscal_year(dt):
        y = dt.year
        return f"{y}-{y+1}" if dt.month >= 4 else f"{y-1}-{y}"

    vendor_cols = [c for c in matches_detailed_df.columns if c.endswith("Vendor_Name")]
    vendors = pd.Series(dtype=str)
    for c in vendor_cols:
        v = matches_detailed_df[c].astype(str).str.strip()
        v = v.replace({"": np.nan, "nan": np.nan}).dropna()
        vendors = pd.concat([vendors, v], ignore_index=True)
    vendors = vendors.dropna().drop_duplicates()

    base = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
    base = base.replace(r'^\s*$', np.nan, regex=True)
    if "Vendor_Name" not in base.columns:
        raise ValueError("Vendor_Name column not found in the source sheet.")
    base["Vendor_Name"] = base["Vendor_Name"].astype(str).str.strip()
    base = base[base["Vendor_Name"].isin(set(vendors))]
    if base.empty:
        return pd.DataFrame(columns=["Vendor_Name","FY","Total_Invoice_Amount","Exception_Noted"]), pd.DataFrame()

    base["Invoice_Date"] = pd.to_datetime(base.get("Invoice_Date", pd.NaT), errors="coerce")
    base["Invoice_Amount"] = pd.to_numeric(base.get("Invoice_Amount", np.nan), errors="coerce")
    base = base.dropna(subset=["Invoice_Date","Invoice_Amount"])
    base["FY"] = base["Invoice_Date"].apply(fiscal_year)

    grp = (
        base.groupby(["Vendor_Name","FY"], dropna=False, as_index=False)["Invoice_Amount"]
        .sum()
        .rename(columns={"Invoice_Amount":"Total_Invoice_Amount"})
    )
    grp["Exception_Noted(50k per Year)"] = np.where(grp["Total_Invoice_Amount"] > threshold, "Alert", "OK")

    alert_summary_df = grp[grp["Exception_Noted(50k per Year)"] == "Alert"].reset_index(drop=True)
    if alert_summary_df.empty:
        return alert_summary_df, pd.DataFrame()

    alert_keys = set(map(tuple, alert_summary_df[["Vendor_Name","FY"]].itertuples(index=False, name=None)))
    alert_detail_df = base[base[["Vendor_Name","FY"]].apply(tuple, axis=1).isin(alert_keys)].reset_index(drop=True)

    return alert_summary_df, alert_detail_df

def vendor_daily_threshold_alerts(
    matches_result_df: pd.DataFrame,
    file_path,
    sheet_name: str = "P2P_Sample (Bots 1-20)",
    threshold: float = 10_000,
):
    vendor_cols = [c for c in matches_result_df.columns if c.endswith("Vendor_Name")] or \
                  (["Vendor_Name"] if "Vendor_Name" in matches_result_df.columns else [])
    if not vendor_cols:
        return pd.DataFrame(columns=["Vendor_Name","Day","Total_Invoice_Amount","Exception_Noted(10k per Day)"]), pd.DataFrame()
    vendors = pd.concat(
        [matches_result_df[c].astype(str).str.strip() for c in vendor_cols],
        ignore_index=True
    ).replace({"": np.nan, "nan": np.nan}).dropna().drop_duplicates()

    base = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str).replace(r'^\s*$', np.nan, regex=True)
    if "Vendor_Name" not in base.columns:
        raise ValueError("Vendor_Name column not found in the source sheet.")
    base["Vendor_Name"] = base["Vendor_Name"].astype(str).str.strip()
    base = base[base["Vendor_Name"].isin(set(vendors))].copy()
    if base.empty:
        return pd.DataFrame(columns=["Vendor_Name","Day","Total_Invoice_Amount","Exception_Noted(10k per Day)"]), pd.DataFrame()

    base["Invoice_Date"] = pd.to_datetime(base.get("Invoice_Date", pd.NaT), errors="coerce")
    amt = base.get("Invoice_Amount", np.nan)
    amt = pd.to_numeric(amt.astype(str).str.replace(",", "", regex=False), errors="coerce")
    base["Invoice_Amount"] = amt
    base = base.dropna(subset=["Invoice_Date", "Invoice_Amount"])
    base["Day"] = base["Invoice_Date"].apply(lambda d: f"{d.year}-{d.month}-{d.day}")

    grp = (
        base.groupby(["Vendor_Name", "Day"], as_index=False, dropna=False)["Invoice_Amount"]
            .sum()
            .rename(columns={"Invoice_Amount": "Total_Invoice_Amount"})
    )
    grp["Exception_Noted(10k per Day)"] = np.where(grp["Total_Invoice_Amount"] > threshold, "Alert", "OK")

    alert_summary_df = grp[grp["Exception_Noted(10k per Day)"] == "Alert"].reset_index(drop=True)
    if alert_summary_df.empty:
        return alert_summary_df, pd.DataFrame()

    keyset = set(map(tuple, alert_summary_df[["Vendor_Name", "Day"]].itertuples(index=False, name=None)))
    alert_detail_df = base[base[["Vendor_Name", "Day"]].apply(tuple, axis=1).isin(keyset)].reset_index(drop=True)
    alert_detail_df["Exception_Noted(10k per Day)"] = "Alert"

    return alert_summary_df, alert_detail_df




# ---------------------------------------------------------------------
# NEW BOT FUNCTIONS (ADDED WITHOUT MODIFYING ANY EXISTING LOGIC)
# ---------------------------------------------------------------------

# ----------------------- P2P 6 GSTIN VALIDATION -----------------------
def validate_gstin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows with invalid GSTIN.
    """
    gstin_pattern = re.compile(
        r"^(0[1-9]|1[0-9]|2[0-8])"   
        r"[A-Z]{5}"
        r"[0-9]{4}"
        r"[A-Z]{1}"
        r"[1-9]{1}"
        r"Z"
        r"[0-9A-Z]{1}$"
    )
    df_out = df.copy()
    df_out["GST_No"] = df_out["GST_No"].astype(str).str.strip()

    df_out["Exception_Noted"] = df_out["GST_No"].apply(
        lambda x: "Valid" if gstin_pattern.match(x) else "Invalid GSTIN"
    )
    return df_out[df_out["Exception_Noted"] != "Valid"]


# ----------------------- P2P 17 INACTIVE VENDOR INVOICES -----------------------
def check_invoice_to_inactive_vendor(df4, df5,
                                     vendor_col="Vendor_Name",
                                     amount_col="Invoice_Amount",
                                     status_col="Active_Status"):
    df = df4.copy()
    vendor_status_map = df5.set_index(vendor_col)[status_col].to_dict()
    df["_vendor_status"] = df[vendor_col].map(vendor_status_map)

    df["Exception_Invoice_Status"] = df.apply(
        lambda row: "Inactive Vendor"
        if str(row.get("_vendor_status", "")).lower() == "inactive"
        and pd.notna(row.get(amount_col))
        and float(row.get(amount_col, 0)) > 0
        else "Active",
        axis=1,
    )

    df = df.drop(columns=["_vendor_status"])
    return df[df["Exception_Invoice_Status"] != "Active"]


# ----------------------- P2P 13 INVOICE vs GRN DATE GAP -----------------------
def check_invoice_vs_grn(df4, days_threshold=10):
    df = df4.copy()
    df["GRN_Date"] = pd.to_datetime(df["GRN_Date"], errors="coerce")
    df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], errors="coerce")

    df["days_gap"] = (df["Invoice_Date"] - df["GRN_Date"]).dt.days

    def flag_gap(v):
        if pd.isna(v):
            return "Missing Date"
        if v < -days_threshold:
            return "Invoice Too Early"
        if v > days_threshold:
            return "Invoice Too Late"
        return "OK"

    df["Exception_Noted"] = df["days_gap"].apply(flag_gap)
    return df[df["Exception_Noted"] != "OK"]


# ----------------------- P2P 20 ROUND-SUM INVOICES -----------------------
def check_round_sum_invoices(df4):
    df = df4.copy()
    for col in ["Invoice_Rate", "Invoice_Qty", "Invoice_Amount"]:
        if col not in df.columns:
            df[col] = np.nan

    def flag(row):
        rate = row["Invoice_Rate"]
        qty = row["Invoice_Qty"]
        amt = row["Invoice_Amount"]

        rate_missing = pd.isna(rate) or float(rate) == 0
        qty_missing = pd.isna(qty) or float(qty) == 0

        if rate_missing and qty_missing:
            return "Both Missing (Rate & Qty)"
        if rate_missing:
            return "Invoice Rate Missing"
        if qty_missing:
            return "Invoice Qty Missing"

        try:
            calc = float(qty) * float(rate)
            if pd.isna(amt):
                return "Amount Missing"
            if not np.isclose(calc, float(amt), atol=1.0):
                return "Qty*Rate != Amount"
        except Exception:
            return "Calculation Error"
        return "OK"

    df["Round_Sum_Flag"] = df.apply(flag, axis=1)
    return df[df["Round_Sum_Flag"] != "OK"]


# ----------------------- P2P 10 EXCESSIVE EMERGENCY PURCHASES -----------------------
def excessive_emergency_purchases(df: pd.DataFrame, threshold=0.1):
    df2 = df.copy()
    total = len(df2)
    emergency_rows = df2[df2["PO_Type"] == "Emergency"]
    emergency_count = len(emergency_rows)

    ratio = emergency_count / total if total > 0 else 0
    status = "Alert" if ratio > threshold else "Below Threshold"

    emergency_rows = emergency_rows.copy()
    emergency_rows["Exception Noted"] = status
    return emergency_rows


# =========================== P2P BOT 11 – Vendor Concentration Risk ===========================

def bot_11_vendor_concentration(df):
    if df is None or df.empty:
        return pd.DataFrame()

    vend_col = _resolve_vendor_column(df)
    if not vend_col:
        return pd.DataFrame()

    amt_col = "Invoice_Amount"
    if amt_col not in df.columns:
        return pd.DataFrame()

    df_tmp = df.copy()

    # Step 1
    vendor_sum = df_tmp.groupby(vend_col, as_index=False)[amt_col].sum()
    vendor_sum.rename(columns={amt_col: "Vendor_Invoice_Sum"}, inplace=True)

    # Step 2
    total_amt = df_tmp[amt_col].sum()

    # Step 3
    vendor_sum["Vendor Concentration %"] = (
        vendor_sum["Vendor_Invoice_Sum"] / total_amt * 100
    )

    # Step 4
    vendor_sum["Exception Noted"] = vendor_sum["Vendor Concentration %"].apply(
        lambda x: "Risk" if x > 15 else "Standard"
    )

    # Step 5 merge
    df_out = df_tmp.merge(
        vendor_sum[[vend_col, "Vendor Concentration %", "Exception Noted"]],
        on=vend_col, how="left"
    )

    # Step 6 only exception
    df_out = df_out[df_out["Exception Noted"] == "Risk"]

    return df_out


# =========================== P2P BOT 14 – Over Receipt ===========================

def bot_14_over_receipt(df2):
    """
    Compares PO_Qty vs GRN_Qty and flags Excess, Match, Shortage with validation.
    """
    try:
        df = df2.copy()

        def check_qty(row):
            po = row['PO_Qty']
            grn = row['GRN_Qty']

            errors = []
            if not isinstance(po, (int, float)) or pd.isna(po):
                errors.append("PO_Qty")
            if not isinstance(grn, (int, float)) or pd.isna(grn):
                errors.append("GRN_Qty")

            if errors:
                return "Value Error in " + ", ".join(errors)

            if grn > po:
                return "Excess"
            elif grn == po:
                return "Match"
            else:
                return "Shortage"

        df["Exception Noted"] = df.apply(check_qty, axis=1)
        return df[df["Exception Noted"] != "Match"]

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# =========================== P2P BOT 15 – Payment Term Adherence ===========================

def bot_15_payment_terms(df2):
    """
    Checks if the payment adheres to payment terms in days.
    """
    try:
        df = df2.copy()

        df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'], errors='coerce')
        df['Payment_Date'] = pd.to_datetime(df['Payment_Date'], errors='coerce')

        def check_payment(row):
            invoice_date = row['Invoice_Date']
            payment_date = row['Payment_Date']
            term_days = row['Payment_Terms_Days']

            errors = []
            if pd.isna(invoice_date): errors.append("Invoice_Date")
            if pd.isna(payment_date): errors.append("Payment_Date")
            if not isinstance(term_days, (int, float)) or pd.isna(term_days):
                errors.append("Payment_Term_Days")

            if errors:
                return "Value Error in " + ", ".join(errors)

            days_diff = (payment_date - invoice_date).days

            if days_diff < 0:
                return "Pre-Payment"
            elif days_diff <= term_days:
                return "Early Payment"
            else:
                return "Late Payment"

        df["Exception Noted"] = df.apply(check_payment, axis=1)
        return df[df["Exception Noted"] != "Early Payment"]

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# =========================== P2P BOT 16 – Duplicate Invoice Detection ===========================

def bot_16_duplicate_invoice(df):
    if df is None or df.empty:
        return pd.DataFrame()

    vend_col = _resolve_vendor_column(df)
    if not vend_col:
        return pd.DataFrame()

    keys = [vend_col, "Invoice_Date", "Invoice_Amount"]
    for k in keys:
        if k not in df.columns:
            return pd.DataFrame()

    df_non_null = df.dropna(subset=keys).copy()
    df_non_null["Exception Noted"] = df_non_null.duplicated(subset=keys, keep=False)
    df_non_null["Exception Noted"] = df_non_null["Exception Noted"].apply(
        lambda x: "Duplicate" if x else ""
    )

    df_out = df.merge(
        df_non_null[keys + ["Exception Noted"]],
        on=keys, how="left"
    )
    df_out["Exception Noted"] = df_out["Exception Noted"].fillna("")

    return df_out[df_out["Exception Noted"] != ""]


# =========================== P2P BOT 18 – Non-PO Invoices ===========================

def bot_18_non_po_invoices(df2):
    """
    Flags invoices where PO_No is blank or NaN.
    """
    try:
        df = df2.copy()
        df["Exception Noted"] = df["PO_No"].apply(
            lambda x: "Empty_PO" if pd.isna(x) or str(x).strip() == "" else ""
        )
        return df[df["Exception Noted"] != ""]
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})



# ---------------- O2C logic (unchanged) ----------------
def check_overdue_delivery(df, delay_threshold=5):
    df["SO_Date"] = pd.to_datetime(df["SO_Date"], errors="coerce")
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")
    df["DateDiff"] = (df["Delivery_Date"] - df["SO_Date"]).dt.days
    overdue = df[df["DateDiff"] > delay_threshold].copy()
    overdue["Exception_Noted (5_Days)"] = "Overdue Delivery"
    first_cols = ['Exception_Noted (5_Days)', 'SO_Date', 'Delivery_Date']
    remaining = [c for c in overdue.columns if c not in first_cols]
    overdue = overdue[first_cols + remaining]
    return overdue.reset_index(drop=True)

def check_dispatch_without_invoice(df):
    no_invoice = df[df["Delivery_No"].notna() & df["Invoice_No"].isna()].copy()
    no_invoice["Exception_Noted"] = "Dispatch Without Invoice"
    first_cols = ['Exception_Noted', 'Delivery_No', 'Invoice_No']
    remaining = [c for c in no_invoice.columns if c not in first_cols]
    no_invoice = no_invoice[first_cols + remaining]
    return no_invoice.reset_index(drop=True)

def get_missing_customer_data(df, check_cols=['GST_No', 'PAN_No', 'Credit_Limit']):
    result_df = df.copy()
    result_df[check_cols] = result_df[check_cols].replace('', np.nan)
    def find_missing(row):
        missing = [col for col in check_cols if pd.isna(row[col])]
        if missing:
            return " + ".join(missing) + " Missing"
        return ""
    result_df['Exception_Noted'] = result_df.apply(find_missing, axis=1)
    result_df = result_df[result_df['Exception_Noted'] != ""]
    first_cols = ['Exception_Noted', 'PAN_No', 'GST_No', 'Credit_Limit']
    remaining = [c for c in result_df.columns if c not in first_cols]
    result_df = result_df[first_cols + remaining]
    return result_df.reset_index(drop=True)


# ---------------------------------------------------------
# NEW O2C BOT FUNCTIONS
# ---------------------------------------------------------
def bot24_sales_without_so(df):
    """O2C24 – Sales without SO"""
    if "SO_No" not in df.columns or "Invoice_No" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["SO_No"] = df["SO_No"].astype(str).str.strip()
    df["Invoice_No"] = df["Invoice_No"].astype(str).str.strip()

    condition = (
        ((df["SO_No"] == "") | (df["SO_No"].isna()) | (df["SO_No"].str.lower() == "nan"))
        &
        ((df["Invoice_No"] != "") & (df["Invoice_No"].str.lower() != "nan"))
    )

    df["Exception_Noted"] = np.where(condition, "Invoice without SO ID", "Ok")
    return df[df["Exception_Noted"] == "Invoice without SO ID"].reset_index(drop=True)


def bot29_multiple_invoices_for_so(df):
    """O2C29 – Multiple Invoices per SO"""
    if "SO_No" not in df.columns or "Invoice_No" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["SO_No"] = df["SO_No"].astype(str).str.strip()
    df["Invoice_No"] = df["Invoice_No"].astype(str).str.strip()

    invoice_counts = (
        df[df["Invoice_No"] != ""]
        .groupby("SO_No")["Invoice_No"]
        .transform("count")
    )

    df["Exception_Noted"] = np.where(invoice_counts > 1, "Multiple Invoices", "Standard")
    return df[df["Exception_Noted"] == "Multiple Invoices"].reset_index(drop=True)


def bot32_zero_or_missing_invoice_amount(df):
    """O2C32 – Zero/Missing Invoice Amount"""
    if "Invoice_Amount" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["Invoice_Amount_str"] = df["Invoice_Amount"].astype(str).str.strip()

    condition = (
        (df["Invoice_Amount"].fillna(0) == 0) |
        (df["Invoice_Amount_str"] == "") |
        (df["Invoice_Amount_str"].str.lower() == "nan")
    )

    df["Exception_Noted"] = np.where(condition, "Zero / Missing Amount", "Ok")
    return df[df["Exception_Noted"] == "Zero / Missing Amount"].reset_index(drop=True)


def bot39_excessive_small_value_sales(df, amount_threshold=10000, percent_threshold=0.1):
    """O2C39 – Excessive small-value sales"""
    if "Taxable_Amount" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["Taxable_Amount_clean"] = pd.to_numeric(df["Taxable_Amount"], errors="coerce")
    df["Is_Small_Value"] = df["Taxable_Amount_clean"] <= amount_threshold

    total_rows = len(df)
    small_value_count = df["Is_Small_Value"].sum()

    if total_rows == 0:
        return pd.DataFrame()

    small_value_percent = (small_value_count / total_rows) * 100
    label = "Alert" if small_value_percent >= percent_threshold else "Standard"

    df.loc[df["Is_Small_Value"], "Exception_Noted"] = label
    return df[df["Is_Small_Value"]].drop(columns=["Is_Small_Value"]).reset_index(drop=True)


# ---------------------------------------------------------
# NEW H2R BOT FUNCTION
# ---------------------------------------------------------
def bot44_duplicate_employees(df):
    """H2R44 – Duplicate Employees"""
    if "Bank_Account" not in df.columns or "PAN_No" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["Bank_Account"] = df["Bank_Account"].astype(str).str.strip()
    df["PAN_No"] = df["PAN_No"].astype(str).str.strip()

    df["Dup_Bank"] = df.duplicated("Bank_Account", keep=False) & (df["Bank_Account"] != "")
    df["Dup_PAN"] = df.duplicated("PAN_No", keep=False) & (df["PAN_No"] != "")

    df["Exception_Noted"] = np.where(
        df["Dup_Bank"] & df["Dup_PAN"], "Duplicate Bank & PAN",
        np.where(df["Dup_Bank"], "Duplicate Bank Account",
        np.where(df["Dup_PAN"], "Duplicate PAN", ""))
    )

    df["Duplicate_Group"] = np.where(
        df["Dup_Bank"] & df["Dup_PAN"],
        "BANKPAN_" + df["Bank_Account"] + "_" + df["PAN_No"],
        np.where(df["Dup_Bank"], "BANK_" + df["Bank_Account"],
        np.where(df["Dup_PAN"], "PAN_" + df["PAN_No"], ""))
    )

    return df[df["Exception_Noted"] != ""].reset_index(drop=True)



# ---------------- H2R logic (NEW) ----------------
def find_ghost_employees(master_file, attendance_file):
    """
    Identify ghost employees:
    - Employees with attendance > 0
    - Not listed in the employee master file
    """
    master_df = pd.DataFrame(master_file)
    attendance_df = pd.DataFrame(attendance_file)

    master_df.columns = master_df.columns.str.strip().str.replace(" ", "_")
    attendance_df.columns = attendance_df.columns.str.strip().str.replace(" ", "_")

    # Normalize types
    if "Employee_ID" in master_df.columns:
        master_df["Employee_ID"] = master_df["Employee_ID"].astype(str).str.strip()
    if "Employee_ID" in attendance_df.columns:
        attendance_df["Employee_ID"] = attendance_df["Employee_ID"].astype(str).str.strip()
    if "Present_Days" in attendance_df.columns:
        attendance_df["Present_Days"] = pd.to_numeric(attendance_df["Present_Days"], errors="coerce").fillna(0)

    valid_ids = set(master_df.get("Employee_ID", pd.Series(dtype=str)))

    ghost_df = attendance_df[
        attendance_df.get("Present_Days", 0) > 0
    ].copy()
    if "Employee_ID" in ghost_df.columns:
        ghost_df = ghost_df[~ghost_df["Employee_ID"].isin(valid_ids)]

    first_cols = [c for c in ['Employee_ID', 'Present_Days'] if c in ghost_df.columns]
    remaining = [c for c in ghost_df.columns if c not in first_cols]
    ghost_df = ghost_df[first_cols + remaining] if first_cols else ghost_df
    return ghost_df.reset_index(drop=True)

def find_attendance_after_exit(
    file_path: str,
    employee_sheet: str = "Employee_Master",
    attendance_sheet: str = "Attendance_Register",
    month_col: str = "Month",
    year_col: str | None = None
) -> pd.DataFrame:
    # Load
    emp = pd.read_excel(file_path, sheet_name=employee_sheet, dtype={"Employee_ID": str, "Employee_Name": str})
    att = pd.read_excel(file_path, sheet_name=attendance_sheet, dtype={"Employee_ID": str, "Employee_Name": str})

    emp["Exit_Date"] = pd.to_datetime(emp.get("Exit_Date"), errors="coerce")
    emp = emp[emp["Exit_Date"].notna()].copy()

    # derive month base
    if year_col and year_col in att.columns and month_col in att.columns:
        mval = str(att[month_col].dropna().astype(str).iloc[0]).strip()
        yval = str(att[year_col].dropna().astype(str).iloc[0]).strip()
        base = pd.to_datetime(f"1 {mval} {yval}", errors="coerce")
    elif month_col in att.columns:
        mval = str(att[month_col].dropna().astype(str).iloc[0]).strip()
        ts = pd.to_datetime(mval, errors="coerce")
        if pd.isna(ts):
            base = pd.to_datetime("1 " + mval, errors="coerce")
        else:
            base = pd.Timestamp(ts.year, ts.month, 1)
    else:
        raise ValueError("Attendance_Register must contain Month (and optionally Year) to infer month/year.")

    if pd.isna(base):
        raise ValueError("Unable to parse Month/Year from Attendance_Register.")

    # daily columns D1..D31
    day_cols = [c for c in att.columns if re.fullmatch(r"D(?:[1-9]|[12]\d|3[01])", str(c))]
    if not day_cols:
        return pd.DataFrame(columns=["Employee_ID","Employee_Name","Exit_Date","Date","Status"])

    long = att.melt(
        id_vars=[c for c in ["Employee_ID","Employee_Name", month_col, year_col] if c and c in att.columns],
        value_vars=day_cols,
        var_name="Day",
        value_name="Status"
    )
    long["Day_Num"] = pd.to_numeric(long["Day"].str.extract(r"(\d+)")[0], errors="coerce")
    long = long[long["Day_Num"].between(1,31)]
    long["Date"] = base + pd.to_timedelta(long["Day_Num"] - 1, unit="D")

    s = long["Status"].astype(str).str.strip().str.upper()
    long = long[~s.eq("A")]

    out = long.merge(emp[["Employee_ID","Employee_Name","Exit_Date"]].astype({"Employee_ID": str}),
                     on=["Employee_ID","Employee_Name"], how="inner")

    out = out[out["Date"] > out["Exit_Date"]].copy()

    ordered = [c for c in ["Employee_ID","Employee_Name", month_col, year_col, "Exit_Date","Date","Status"] if c in out.columns]
    extra = [c for c in out.columns if c not in ordered + ["Day","Day_Num"]]
    return out[ordered + extra].sort_values(["Employee_ID","Date"]).reset_index(drop=True)

# ---------------- Save/Load ----------------
def save_job_results(job_id, results, proc_status, statuses,
                     df_vendor=None, df_p2p=None, df_emp=None, df_o2c=None, df_cust=None,
                     df_att=None):
    for key, df in results.items():
        try: df.to_csv(os.path.join(RESULTS_DIR, f"{job_id}_{key}.csv"), index=False)
        except Exception: pass
    if df_vendor is not None:
        try: df_vendor.to_csv(os.path.join(RESULTS_DIR, f"{job_id}_RAW_VENDOR.csv"), index=False)
        except Exception: pass
    if df_p2p is not None:
        try: df_p2p.to_csv(os.path.join(RESULTS_DIR, f"{job_id}_RAW_P2P.csv"), index=False)
        except Exception: pass
    if df_emp is not None:
        try: df_emp.to_csv(os.path.join(RESULTS_DIR, f"{job_id}_RAW_EMP.csv"), index=False)
        except Exception: pass
    if df_o2c is not None:
        try: df_o2c.to_csv(os.path.join(RESULTS_DIR, f"{job_id}_RAW_O2C.csv"), index=False)
        except Exception: pass
    if df_cust is not None:
        try: df_cust.to_csv(os.path.join(RESULTS_DIR, f"{job_id}_RAW_CUST.csv"), index=False)
        except Exception: pass
    if df_att is not None:
        try: df_att.to_csv(os.path.join(RESULTS_DIR, f"{job_id}_RAW_ATT.csv"), index=False)
        except Exception: pass
    with open(os.path.join(RESULTS_DIR, f"{job_id}_meta.json"), "w") as f:
        json.dump({"proc_status": proc_status, "statuses": statuses}, f)

def load_job(job_id):
    dfs, saved_proc_status = {}, None
    vendor_raw, p2p_raw, emp_raw = None, None, None
    o2c_raw, cust_raw, att_raw = None, None, None
    meta_path = os.path.join(RESULTS_DIR, f"{job_id}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            saved_proc_status = meta.get("proc_status")
    for k in PROCESS_TITLES.keys():
        p = os.path.join(RESULTS_DIR, f"{job_id}_{k}.csv")
        if os.path.exists(p):
            try: dfs[k] = pd.read_csv(p)
            except Exception: pass
    vr = os.path.join(RESULTS_DIR, f"{job_id}_RAW_VENDOR.csv")
    pr = os.path.join(RESULTS_DIR, f"{job_id}_RAW_P2P.csv")
    er = os.path.join(RESULTS_DIR, f"{job_id}_RAW_EMP.csv")
    or_ = os.path.join(RESULTS_DIR, f"{job_id}_RAW_O2C.csv")
    cr_ = os.path.join(RESULTS_DIR, f"{job_id}_RAW_CUST.csv")
    ar_ = os.path.join(RESULTS_DIR, f"{job_id}_RAW_ATT.csv")
    if os.path.exists(vr):
        try: vendor_raw = pd.read_csv(vr)
        except Exception: pass
    if os.path.exists(pr):
        try: p2p_raw = pd.read_csv(pr)
        except Exception: pass
    if os.path.exists(er):
        try: emp_raw = pd.read_csv(er)
        except Exception: pass
    if os.path.exists(or_):
        try: o2c_raw = pd.read_csv(or_)
        except Exception: pass
    if os.path.exists(cr_):
        try: cust_raw = pd.read_csv(cr_)
        except Exception: pass
    if os.path.exists(ar_):
        try: att_raw = pd.read_csv(ar_)
        except Exception: pass
    return dfs, saved_proc_status, vendor_raw, p2p_raw, emp_raw, o2c_raw, cust_raw, att_raw
