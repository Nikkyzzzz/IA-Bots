# logic.py
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Tuple, Any

# We import your existing logic6 (unchanged)
import logic6

# ---------- Canonical field targets that logic6 expects ----------
# Map *Required Field* → *Canonical Column Name in logic6*
CANONICAL_FIELDS = {
    "P2P": {
        "P2P Sample": {
            "Vendor Name": "Vendor_Name",
            "PO No": "PO_No",
            "PO Date": "PO_Date",
            "PO Quantity": "PO_Qty",
            "PO Amount": "PO_Amt",
            # update 17-nov start
            "PO Created By": "PO_Created_By",
            # update 17-nov completed
            "PO Approved By": "PO_Approved_By",
            "GRN No": "GRN_No",
            "GRN Date": "GRN_Date",
            "GRN Quantity": "GRN_Qty",
            "Invoice Date": "Invoice_Date",
            "Invoice Quantity": "Invoice_Qty",
            "Invoice Amount": "Invoice_Amount",
            "Creator ID": "Creator_ID",
        },
        "Vendor Master": {
            "Vendor Name": "Vendor_Name",
            "GST": "GST_No",
            "PAN": "PAN_No",
            "Bank Account": "Bank_Account",
            "Creator ID": "Creator_ID",
        },
        "Employee Master": {
            "Employee ID": "Employee_ID",
            "Employee Name": "Employee_Name",
            "Department": "Department",
            "Creator ID": "Creator_ID",
            # update 17-nov start
            "Designation": "Designation",
        },
        "Authority Matrix": {                    # NEW SHEET
            "Department": "Department",
            "Role": "Role",
            "Creation Authority": "Creation Authority",
            "Creation Amount Limit": "Creation Amount Limit",
            "Approval Authority": "Approval Authority",
            "Approval Amount Limit": "Approval Amount Limit",
        },
        # update 17-nov completed
    },
    "O2C": {
        "O2C Sample": {
            "SO Date": "SO_Date",
            "Delivery Date": "Delivery_Date",
            "Invoice No": "Invoice_No",
            # 🔽 NEW REQUIRED FIELDS FOR O2C BOTS
            "SO No": "SO_No",                 # Bot 24, 29
            "Invoice Amount": "Invoice_Amount",   # Bot 32, 39
            "Taxable Amount": "Taxable_Amount",   # Bot 39
            # Delivery_No is used by logic6.check_dispatch_without_invoice;
            # we'll create it empty if not provided.
            # Delivery_No is used by logic6.check_dispatch_without_invoice;
            # we'll create it empty if not provided.
            
        },
        "Customer Master": {
            "GST No": "GST_No",
            "PAN No": "PAN_No",
            "Credit Limit": "Credit_Limit",
        },
    },
    "H2R": {
        "Employee Master": {
            "Employee ID": "Employee_ID",
            "Employee Name": "Employee_Name",
            "Exit Date": "Exit_Date",
            "Status": "Status",
            "Bank Account": "Bank_Account",  # NEW
            "PAN No": "PAN_No",              # NEW
        },
        "Attendance Register": {
            "Employee ID": "Employee_ID",
            "Employee Name": "Employee_Name",
            "Month": "Month",
            # Daily D1..D31 columns remain as-is if present
        },
    },
}

def _build_rename_map(required_to_actual: Dict[str, str], req_to_canon: Dict[str, str]) -> Dict[str, str]:
    """
    Convert {Required Field -> Actual Column} to {Actual Column -> Canonical Column}.
    """
    rename = {}
    for req, actual in required_to_actual.items():
        if actual and req in req_to_canon:
            rename[actual] = req_to_canon[req]
    return rename

def _read_sheet_from(bytes_blob: bytes, sheet_name: str) -> pd.DataFrame:
    bio = BytesIO(bytes_blob)
    return pd.read_excel(bio, sheet_name=sheet_name)

def prepare_dataframes(
    file_bytes_map: Dict[str, bytes],
    sheet_mapping_pairs: Dict[str, Dict[str, str]],
    column_mapping_pairs: Dict[str, Dict[str, Dict[str, str]]],
    # Update 17-nov start
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Update 17-nov completed
    """
    Returns: (df_vendor, df_p2p, df_emp_p2p, df_auth_p2p, df_o2c, df_cust, df_emp_h2r, df_att)
    Missing ones can be None.
    """
    # Helper: choose bytes per category (prefer category, else master)
    def _cat_bytes(cat: str):
        if cat == "P2P":
            return file_bytes_map.get("P2P") or file_bytes_map.get("MASTER")
        if cat == "O2C":
            return file_bytes_map.get("O2C") or file_bytes_map.get("MASTER")
        if cat == "H2R":
            return file_bytes_map.get("H2R") or file_bytes_map.get("MASTER")
        return None

    # Prepare containers
    df_vendor = df_p2p = df_emp_p2p = df_o2c = df_cust = df_att = None
    # Update 17-nov start
    df_auth_p2p = None          # NEW
    # Update 17-nov completed

    # ---------- P2P ----------
    if "P2P" in sheet_mapping_pairs:
        p2p_bytes = _cat_bytes("P2P")
        if p2p_bytes:
            mapping = sheet_mapping_pairs["P2P"]
            fields_map = column_mapping_pairs.get("P2P", {})

            # P2P Sample
            if "P2P Sample" in mapping:
                sheet = mapping["P2P Sample"]
                df = _read_sheet_from(p2p_bytes, sheet)
                rename = _build_rename_map(fields_map.get("P2P Sample", {}), CANONICAL_FIELDS["P2P"]["P2P Sample"])
                if rename:
                    df = df.rename(columns=rename)
                # Fill any missing canonical columns used by logic6
                for c in [
                    "PO_No", "PO_Qty", "PO_Amt",
                    "GRN_Qty",
                    "Invoice_Qty", "Invoice_Amount",
                    "Vendor_Name", "PO_Date", "Invoice_Date",
                    "PO_Approved_By", "PO_Created_By", "Item_Code"
                ]:
                    if c not in df.columns:
                        df[c] = np.nan
                df_p2p = df

            # Vendor Master
            if "Vendor Master" in mapping:
                sheet = mapping["Vendor Master"]
                df = _read_sheet_from(p2p_bytes, sheet)
                rename = _build_rename_map(fields_map.get("Vendor Master", {}), CANONICAL_FIELDS["P2P"]["Vendor Master"])
                if rename:
                    df = df.rename(columns=rename)
                for c in ["PAN_No", "GST_No", "Bank_Account", "Vendor_Name", "Creator_ID"]:
                    if c not in df.columns:
                        df[c] = np.nan
                df_vendor = df

            # Employee Master (for department lookups / approvals)
            if "Employee Master" in mapping:
                sheet = mapping["Employee Master"]
                df = _read_sheet_from(p2p_bytes, sheet)
                rename = _build_rename_map(fields_map.get("Employee Master", {}), CANONICAL_FIELDS["P2P"]["Employee Master"])
                if rename:
                    df = df.rename(columns=rename)
                df_emp_p2p = df

            # --- Authority Matrix (P2P) ---
            if "Authority Matrix" in mapping:
                sheet = mapping["Authority Matrix"]
                if sheet:
                    df = _read_sheet_from(p2p_bytes, sheet)
                    rename = _build_rename_map(
                        fields_map.get("Authority Matrix", {}),
                        CANONICAL_FIELDS["P2P"]["Authority Matrix"],
                    )
                    if rename:
                        df = df.rename(columns=rename)
                    df_auth_p2p = df

    # ---------- O2C ----------
    if "O2C" in sheet_mapping_pairs:
        o2c_bytes = _cat_bytes("O2C")
        if o2c_bytes:
            mapping = sheet_mapping_pairs["O2C"]
            fields_map = column_mapping_pairs.get("O2C", {})

            if "O2C Sample" in mapping:
                sheet = mapping["O2C Sample"]
                df = _read_sheet_from(o2c_bytes, sheet)
                rename = _build_rename_map(
                    fields_map.get("O2C Sample", {}),
                    CANONICAL_FIELDS["O2C"]["O2C Sample"],
                )
                if rename:
                    df = df.rename(columns=rename)

                # 🔽 Make sure the canonical columns used by the new bots exist
                for c in ["SO_No", "Invoice_No", "Invoice_Amount", "Taxable_Amount", "Delivery_No"]:
                    if c not in df.columns:
                        df[c] = np.nan

                df_o2c = df

            if "Customer Master" in mapping:
                sheet = mapping["Customer Master"]
                df = _read_sheet_from(o2c_bytes, sheet)
                rename = _build_rename_map(
                    fields_map.get("Customer Master", {}),
                    CANONICAL_FIELDS["O2C"]["Customer Master"],
                )
                if rename:
                    df = df.rename(columns=rename)
                df_cust = df

    # ---------- H2R ----------
    if "H2R" in sheet_mapping_pairs:
        h2r_bytes = _cat_bytes("H2R")
        if h2r_bytes:
            mapping = sheet_mapping_pairs["H2R"]
            fields_map = column_mapping_pairs.get("H2R", {})

            if "Employee Master" in mapping:
                df = _read_sheet_from(h2r_bytes, mapping["Employee Master"])
                rename = _build_rename_map(fields_map.get("Employee Master", {}), CANONICAL_FIELDS["H2R"]["Employee Master"])
                if rename:
                    df = df.rename(columns=rename)
                if "Employee_ID" in df.columns:
                    df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip()
                df_emp_h2r = df
            else:
                df_emp_h2r = None

            if "Attendance Register" in mapping:
                df = _read_sheet_from(h2r_bytes, mapping["Attendance Register"])
                rename = _build_rename_map(fields_map.get("Attendance Register", {}), CANONICAL_FIELDS["H2R"]["Attendance Register"])
                if rename:
                    df = df.rename(columns=rename)
                # Derive Present_Days if day columns exist
                day_cols = [c for c in df.columns if str(c).startswith("D") and str(c)[1:].isdigit()]
                if day_cols:
                    s = df[day_cols].astype(str).apply(lambda col: col.str.strip().str.upper())
                    present = (~s.eq("A")).sum(axis=1)
                    df["Present_Days"] = present
                else:
                    if "Present_Days" not in df.columns:
                        df["Present_Days"] = 0
                if "Employee_ID" in df.columns:
                    df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip()
                df_att = df
            else:
                df_att = None

            # Return H2R pair as df_emp (master) and df_attendance
            return df_vendor, df_p2p, df_emp_p2p, df_auth_p2p, df_o2c, df_cust, (df_emp_h2r if 'df_emp_h2r' in locals() else None), df_att

    # If H2R not present, return 8-tuple with last two as None
    return df_vendor, df_p2p, df_emp_p2p, df_auth_p2p, df_o2c, df_cust, None, None



def run_all_bots_with_mappings(
    file_bytes_map: Dict[str, bytes],
    sheet_mapping_pairs: Dict[str, Dict[str, str]],
    column_mapping_pairs: Dict[str, Dict[str, Dict[str, str]]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, str], Dict[str, pd.DataFrame]]:
    """
    Executes all bots using mapped columns. Returns:
    - results: dict code -> DataFrame
    - proc_status: dict code -> status
    - cat_status: dict category -> status
    - raw_dfs: dict of raw dfs used (for next-level analytics compatibility)
    """
    # Prepare dataframes
    df_vendor, df_p2p, df_emp_p2p, df_auth_p2p, df_o2c, df_cust, df_emp_h2r, df_att = prepare_dataframes(
        file_bytes_map, sheet_mapping_pairs, column_mapping_pairs
    )

    results: Dict[str, pd.DataFrame] = {}

    # Initialise process statuses (now including new P2P bots)
    proc_status = {
        # P2P
        'P2P1': 'Pending',
        'P2P2': 'Pending',
        'P2P3': 'Pending',
        'P2P4': 'Pending',
        'P2P5': 'Pending',
        'P2P6': 'Pending',   # GSTIN validation
        'P2P8': 'Pending',   # PO Approval Bypass
        'P2P10': 'Pending',  # Excessive Emergency
        'P2P13': 'Pending',  # Invoice vs GRN gap
        'P2P17': 'Pending',  # Invoices to inactive vendors
        'P2P20': 'Pending',  # Round-sum invoices

        # O2C (old + NEW)
        'O2C1': 'Pending',
        'O2C2': 'Pending',
        'O2C3': 'Pending',
        'O2C24': 'Pending',  # Sales without SO
        'O2C29': 'Pending',  # Multiple invoices per SO
        'O2C32': 'Pending',  # Zero / missing invoice amount
        'O2C39': 'Pending',  # Excessive small-value sales

        # H2R (old + NEW)
        'H2R1': 'Pending',
        'H2R2': 'Pending',
        'H2R44': 'Pending',  # Duplicate employees
    }

    # ---------- P2P ----------

    # Vendor-based bots (P2P1, P2P5, P2P6)
    if df_vendor is not None:
        try:
            results['P2P1'] = logic6.find_missing_vendor_fields(df_vendor)
            proc_status['P2P1'] = 'Complete'
        except Exception:
            proc_status['P2P1'] = 'Failed'

        try:
            # Duplicate Vendors — run on DF (no workbook dependency)
            results['P2P5'] = _find_matching_rows_from_df(df_vendor)
            proc_status['P2P5'] = 'Complete'
        except Exception:
            proc_status['P2P5'] = 'Failed'

        # P2P6 – Vendor GSTIN Validation (on Vendor Master)
        try:
            if hasattr(logic6, "validate_gstin"):
                results['P2P6'] = logic6.validate_gstin(df_vendor.copy())
                proc_status['P2P6'] = 'Complete'
            else:
                proc_status['P2P6'] = 'Failed'
        except Exception:
            proc_status['P2P6'] = 'Failed'

    # P2P Sample based bots
    if df_p2p is not None:
        # Existing bots
        try:
            results['P2P2'] = logic6.find_po_grn_invoice_mismatches(df_p2p.copy())
            proc_status['P2P2'] = 'Complete'
        except Exception:
            proc_status['P2P2'] = 'Failed'

        try:
            results['P2P3'] = logic6.get_invalid_rows(df_p2p.copy())
            proc_status['P2P3'] = 'Complete'
        except Exception:
            proc_status['P2P3'] = 'Failed'

        try:
            results['P2P4'] = logic6.generate_result(df_p2p.copy())
            proc_status['P2P4'] = 'Complete'
        except Exception:
            proc_status['P2P4'] = 'Failed'

        # P2P8 – PO Approval Bypass
        try:
            if (df_auth_p2p is not None) and (df_emp_p2p is not None) and hasattr(logic6, "check_po_approval_bypass"):
                results['P2P8'] = logic6.check_po_approval_bypass(
                    df_p2p.copy(), df_auth_p2p.copy(), df_emp_p2p.copy()
                )
                proc_status['P2P8'] = 'Complete'
            else:
                proc_status['P2P8'] = 'Failed'
        except Exception:
            proc_status['P2P8'] = 'Failed'

        # P2P10 – Excessive Emergency Purchases
        try:
            if hasattr(logic6, "excessive_emergency_purchases"):
                results['P2P10'] = logic6.excessive_emergency_purchases(df_p2p.copy(), threshold=0.10)
                proc_status['P2P10'] = 'Complete'
            else:
                proc_status['P2P10'] = 'Failed'
        except Exception:
            proc_status['P2P10'] = 'Failed'

        # 🔹 NEW: P2P11 – Vendor Concentration Risk
        try:
            if hasattr(logic6, "bot_11_vendor_concentration"):
                results['P2P11'] = logic6.bot_11_vendor_concentration(df_p2p.copy())
                proc_status['P2P11'] = 'Complete'
            else:
                proc_status['P2P11'] = 'Failed'
        except Exception:
            proc_status['P2P11'] = 'Failed'

        # P2P13 – Invoice vs GRN Date Gap
        try:
            if hasattr(logic6, "check_invoice_vs_grn"):
                results['P2P13'] = logic6.check_invoice_vs_grn(df_p2p.copy(), days_threshold=10)
                proc_status['P2P13'] = 'Complete'
            else:
                proc_status['P2P13'] = 'Failed'
        except Exception:
            proc_status['P2P13'] = 'Failed'

        # 🔹 NEW: P2P14 – Over Receipt
        try:
            if hasattr(logic6, "bot_14_over_receipt"):
                results['P2P14'] = logic6.bot_14_over_receipt(df_p2p.copy())
                proc_status['P2P14'] = 'Complete'
            else:
                proc_status['P2P14'] = 'Failed'
        except Exception:
            proc_status['P2P14'] = 'Failed'

        # 🔹 NEW: P2P15 – Payment Term Adherence
        try:
            if hasattr(logic6, "bot_15_payment_terms"):
                results['P2P15'] = logic6.bot_15_payment_terms(df_p2p.copy())
                proc_status['P2P15'] = 'Complete'
            else:
                proc_status['P2P15'] = 'Failed'
        except Exception:
            proc_status['P2P15'] = 'Failed'

        # 🔹 NEW: P2P16 – Duplicate Invoice Detection
        try:
            if hasattr(logic6, "bot_16_duplicate_invoice"):
                results['P2P16'] = logic6.bot_16_duplicate_invoice(df_p2p.copy())
                proc_status['P2P16'] = 'Complete'
            else:
                proc_status['P2P16'] = 'Failed'
        except Exception:
            proc_status['P2P16'] = 'Failed'

        # P2P17 – Invoices to Inactive Vendors (needs Vendor Master)
        try:
            if (df_vendor is not None) and hasattr(logic6, "check_invoice_to_inactive_vendor"):
                results['P2P17'] = logic6.check_invoice_to_inactive_vendor(df_p2p.copy(), df_vendor.copy())
                proc_status['P2P17'] = 'Complete'
            else:
                proc_status['P2P17'] = 'Failed'
        except Exception:
            proc_status['P2P17'] = 'Failed'

        # 🔹 NEW: P2P18 – Non-PO Invoices
        try:
            if hasattr(logic6, "bot_18_non_po_invoices"):
                results['P2P18'] = logic6.bot_18_non_po_invoices(df_p2p.copy())
                proc_status['P2P18'] = 'Complete'
            else:
                proc_status['P2P18'] = 'Failed'
        except Exception:
            proc_status['P2P18'] = 'Failed'

        # P2P20 – Round-Sum Invoices
        try:
            if hasattr(logic6, "check_round_sum_invoices"):
                results['P2P20'] = logic6.check_round_sum_invoices(df_p2p.copy())
                proc_status['P2P20'] = 'Complete'
            else:
                proc_status['P2P20'] = 'Failed'
        except Exception:
            proc_status['P2P20'] = 'Failed'

    # ---------- O2C ----------
    
    if df_o2c is not None:
        try:
            results['O2C1'] = logic6.check_overdue_delivery(df_o2c.copy())
            proc_status['O2C1'] = 'Complete'
        except Exception:
            proc_status['O2C1'] = 'Failed'

        try:
            # ensure columns exist
            d = df_o2c.copy()
            if "Delivery_No" not in d.columns:
                d["Delivery_No"] = np.nan
            results['O2C2'] = logic6.check_dispatch_without_invoice(d)
            proc_status['O2C2'] = 'Complete'
        except Exception:
            proc_status['O2C2'] = 'Failed'

        # 🔽 NEW: O2C24 – Sales without SO
        try:
            if hasattr(logic6, "bot24_sales_without_so"):
                results["O2C24"] = logic6.bot24_sales_without_so(df_o2c.copy())
                proc_status["O2C24"] = "Complete"
            else:
                proc_status["O2C24"] = "Failed"
        except Exception:
            proc_status["O2C24"] = "Failed"

        # 🔽 NEW: O2C29 – Multiple Invoices per SO
        try:
            if hasattr(logic6, "bot29_multiple_invoices_for_so"):
                results["O2C29"] = logic6.bot29_multiple_invoices_for_so(df_o2c.copy())
                proc_status["O2C29"] = "Complete"
            else:
                proc_status["O2C29"] = "Failed"
        except Exception:
            proc_status["O2C29"] = "Failed"

        # 🔽 NEW: O2C32 – Zero / Missing Invoice Amount
        try:
            if hasattr(logic6, "bot32_zero_or_missing_invoice_amount"):
                results["O2C32"] = logic6.bot32_zero_or_missing_invoice_amount(df_o2c.copy())
                proc_status["O2C32"] = "Complete"
            else:
                proc_status["O2C32"] = "Failed"
        except Exception:
            proc_status["O2C32"] = "Failed"

        # 🔽 NEW: O2C39 – Excessive Small-value Sales
        try:
            if hasattr(logic6, "bot39_excessive_small_value_sales"):
                results["O2C39"] = logic6.bot39_excessive_small_value_sales(
                    df_o2c.copy(),
                    amount_threshold=10000,
                    percent_threshold=0.10,
                )
                proc_status["O2C39"] = "Complete"
            else:
                proc_status["O2C39"] = "Failed"
        except Exception:
            proc_status["O2C39"] = "Failed"


    if df_cust is not None:
        try:
            results['O2C3'] = logic6.get_missing_customer_data(df_cust.copy())
            proc_status['O2C3'] = 'Complete'
        except Exception:
            proc_status['O2C3'] = 'Failed'

    # ---------- H2R ----------
    if (df_emp_h2r is not None) and (df_att is not None):
        try:
            results['H2R1'] = logic6.find_ghost_employees(df_emp_h2r, df_att)
            proc_status['H2R1'] = 'Complete'
        except Exception:
            proc_status['H2R1'] = 'Failed'

        try:
            # Build a temp in-memory workbook for the function that expects file bytes
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
                df_emp_h2r.to_excel(writer, sheet_name="Employee_Master", index=False)
                df_att.to_excel(writer, sheet_name="Attendance_Register", index=False)
            bio.seek(0)
            results['H2R2'] = logic6.find_attendance_after_exit(
                bio,
                employee_sheet="Employee_Master",
                attendance_sheet="Attendance_Register",
                month_col="Month",
                year_col=None,
            )
            proc_status['H2R2'] = 'Complete'
        except Exception:
            proc_status['H2R2'] = 'Failed'

        # 🔽 NEW: H2R44 – Duplicate Employees
        try:
            if hasattr(logic6, "bot44_duplicate_employees"):
                results["H2R44"] = logic6.bot44_duplicate_employees(df_emp_h2r.copy())
                proc_status["H2R44"] = "Complete"
            else:
                proc_status["H2R44"] = "Failed"
        except Exception:
            proc_status["H2R44"] = "Failed"


    # ---------- Category rollups ----------
    cat_status = {
        'P2P': logic6.compute_category_status(
            proc_status,
            [
                'P2P1', 'P2P2', 'P2P3', 'P2P4', 'P2P5', 'P2P6',
                'P2P8', 'P2P10', 'P2P11', 'P2P13', 'P2P14',
                'P2P15', 'P2P16', 'P2P17', 'P2P18', 'P2P20'
            ]
        ),
        'O2C': logic6.compute_category_status(proc_status, ['O2C1', 'O2C2', 'O2C3', 'O2C24', 'O2C29', 'O2C32', 'O2C39']),
        'H2R': logic6.compute_category_status(proc_status, ['H2R1', 'H2R2', 'H2R44']),
    }

    # Raw dfs for optional next-level tabs
    raw_dfs = {
        "VENDOR_RAW": df_vendor,
        "P2P_RAW": df_p2p,
        "EMP_RAW": df_emp_p2p,
        "AUTH_MATRIX_RAW": df_auth_p2p,
        "O2C_RAW": df_o2c,
        "CUST_RAW": df_cust,
        "ATT_RAW": df_att,
    }
    return results, proc_status, cat_status, raw_dfs



def _find_matching_rows_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate Vendors (logic6.find_matching_rows) but working directly on a DF that
    already uses canonical columns (Vendor_Name, PAN_No, GST_No, Bank_Account).
    """
    from itertools import combinations

    d = df.copy().replace(r'^\s*$', pd.NA, regex=True).reset_index(drop=True)
    d.insert(0, "RowNo", d.index + 1)

    keys = {
        "PAN_No": "PAN Match",
        "GST_No": "GST Match",
        "Vendor_Name": "Vendor Name Match",
        "Bank_Account": "Bank Account Match",
    }

    pairs = {}
    for col, label in keys.items():
        if col in d.columns:
            tmp = d.dropna(subset=[col])
            groups = tmp.groupby(col, dropna=True)["RowNo"].apply(list)
            for rows in groups:
                if len(rows) >= 2:
                    for a, b in combinations(sorted(rows), 2):
                        pairs.setdefault((a, b), set()).add(label)

    rows_info = d.set_index("RowNo")
    records = []
    for (a, b), labels in sorted(pairs.items()):
        rec = {"Row_A": a, "Row_B": b, "Exception_Noted": ", ".join(sorted(labels))}
        # flatten
        ra = rows_info.loc[a].to_dict()
        rb = rows_info.loc[b].to_dict()
        for k, v in ra.items(): rec[f"A_{k}"] = v
        for k, v in rb.items(): rec[f"B_{k}"] = v
        records.append(rec)
    return pd.DataFrame(records)
