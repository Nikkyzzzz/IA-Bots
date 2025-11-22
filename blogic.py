# ============================== blogic.py — Banking Mapping & Runner ==============================
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Tuple

# Import the banking-specific logics
import blogic6

# ---------- Canonical field targets ----------
CANONICAL_FIELDS = {
    "Banking": {
        "Loan Dump": {
            "ASSET": "ASSET",
            "INT_RATE": "INT_RATE",
            "URI": "URI",
            "CUST_CATEGORY": "CUST_CATEGORY",
            "PROVISION": "PROVISION",
            "AMT_OS": "AMT_OS",
            "RESTRUCTURED_FLG": "RESTRUCTURED_FLG",
            "RESTR_DATE": "RESTR_DATE",
            "FB_NFB_FLG": "FB_NFB_FLG",
            "OUT_ORD_DT": "OUT_ORD_DT",
            "DRAW_LMT": "DRAW_LMT",
            "SECTOR": "SECTOR",
            "SANC_LMT": "SANC_LMT",
            "PIN CODE": "PIN_CODE",   # ✅ added if blacklist uploaded
        }
    },
    "Blacklisted PIN CODE": {
        "Blacklisted PIN CODE": {
            "PIN CODE": "PIN CODE",
        }
    },
    "Loan Book (31.03.2025)": {
        "Loan Book (31.03.2025)": {
            "PROJECT NO": "PROJECT NO",
            "LOAN OUTSTANDING (Rs.)": "LOAN OUTSTANDING (Rs.)",
            "Asset classification": "Asset classification",
        }
    },
    "Loan Book (30.06.2025)": {
        "Loan Book (30.06.2025)": {
            "PROJECT NO": "PROJECT NO",
            "LOAN OUTSTANDING (Rs.)": "LOAN OUTSTANDING (Rs.)",
            "Asset classification": "Asset classification",
        }
    },
}

def _build_rename_map(required_to_actual: Dict[str, str], req_to_canon: Dict[str, str]) -> Dict[str, str]:
    rename = {}
    for req, actual in required_to_actual.items():
        if actual and req in req_to_canon:
            rename[actual] = req_to_canon[req]
    return rename

def _read_sheet_from(bytes_blob: bytes, sheet_name: str, header_row: int = 0) -> pd.DataFrame:
    bio = BytesIO(bytes_blob)
    return pd.read_excel(bio, sheet_name=sheet_name, header=header_row)

# ---------- DataFrame preparation ----------
def prepare_dataframe_for_cat(
    cat: str,
    file_bytes_map: Dict[str, bytes],
    sheet_mapping_pairs: Dict[str, Dict[str, str]],
    column_mapping_pairs: Dict[str, Dict[str, Dict[str, str]]],
) -> pd.DataFrame:
    if cat not in sheet_mapping_pairs:
        return None
    file_bytes = file_bytes_map.get(cat)
    if not file_bytes:
        return None
    mapping = sheet_mapping_pairs[cat]
    fields_map = column_mapping_pairs.get(cat, {})

    # For each required sheet
    for req_sheet, mapped_sheet in mapping.items():
        sheet = mapped_sheet
        if not sheet:
            continue
        header_row = 1 if cat == "Loan Book (31.03.2025)" else (2 if cat == "Loan Book (30.06.2025)" else 0)
        df = _read_sheet_from(file_bytes, sheet, header_row=header_row)
        rename = _build_rename_map(fields_map.get(req_sheet, {}), CANONICAL_FIELDS.get(cat, {}).get(req_sheet, {}))
        if rename:
            df = df.rename(columns=rename)
        # Ensure canonical columns exist
        for col in CANONICAL_FIELDS.get(cat, {}).get(req_sheet, {}).values():
            if col not in df.columns:
                df[col] = np.nan
        return df
    return None

# ---------- Runner ----------
def run_all_bots_with_mappings(
    file_bytes_map: Dict[str, bytes],
    sheet_mapping_pairs: Dict[str, Dict[str, str]],
    column_mapping_pairs: Dict[str, Dict[str, Dict[str, str]]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, pd.DataFrame]]:
    results: Dict[str, pd.DataFrame] = {}
    proc_status: Dict[str, str] = {}
    raw_dfs: Dict[str, pd.DataFrame] = {}

    # --- CCIS / Banking bots ---
    df_banking = prepare_dataframe_for_cat("Banking", file_bytes_map, sheet_mapping_pairs, column_mapping_pairs)
    if df_banking is not None:
        raw_dfs["BANKING_RAW"] = df_banking
        # Set input_row_count in session state for use in b7.py Output tab
        try:
            import streamlit as st
            st.session_state["input_row_count"] = len(df_banking)
        except Exception:
            pass
        bot_map = {
            "zero_or_null_roi_loans": blogic6.zero_or_null_roi_loans,
            "standard_accounts_with_uri_zero": blogic6.standard_accounts_with_uri_zero,
            "provision_verification_substandard_npa": blogic6.provision_verification_substandard_npa,
            "restructured_standard_accounts": blogic6.restructured_standard_accounts,
            "provision_verification_doubtful3_npa": blogic6.provision_verification_doubtful3_npa,
            "npa_fb_accounts_overdue": blogic6.npa_fb_accounts_overdue,
            "negative_amt_outstanding": blogic6.negative_amt_outstanding,
            "standard_accounts_overdue_details": blogic6.standard_accounts_overdue_details,
            "standard_accounts_with_odd_interest": blogic6.standard_accounts_with_odd_interest,
            "agri0_sector_over_limit": blogic6.agri0_sector_over_limit,
            "misaligned_scheme_for_facilities": blogic6.misaligned_scheme_for_facilities,
        }
        for key, fn in bot_map.items():
            try:
                results[key] = fn(df_banking.copy())
                proc_status[key] = "Complete"
            except Exception:
                proc_status[key] = "Failed"

    # --- Bot 12: Loans & Advances to Blacklisted Areas ---
    # This bot uses the same main input DataFrame (df_banking) as the first 11 bots,
    # along with the Blacklist input. Its total input row count should match the others.
    df_blacklist = prepare_dataframe_for_cat("Blacklisted PIN CODE", file_bytes_map, sheet_mapping_pairs, column_mapping_pairs)
    if df_banking is not None and df_blacklist is not None:
        try:
            results["Loans & Advances to Blacklisted Areas"] = blogic6.match_pincode(df_banking.copy(), df_blacklist.copy())
            proc_status["Loans & Advances to Blacklisted Areas"] = "Complete"
        except Exception:
            proc_status["Loans & Advances to Blacklisted Areas"] = "Failed"
        raw_dfs["BLACKLIST_RAW"] = df_blacklist

    # --- Loan Book bots (requires both Mar + Jun) ---
    df_loan_mar = prepare_dataframe_for_cat("Loan Book (31.03.2025)", file_bytes_map, sheet_mapping_pairs, column_mapping_pairs)
    df_loan_jun = prepare_dataframe_for_cat("Loan Book (30.06.2025)", file_bytes_map, sheet_mapping_pairs, column_mapping_pairs)
    if df_loan_mar is not None and df_loan_jun is not None:
        try:
            results["Blank Asset Classification"] = blogic6.merge_and_blank_asset_classification(df_loan_mar.copy(), df_loan_jun.copy())
            proc_status["Blank Asset Classification"] = "Complete"
        except Exception:
            proc_status["Blank Asset Classification"] = "Failed"
        raw_dfs["LOAN_MAR_RAW"] = df_loan_mar
        raw_dfs["LOAN_JUN_RAW"] = df_loan_jun

    return results, proc_status, raw_dfs
