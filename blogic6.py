# ============================== blogic6.py — Banking Bot Logics ==============================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------- 1. Zero or Null ROI Loans ---------------- # New
def zero_or_null_roi_loans(df: pd.DataFrame) -> pd.DataFrame:
    df1 = df.copy()
    df1['ASSET'] = df1['ASSET'].astype(str)
    df1['INT_RATE'] = df1['INT_RATE'].astype(str)
    return df1[
        (df1["ASSET"].isin(["11", "12"])) &
        (df1["INT_RATE"].isin(["0", "-"]))
    ]

# ---------------- 2. Standard Accounts with URI Zero ---------------- # Same
def standard_accounts_with_uri_zero(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2['ASSET'] = df2['ASSET'].astype(str)
    df2["URI"] = pd.to_numeric(df2["URI"], errors="coerce")
    return df2[(df2["ASSET"].isin(["11", "12"])) & (df2["URI"] == 0)]

# ---------------- 3. Provision Verification for Sub-Standard NPA ---------------- # New
def provision_verification_substandard_npa(df: pd.DataFrame) -> pd.DataFrame:
    df3 = df.copy()
    df3['ASSET'] = df3['ASSET'].astype(str)
    df3['CUST_CATEGORY'] = df3['CUST_CATEGORY'].astype(str)
    df3["PROVISION"] = pd.to_numeric(df3["PROVISION"], errors="coerce")
    df3["AMT_OS"] = pd.to_numeric(df3["AMT_OS"], errors="coerce")
    return df3[
        (df3["CUST_CATEGORY"] == "NPA") &
        (df3["ASSET"].isin(["21", "22"])) &
        ((df3["PROVISION"] * 100 / df3["AMT_OS"]) < 15)
    ]

# ---------------- 4. Restructured Standard Accounts ---------------- # Old but new insight
def restructured_standard_accounts(df: pd.DataFrame) -> pd.DataFrame:
    df4 = df.copy()
    df4['ASSET'] = df4['ASSET'].astype(str)
    df4['RESTRUCTURED_FLG'] = df4['RESTRUCTURED_FLG'].astype(str)
    df4["AMT_OS"] = pd.to_numeric(df4["AMT_OS"], errors="coerce")
    df4["PROVISION"] = pd.to_numeric(df4["PROVISION"], errors="coerce")
    df4["RESTR_DATE"] = pd.to_datetime(df4["RESTR_DATE"], errors="coerce")
    two_years_ago = datetime.today() - timedelta(days=730)

    return df4[
        (df4["RESTRUCTURED_FLG"] == "Y") &
        (df4["ASSET"].isin(["11", "12"])) &
        (df4["AMT_OS"] != 0) &
        ((df4["PROVISION"] * 100 / df4["AMT_OS"]) < 15) &
        (df4["RESTR_DATE"] > two_years_ago)
    ]

# ---------------- 5. Provision Verification for Doubtful-3 NPA ---------------- # same as previous
def provision_verification_doubtful3_npa(df: pd.DataFrame) -> pd.DataFrame:
    df5 = df.copy()
    df5['ASSET'] = df5['ASSET'].astype(str)
    df5['CUST_CATEGORY'] = df5['CUST_CATEGORY'].astype(str)
    df5["PROVISION"] = pd.to_numeric(df5["PROVISION"], errors="coerce")
    df5["AMT_OS"] = pd.to_numeric(df5["AMT_OS"], errors="coerce")
    return df5[
        (df5["CUST_CATEGORY"] == "NPA") &
        (df5["ASSET"].isin(["31", "32", "33"])) &
        (df5["PROVISION"] != df5["AMT_OS"])
    ]

# ---------------- 6. NPA FB Accounts with Overdue Flags ---------------- # old but new insight
def npa_fb_accounts_overdue(df: pd.DataFrame) -> pd.DataFrame:
    df6 = df.copy()
    df6['FB_NFB_FLG'] = df6['FB_NFB_FLG'].astype(str)
    df6["OUT_ORD_DT"] = pd.to_datetime(df6["OUT_ORD_DT"], errors="coerce")
    three_months_ago = datetime.today() - timedelta(days=90)
    return df6[
        (df6["FB_NFB_FLG"] == "FB") &
        (df6["OUT_ORD_DT"] <= three_months_ago)
    ]

# ---------------- 7. Negative Amount Outstanding ---------------- # same as previous    (Requires Blacklisted Pin codes)
def negative_amt_outstanding(df: pd.DataFrame) -> pd.DataFrame:
    df7 = df.copy()
    df7["AMT_OS"] = pd.to_numeric(df7["AMT_OS"], errors="coerce")
    return df7[df7["AMT_OS"] < 0]

# ---------------- 8. Standard Accounts Overdue Details ---------------- # same but new insight (1 step removed)
def standard_accounts_overdue_details(df: pd.DataFrame) -> pd.DataFrame:
    df8 = df.copy()
    df8["AMT_OS"] = pd.to_numeric(df8["AMT_OS"], errors="coerce")
    df8["DRAW_LMT"] = pd.to_numeric(df8["DRAW_LMT"], errors="coerce")
    return df8[
        (df8["AMT_OS"] - df8["DRAW_LMT"]) > (0.1 * df8["DRAW_LMT"])
    ]

# ---------------- 9. Standard Accounts with Odd Interest Rates ---------------- # same but new insight (+1 check added)
def standard_accounts_with_odd_interest(df: pd.DataFrame) -> pd.DataFrame:
    df9 = df.copy()
    df9['ASSET'] = df9['ASSET'].astype(str)
    df9["INT_RATE"] = pd.to_numeric(df9["INT_RATE"], errors="coerce")
    return df9[
        (df9["ASSET"].isin(["11", "12"])) &
        (~df9["INT_RATE"].isna()) &
        ((df9["INT_RATE"] % 0.05) != 0)
    ]

# ---------------- 10. Agri0 Sector Over Limit ---------------- # same
def agri0_sector_over_limit(df: pd.DataFrame) -> pd.DataFrame:
    df10 = df.copy()
    df10['ASSET'] = df10['ASSET'].astype(str)
    df10['SECTOR'] = df10['SECTOR'].astype(str)
    df10["AMT_OS"] = pd.to_numeric(df10["AMT_OS"], errors="coerce")
    df10["SANC_LMT"] = pd.to_numeric(df10["SANC_LMT"], errors="coerce")
    return df10[
        (df10["ASSET"].isin(["11", "12"])) &
        (df10["SECTOR"] == "01.Agri") &
        (df10["AMT_OS"] > (1.34 * df10["SANC_LMT"]))
    ]

# ---------------- Logic 11 ---------------- # from previous version
def misaligned_scheme_for_facilities(df: pd.DataFrame) -> pd.DataFrame:
    df3 = df.copy()
    mode_map = df3.groupby('FACILITYCD')['SCHEME_CD'].agg(lambda x: x.mode().iloc[0])
    df3['MAJ_SCHEME'] = df3['FACILITYCD'].map(mode_map)
    return df3[df3['SCHEME_CD'] != df3['MAJ_SCHEME']].drop(columns=['MAJ_SCHEME'])


import pandas as pd

# ==============================================================
# New Bot: Merge Loan Books and Blank Asset Classification
# ==============================================================
def merge_and_blank_asset_classification(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two DataFrames (Loan Book 31.03.2025 and Loan Book 30.06.2025) on 'PROJECT NO',
    calculates the difference in 'LOAN OUTSTANDING (Rs.)', and blanks out 'Asset classification'
    columns where the difference is zero and classifications match.

    Args:
        df1 (pd.DataFrame): Loan Book (31.03.2025)
        df2 (pd.DataFrame): Loan Book (30.06.2025)

    Returns:
        pd.DataFrame: Merged DataFrame with 'Difference' and updated 'Asset classification' columns.
    """
    merged_df = pd.merge(
        df1,
        df2,
        on="PROJECT NO",
        how="inner",
        suffixes=("_File1", "_File2")
    )

    # Calculate difference
    merged_df["Difference"] = (
        merged_df["LOAN OUTSTANDING (Rs.)_File1"].fillna(0)
        - merged_df["LOAN OUTSTANDING (Rs.)_File2"].fillna(0)
    )

    # Blank classification where difference = 0 and classifications match
    condition = (
        (merged_df["Difference"] == 0)
        & (merged_df["Asset classification_File1"] == merged_df["Asset classification_File2"])
    )

    merged_df.loc[condition, ["Asset classification_File1", "Asset classification_File2"]] = ""

    return merged_df


# ==============================================================
# New Bot: Match PIN Codes between CCIS and Blacklisted PIN file
# ==============================================================
def match_pincode(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Filters rows from the first DataFrame (CCIS) where 'PIN_CODE' matches
    any PIN from the second DataFrame (Blacklisted PIN list).

    Args:
        df1 (pd.DataFrame): CCIS file containing 'PIN_CODE'
        df2 (pd.DataFrame): Blacklisted PIN file containing 'PIN CODE'

    Returns:
        pd.DataFrame: Rows from df1 where PIN_CODE is blacklisted.
    """
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Standardize column values
    df1_copy["PIN_CODE"] = df1_copy["PIN_CODE"].astype(str).str.strip()
    df2_copy["PIN CODE"] = df2_copy["PIN CODE"].astype(str).str.strip()

    # Unique blacklist set
    pin_set = set(df2_copy["PIN CODE"].unique())

    # Filter CCIS by blacklist
    return df1_copy[df1_copy["PIN_CODE"].isin(pin_set)]


# ---------------- Bot Titles for UI ---------------- #
PROCESS_TITLES = {
    # Base CCIS bots
    "zero_or_null_roi_loans": ("zero_or_null_roi_loans", "Zero or Null ROI Loans"),
    "standard_accounts_with_uri_zero": ("standard_accounts_with_uri_zero", "Standard Accounts with URI Zero"),
    "provision_verification_substandard_npa": ("provision_verification_substandard_npa", "Provision Verification (Sub-Standard NPA)"),
    "restructured_standard_accounts": ("restructured_standard_accounts", "Restructured Standard Accounts"),
    "provision_verification_doubtful3_npa": ("provision_verification_doubtful3_npa", "Provision Verification (Doubtful-3 NPA)"),
    "npa_fb_accounts_overdue": ("npa_fb_accounts_overdue", "NPA FB Accounts with Overdue Flags"),
    "negative_amt_outstanding": ("negative_amt_outstanding", "Negative Amount Outstanding"),
    "standard_accounts_overdue_details": ("standard_accounts_overdue_details", "Standard Accounts Overdue Details"),
    "standard_accounts_with_odd_interest": ("standard_accounts_with_odd_interest", "Standard Accounts with Odd Interest"),
    "agri0_sector_over_limit": ("agri0_sector_over_limit", "Agri0 Sector Over Limit"),
    "misaligned_scheme_for_facilities": ("misaligned_scheme_for_facilities", "Misaligned Scheme for Facilities"),

    # Extra bots
    "Loans & Advances to Blacklisted Areas": ("match_pincode", "Loans & Advances to Blacklisted Areas"),
    "Blank Asset Classification": ("merge_and_blank_asset_classification", "Blank Asset Classification"),
}
