import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------
# 1️⃣ Distinct PROJECT NO by Asset Classification
# -------------------------------
def compare_project_counts_plotly(file1: str, file2: str, sheet_name=0, key=None):
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    if "Asset Classification as on 30.06.2025" in df1.columns:
        df1.rename(columns={"Asset Classification as on 30.06.2025": "Asset Classification"}, inplace=True)
    if "Asset classification" in df2.columns:
        df2.rename(columns={"Asset classification": "Asset Classification"}, inplace=True)

    df1 = df1.dropna(subset=["Asset Classification"])
    df2 = df2.dropna(subset=["Asset Classification"])

    for df in [df1, df2]:
        df["PROJECT NO"] = df["PROJECT NO"].astype(str).str.strip()
        df["Asset Classification"] = df["Asset Classification"].astype(str).str.strip()
        df.drop(df[df["Asset Classification"] == '0'].index, inplace=True)

    counts1 = df1.groupby("Asset Classification")["PROJECT NO"].nunique()
    counts2 = df2.groupby("Asset Classification")["PROJECT NO"].nunique()

    # -------------------------------
    # Commented updated section
    # comparison = pd.DataFrame({"Base Period": counts1, "Comparison Period": counts2}).fillna(0).astype(int).reset_index()
    # -------------------------------

    # ✅ Fixed consistent bar order
    categories = ["Substandard", "Standard", "LOSS", "Doubtful-3", "Doubtful-2"]  # define desired order
    comparison = pd.DataFrame({"Base Period": counts1, "Comparison Period": counts2}).reindex(categories).fillna(0).astype(int).reset_index()
    comparison_long = comparison.melt(
        id_vars="Asset Classification",
        value_vars=["Base Period", "Comparison Period"],
        var_name="Source File",
        value_name="Count"
    )

    fig = px.bar(
        comparison_long,
        x="Count",
        y="Asset Classification",
        color="Source File",
        barmode="group",
        text="Count",
        orientation="h",
        height=500,
        labels={"Count": "Count of Distinct PROJECT NO", "Asset Classification": "Asset Classification"},
        title="Loan Count by Asset Classification"
    )
    fig.update_layout(template="plotly_white", xaxis=dict(title="Loan Count"), yaxis=dict(automargin=True),
                      legend_title_text="Source File", uniformtext_minsize=8, uniformtext_mode="hide")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig


# -------------------------------
# 2️⃣ Total LOAN OUTSTANDING by Asset Classification
# -------------------------------
def compare_loan_outstanding_plotly(file1: str, file2: str, sheet_name=0, key=None):
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    if "Asset Classification as on 30.06.2025" in df1.columns:
        df1.rename(columns={"Asset Classification as on 30.06.2025": "Asset Classification"}, inplace=True)
    if "Asset classification" in df2.columns:
        df2.rename(columns={"Asset classification": "Asset Classification"}, inplace=True)

    df1 = df1.dropna(subset=["Asset Classification"])
    df2 = df2.dropna(subset=["Asset Classification"])

    for df in [df1, df2]:
        df["Asset Classification"] = df["Asset Classification"].astype(str).str.strip()
        df.drop(df[df["Asset Classification"] == '0'].index, inplace=True)

    col = "LOAN OUTSTANDING (Rs.)"
    if col not in df1.columns or col not in df2.columns:
        raise KeyError(f"Column '{col}' missing")
    df1[col] = pd.to_numeric(df1[col], errors='coerce').fillna(0)
    df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)

    sums1 = df1.groupby("Asset Classification", sort=False)[col].sum()
    sums2 = df2.groupby("Asset Classification", sort=False)[col].sum()

    # -------------------------------
    # Commented updated section
    # comparison = pd.DataFrame({"Base Period": sums1, "Comparison Period": sums2}).fillna(0).reset_index()
    # -------------------------------

    # ✅ Fixed consistent bar order
    categories = ["Substandard", "Standard", "LOSS", "Doubtful-3", "Doubtful-2"]
    comparison = pd.DataFrame({"Base Period": sums1, "Comparison Period": sums2}).reindex(categories).fillna(0).reset_index()
    comparison_long = comparison.melt(
        id_vars="Asset Classification",
        value_vars=["Base Period", "Comparison Period"],
        var_name="Source File",
        value_name="Loan Outstanding (Rs.)"
    )

    fig = px.bar(
        comparison_long,
        x="Loan Outstanding (Rs.)",
        y="Asset Classification",
        color="Source File",
        barmode="group",
        text="Loan Outstanding (Rs.)",
        orientation="h",
        height=500,
        labels={"Loan Outstanding (Rs.)": "Total Loan Outstanding (Rs.)", "Asset Classification": "Asset Classification"},
        title="Loan Amount by Asset Classification"
    )
    fig.update_layout(template="plotly_white", xaxis=dict(title="Loan Amount (Rs.)"), yaxis=dict(automargin=True),
                      legend_title_text="Source File", uniformtext_minsize=8, uniformtext_mode="hide")
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig




# -------------------------------
# 3️⃣ Distinct PROJECT NO by SMA (excluding '0')
# -------------------------------
def compare_project_counts_sma(file1: str, file2: str, sheet_name=0, key=None):
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    # Normalize column names
    if "SMA Staging as on 30.06.2025)" in df1.columns:
        df1.rename(columns={"SMA Staging as on 30.06.2025)": "SMA"}, inplace=True)
    if "SMA" in df2.columns:
        df2.rename(columns={"SMA": "SMA"}, inplace=True)

    # Drop NA for SMA
    df1 = df1.dropna(subset=["SMA"])
    df2 = df2.dropna(subset=["SMA"])

    # Clean data
    for df in [df1, df2]:
        df["PROJECT NO"] = df["PROJECT NO"].astype(str).str.strip()
        df["SMA"] = df["SMA"].astype(str).str.strip()
        df.drop(df[df["SMA"] == '0'].index, inplace=True)

    counts1 = df1.groupby("SMA", sort=False)["PROJECT NO"].nunique()
    counts2 = df2.groupby("SMA", sort=False)["PROJECT NO"].nunique()

    # -------------------------------
    # ✅ Updated section: Rename source files in charts
    # -------------------------------
    comparison = pd.DataFrame({"Base Period": counts1, "Comparison Period": counts2}).fillna(0).reset_index()

    comparison_long = comparison.melt(
        id_vars="SMA",
        value_vars=["Base Period", "Comparison Period"],
        var_name="Source File",
        value_name="Distinct Projects"
    )

    fig = px.bar(
        comparison_long,
        x="Distinct Projects",
        y="SMA",
        color="Source File",
        barmode="group",
        text="Distinct Projects",
        orientation="h",
        height=500,
        labels={"Distinct Projects": "Count of Distinct PROJECT NO", "SMA": "SMA"},
        title="Loan Count by SMA Classification"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Loan Count"),
        yaxis=dict(title="SMA Classifications", automargin=True),  # updated
        legend_title_text="Source File",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig



# -------------------------------
# 4️⃣ Total LOAN OUTSTANDING by SMA (excluding '0')
# -------------------------------
def compare_loan_outstanding_sma(file1: str, file2: str, sheet_name=0, key=None):
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    # Normalize column names
    if "SMA Staging as on 30.06.2025)" in df1.columns:
        df1.rename(columns={"SMA Staging as on 30.06.2025)": "SMA"}, inplace=True)
    if "SMA" in df2.columns:
        df2.rename(columns={"SMA": "SMA"}, inplace=True)

    # Drop NA for SMA
    df1 = df1.dropna(subset=["SMA"])
    df2 = df2.dropna(subset=["SMA"])

    # Clean data
    for df in [df1, df2]:
        df["SMA"] = df["SMA"].astype(str).str.strip()
        df.drop(df[df["SMA"] == '0'].index, inplace=True)

    # Ensure LOAN OUTSTANDING column exists
    col = "LOAN OUTSTANDING (Rs.)"
    if col not in df1.columns or col not in df2.columns:
        raise KeyError(f"Column '{col}' missing")
    df1[col] = pd.to_numeric(df1[col], errors='coerce').fillna(0)
    df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)

    sums1 = df1.groupby("SMA", sort=False)[col].sum()
    sums2 = df2.groupby("SMA", sort=False)[col].sum()

    # -------------------------------
    # ✅ Updated section: Rename source files in charts
    # -------------------------------
    comparison = pd.DataFrame({"Base Period": sums1, "Comparison Period": sums2}).fillna(0).reset_index()

    comparison_long = comparison.melt(
        id_vars="SMA",
        value_vars=["Base Period", "Comparison Period"],
        var_name="Source File",
        value_name="Loan Outstanding (Rs.)"
    )

    fig = px.bar(
        comparison_long,
        x="Loan Outstanding (Rs.)",
        y="SMA",
        color="Source File",
        barmode="group",
        text="Loan Outstanding (Rs.)",
        orientation="h",
        height=500,
        labels={"Loan Outstanding (Rs.)": "Total Loan Outstanding (Rs.)", "SMA": "SMA"},
        title="Loan Amount by SMA Classification"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Loan Amount (Rs.)"),
        yaxis=dict(title="SMA Classifications", automargin=True),  # updated
        legend_title_text="Source File",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig

# # st.subheader("Asset Classification Comparison")
# col1, col2 = st.columns(2)
# with col1:
#     compare_project_counts_plotly("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx", key="asset_count")
# with col2:
#     compare_loan_outstanding_plotly("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx", key="asset_outstanding")

# # st.subheader("SMA Comparison (excluding '0')")
# col3, col4 = st.columns(2)
# with col3:
#     compare_project_counts_sma("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx", key="sma_count")
# with col4:
#     compare_loan_outstanding_sma("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx", key="sma_outstanding")
