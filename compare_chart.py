import pandas as pd
import plotly.express as px
import streamlit as st

def compare_project_counts_plotly(file1: str, file2: str, sheet_name=0):
    """
    Compare two Excel files and plot a modern interactive horizontal bar chart
    showing count of distinct PROJECT NO by Asset Classification using Plotly.
    """

    # Read Excel files with specific headers
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    # Normalize classification column names
    if "Asset Classification as on 30.06.2025" in df1.columns:
        df1.rename(columns={"Asset Classification as on 30.06.2025": "Asset Classification"}, inplace=True)
    if "Asset classification" in df2.columns:
        df2.rename(columns={"Asset classification": "Asset Classification"}, inplace=True)

    # Convert PROJECT NO and Asset Classification to string
    df1["PROJECT NO"] = df1["PROJECT NO"].astype(str).str.strip()
    df2["PROJECT NO"] = df2["PROJECT NO"].astype(str).str.strip()
    df1["Asset Classification"] = df1["Asset Classification"].astype(str).str.strip()
    df2["Asset Classification"] = df2["Asset Classification"].astype(str).str.strip()

    # Group by classification
    counts1 = df1.groupby("Asset Classification")["PROJECT NO"].nunique()
    counts2 = df2.groupby("Asset Classification")["PROJECT NO"].nunique()

    # Combine into tidy DataFrame
    comparison = pd.DataFrame({
        "File1": counts1,
        "File2": counts2
    }).fillna(0).astype(int).reset_index()

    # Melt to long format for Plotly
    comparison_long = comparison.melt(
        id_vars="Asset Classification",
        value_vars=["File1", "File2"],
        var_name="File",
        value_name="Count"
    )

    # Create interactive horizontal grouped bar chart
    fig = px.bar(
        comparison_long,
        x="Count",
        y="Asset Classification",
        color="File",
        barmode="group",  # options: 'stack' or 'group'
        text="Count",
        orientation="h",
        height=500,
        labels={"Count": "Count of Distinct PROJECT NO", "Asset Classification": "Asset Classification"},
        title="Comparative Count of Distinct PROJECT NO by Asset Classification"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Count of Distinct PROJECT NO"),
        yaxis=dict(title="Asset Classification", automargin=True),
        legend_title_text="Source File",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    fig.update_traces(textposition="outside")

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    return fig
compare_project_counts_plotly("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx")


import pandas as pd
import plotly.express as px
import streamlit as st

def compare_loan_outstanding_plotly(file1: str, file2: str, sheet_name=0):
    """
    Compare two Excel files and plot a modern interactive horizontal bar chart
    showing total LOAN OUTSTANDING (Rs.) by Asset Classification using Plotly.
    """

    # Read Excel files with specific headers
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    # Normalize classification column names
    if "Asset Classification as on 30.06.2025" in df1.columns:
        df1.rename(columns={"Asset Classification as on 30.06.2025": "Asset Classification"}, inplace=True)
    if "Asset classification" in df2.columns:
        df2.rename(columns={"Asset classification": "Asset Classification"}, inplace=True)

    # Clean data
    df1["Asset Classification"] = df1["Asset Classification"].astype(str).str.strip()
    df2["Asset Classification"] = df2["Asset Classification"].astype(str).str.strip()

    # Ensure LOAN OUTSTANDING column exists and convert to numeric
    for col in ["LOAN OUTSTANDING (Rs.)"]:
        if col not in df1.columns or col not in df2.columns:
            raise KeyError(f"Column '{col}' missing in one of the files")
        df1[col] = pd.to_numeric(df1[col], errors='coerce').fillna(0)
        df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)

    # Group by Asset Classification and sum LOAN OUTSTANDING
    sums1 = df1.groupby("Asset Classification", sort=False)["LOAN OUTSTANDING (Rs.)"].sum()
    sums2 = df2.groupby("Asset Classification", sort=False)["LOAN OUTSTANDING (Rs.)"].sum()

    # Combine into tidy DataFrame
    comparison = pd.DataFrame({
        "File1": sums1,
        "File2": sums2
    }).fillna(0).reset_index()

    # Melt for Plotly
    comparison_long = comparison.melt(
        id_vars="Asset Classification",
        value_vars=["File1", "File2"],
        var_name="File",
        value_name="Loan Outstanding (Rs.)"
    )

    # Plotly horizontal grouped bar chart
    fig = px.bar(
        comparison_long,
        x="Loan Outstanding (Rs.)",
        y="Asset Classification",
        color="File",
        barmode="group",  # use 'stack' if you want stacked bars
        text="Loan Outstanding (Rs.)",
        orientation="h",
        height=500,
        labels={"Loan Outstanding (Rs.)": "Total Loan Outstanding (Rs.)", "Asset Classification": "Asset Classification"},
        title="Comparative Total LOAN OUTSTANDING (Rs.) by Asset Classification"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Total Loan Outstanding (Rs.)"),
        yaxis=dict(title="Asset Classification", automargin=True),
        legend_title_text="Source File",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    return fig
compare_loan_outstanding_plotly("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx")


def compare_project_counts_sma(file1: str, file2: str, sheet_name=0):
    """
    Compare two Excel files and plot a modern interactive horizontal bar chart
    showing distinct count of PROJECT NO by SMA (excluding SMA='0') using Plotly.
    """

    # Read Excel files with specific headers
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    # Normalize SMA column names
    if "SMA Staging as on 30.06.2025)" in df1.columns:
        df1.rename(columns={"SMA Staging as on 30.06.2025)": "SMA"}, inplace=True)
    if "SMA" in df2.columns:
        df2.rename(columns={"SMA": "SMA"}, inplace=True)

    # Convert PROJECT NO and SMA to string and clean
    for df in [df1, df2]:
        df["PROJECT NO"] = df["PROJECT NO"].astype(str).str.strip()
        df["SMA"] = df["SMA"].astype(str).str.strip()

        # Exclude SMA == '0'
        df.drop(df[df["SMA"] == '0'].index, inplace=True)

    # Group by SMA and count distinct PROJECT NO
    counts1 = df1.groupby("SMA", sort=False)["PROJECT NO"].nunique()
    counts2 = df2.groupby("SMA", sort=False)["PROJECT NO"].nunique()

    # Combine into tidy DataFrame
    comparison = pd.DataFrame({
        "File1": counts1,
        "File2": counts2
    }).fillna(0).reset_index()

    # Melt for Plotly
    comparison_long = comparison.melt(
        id_vars="SMA",
        value_vars=["File1", "File2"],
        var_name="File",
        value_name="Distinct Projects"
    )

    # Plotly horizontal grouped bar chart
    fig = px.bar(
        comparison_long,
        x="Distinct Projects",
        y="SMA",
        color="File",
        barmode="group",  # use 'stack' for stacked bars
        text="Distinct Projects",
        orientation="h",
        height=500,
        labels={"Distinct Projects": "Count of Distinct PROJECT NO", "SMA": "SMA"},
        title="Comparative Distinct Count of PROJECT NO by SMA (excluding '0')"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Count of Distinct PROJECT NO"),
        yaxis=dict(title="SMA", automargin=True),
        legend_title_text="Source File",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    return fig


compare_project_counts_sma("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx")


def compare_loan_outstanding_sma(file1: str, file2: str, sheet_name=0):
    """
    Compare two Excel files and plot a modern interactive horizontal bar chart
    showing total LOAN OUTSTANDING (Rs.) by SMA (excluding SMA='0') using Plotly.
    """

    # Read Excel files with specific headers
    df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1)
    df2 = pd.read_excel(file2, sheet_name=sheet_name, header=2)

    # Normalize SMA column names
    if "SMA Staging as on 30.06.2025)" in df1.columns:
        df1.rename(columns={"SMA Staging as on 30.06.2025)": "SMA"}, inplace=True)
    if "SMA" in df2.columns:
        df2.rename(columns={"SMA": "SMA"}, inplace=True)

    # Convert SMA to string, clean, and exclude '0'
    for df in [df1, df2]:
        df["SMA"] = df["SMA"].astype(str).str.strip()
        df.drop(df[df["SMA"] == '0'].index, inplace=True)

    # Ensure LOAN OUTSTANDING column exists and convert to numeric
    for col in ["LOAN OUTSTANDING (Rs.)"]:
        if col not in df1.columns or col not in df2.columns:
            raise KeyError(f"Column '{col}' missing in one of the files")
        df1[col] = pd.to_numeric(df1[col], errors='coerce').fillna(0)
        df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)

    # Group by SMA and sum LOAN OUTSTANDING
    sums1 = df1.groupby("SMA", sort=False)["LOAN OUTSTANDING (Rs.)"].sum()
    sums2 = df2.groupby("SMA", sort=False)["LOAN OUTSTANDING (Rs.)"].sum()

    # Combine into tidy DataFrame
    comparison = pd.DataFrame({
        "File1": sums1,
        "File2": sums2
    }).fillna(0).reset_index()

    # Melt for Plotly
    comparison_long = comparison.melt(
        id_vars="SMA",
        value_vars=["File1", "File2"],
        var_name="File",
        value_name="Loan Outstanding (Rs.)"
    )

    # Plotly horizontal grouped bar chart
    fig = px.bar(
        comparison_long,
        x="Loan Outstanding (Rs.)",
        y="SMA",
        color="File",
        barmode="group",  # use 'stack' for stacked bars
        text="Loan Outstanding (Rs.)",
        orientation="h",
        height=500,
        labels={"Loan Outstanding (Rs.)": "Total Loan Outstanding (Rs.)", "SMA": "SMA"},
        title="Comparative Total LOAN OUTSTANDING (Rs.) by SMA (excluding '0')"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Total Loan Outstanding (Rs.)"),
        yaxis=dict(title="SMA", automargin=True),
        legend_title_text="Source File",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    return fig
compare_loan_outstanding_sma("Loan Book (31.03.2025).xlsx", "Loan Book (30.06.2025).xlsx")


