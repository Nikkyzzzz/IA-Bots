# ============================== PDF EXTRACTION MODULE ==============================
import os
import json
import time
import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
try:
    import cohere
except ImportError:
    cohere = None

# ============================== PDF EXTRACTION CONFIGURATION ==============================
# Load API key from .env file
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere client if API key is available
cohere_client = None
if COHERE_API_KEY and cohere:
    try:
        cohere_client = cohere.Client(COHERE_API_KEY)
    except Exception as e:
        st.warning(f"Failed to initialize Cohere client: {e}")

# Prompt template for PDF extraction
EXTRACTION_PROMPT = """
You are an expert financial document analyzer specializing in loan and project documentation. 
The input contains text from multiple pages of a PDF with detailed project and financial information.

SEARCH STRATEGY FOR EACH FIELD:
0. Company Name: Look for "Company", "M/s", "Name of the Borrower", "Borrower", "Applicant", company names in headers/titles, document titles
1. Project Number: Look for "Project No", "Application No", "Loan No", "Reference No", or numeric identifiers
2. Loan amount: Search for "Loan Amount", "Sanctioned Amount", "Credit Facility", "Rs.", "Crore", "Cr", amounts in numerical format
3. Project Type & Sector: Find "Project Type", "Sector", "Industry", "Business", "Manufacturing", "Solar", "Wind", "Biomass", etc.
4. Grade: Look for "Grade", "Rating", "Credit Rating", "Internal Grade", "Risk Grade", "A", "AA", "B", "BB", etc.
5. Interest: Search for "Interest Rate", "ROI", "Rate of Interest", "Coupon", "%", "per annum", "p.a."
6. Project Cost: Find "Project Cost", "Total Cost", "Investment", "Capital Cost", "Total Project Investment"
7. Promotor Contribution: Look for "Promoter Contribution", "Equity", "Own Contribution", "Margin Money"
8. Minimum promoter contribution: Calculate from promoter contribution/project cost or find explicit percentages
9. Debt Equity Ratio: Search for "Debt Equity", "D:E", "Debt-Equity", "Leverage", ratios like "70:30", "75:25"
10. Average DSCR: Find "DSCR", "Debt Service Coverage", "Coverage Ratio", numerical values like "1.2", "1.5"
11. Average DSCR requirement: Look for minimum DSCR requirements, thresholds, or compliance criteria
12. Average Asset Coverage: Search for "Asset Coverage", "Security Coverage", "Collateral Coverage"
13. Contingent Liability(CL/NW): Find "CL/NW", "Networth","Ratio,"guarantee"
14. Moratorium/Grace period: VERY IMPORTANT - Look for ANY of these terms in tables, rows, or text:
    - "Moratorium"
    - "Moratorium/Grace period"
    - "Grace period" 
    - "Grace Period (Months)"
    - "Repayment Holiday"
    - "COD" (Commercial Operation Date)
    - "SCOD" (Scheduled Commercial Operation Date)
    - "months from COD"
    - "months from SCOD"
    - "6 months"
    - "12 months"
    - "18 months"
    - Search in numbered rows, tables, and structured data
    - Look for patterns like "X months from Y"
    - Pay special attention to table rows numbered 20, 21, 22, 23

ENHANCED EXTRACTION RULES:
- Scan ALL pages thoroughly, especially looking for TABLES and STRUCTURED DATA
- Pay special attention to numbered rows (like row 20, 21, 22, etc.)
- Look for tables with S.No, descriptions, and values
- Extract numerical values with units (Rs., Cr., %, months, years)
- For ratios, convert to standard format (e.g., "70:30" or "1.5:1")
- Search for synonyms and alternative terms for each field
- Look in headers, footers, margin notes, and especially TABLE CELLS
- Check both text blocks and tabular data structures
- Extract values from financial statements, loan terms, project descriptions, and TABULAR SUMMARIES

SPECIAL FOCUS ON MORATORIUM:
- If you see "Moratorium/Grace period (Months)" followed by a value like "6 months from COD", extract it
- Look for table rows numbered 20, 21, 22, 23 that often contain loan terms
- Search for any mention of moratorium in structured financial tables
- Don't miss data in table cells or structured formats

Return JSON array with ONE object per company/project:
{
  "Company Name": "actual company name from document",
  "Project Number": "exact project/loan number found",
  "Loan amount": "amount with currency (Rs. X Cr./Lakhs)",
  "Project Type & Sector": "detailed sector description",
  "Grade": "exact grade/rating found",
  "Interest": "rate with % symbol",
  "Project Cost": "total cost with currency",
  "Promotor Contribution": "contribution amount with currency",
  "Minimum promoter contribution": "percentage with % symbol",
  "Debt Equity Ratio": "ratio in X:Y or X.Y:1 format",
  "Average DSCR (new clients)": "numerical DSCR value",
  "Average DSCR requirement": "minimum DSCR threshold",
  "Average Asset Coverage ratio": "coverage ratio value",
  "Contingent Liability": "liability amount or description",
  "Moratorium/grace period": "duration in months/years with description"
}

CRITICAL: Extract REAL VALUES from the document. Do NOT use "Not specified" unless the information is genuinely absent after thorough search of ALL text and tables. Pay special attention to TABULAR DATA and numbered rows.

Return only valid JSON array.
"""

# ---------------- PDF Processing Function ---------------- #
def run_pdf_extraction(uploaded_pdfs, ui_refs=None):
    """Process PDF extraction with optional progress tracking"""
    if not cohere_client:
        error_msg = ""
        if not COHERE_API_KEY:
            error_msg = "❌ Cohere API key not found. Please set COHERE_API_KEY in your .env file."
        elif not cohere:
            error_msg = "❌ Cohere library not installed. Please install it with: pip install cohere"
        else:
            error_msg = "❌ Failed to initialize Cohere client."
        
        if ui_refs and "data_extraction" in ui_refs:
            ui_refs["data_extraction"]["status"].markdown(f"- ❌ **PDF Data Extraction** — API Error")
        elif ui_refs is not None:  # Only use st functions when not in background mode
            st.error(error_msg)
        return {}
    
    if not uploaded_pdfs:
        warning_msg = "No PDF files found for extraction."
        if ui_refs and "data_extraction" in ui_refs:
            ui_refs["data_extraction"]["status"].markdown("- ❌ **PDF Data Extraction** — No Files")
        elif ui_refs is not None:  # Only use st functions when not in background mode
            st.warning(warning_msg)
        return {}
    
    # Initialize consolidated data structure
    all_companies_data = []
    total_files = len(uploaded_pdfs)
    
    # Update UI for data extraction (if UI refs provided)
    status_ph = None
    prog_ph = None
    if ui_refs and "data_extraction" in ui_refs:
        status_ph = ui_refs["data_extraction"]["status"]
        prog_ph = ui_refs["data_extraction"]["prog"]
    
    start_time = time.time()
    
    for i, uploaded_file in enumerate(uploaded_pdfs, start=1):
        if status_ph:
            status_ph.markdown(f"- ⏳ **PDF Data Extraction** — Processing {uploaded_file.name} ({i}/{total_files})")
        
        # Extract text from ALL pages for comprehensive data extraction
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                extracted_texts = []
                total_pages = len(pdf.pages)
                
                # Extract from all pages, but prioritize key pages
                priority_pages = [0, 19, 22, 30]  # Page 1, 20, 23, 31 (0-based)
                
                # First extract priority pages with both text and table extraction
                for p in priority_pages:
                    if p < total_pages:
                        page = pdf.pages[p]
                        
                        # Extract regular text
                        text = page.extract_text()
                        if text:
                            extracted_texts.append(f"Priority Page {p+1} Text:\n{text}")
                        
                        # Extract tables specifically (better for structured data)
                        try:
                            tables = page.extract_tables()
                            for i, table in enumerate(tables):
                                if table:
                                    table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                                    extracted_texts.append(f"Priority Page {p+1} Table {i+1}:\n{table_text}")
                        except:
                            pass
                
                # Then extract other pages (up to 50 pages max to avoid token limits)
                other_pages = [i for i in range(min(50, total_pages)) if i not in priority_pages]
                for p in other_pages:
                    page = pdf.pages[p]
                    text = page.extract_text()
                    if text and len(text.strip()) > 100:  # Only include substantial content
                        extracted_texts.append(f"Page {p+1}:\n{text[:2000]}")  # Limit text per page
                        
                    # Also try to extract tables from other pages
                    try:
                        tables = page.extract_tables()
                        for i, table in enumerate(tables):
                            if table and len(table) > 0:
                                table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                                if "moratorium" in table_text.lower() or "grace" in table_text.lower():
                                    extracted_texts.append(f"Page {p+1} Important Table {i+1}:\n{table_text}")
                    except:
                        pass
            
            joined_text = "\n\n".join(extracted_texts)
            
            # Send to Cohere LLM with retry mechanism
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = cohere_client.chat(
                        model="command-r-plus-08-2024",
                        message=f"{EXTRACTION_PROMPT}\n\nDocument Content:\n{joined_text}",
                        max_tokens=4000,
                        temperature=0.1  # Low temperature for consistent extraction
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1)  # Wait before retry
            
            try:
                # Try to parse JSON response
                response_text = response.text.strip()
                
                # Clean response if it has markdown code blocks
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()
                
                rows = json.loads(response_text)
                
                # Ensure rows is a list
                if isinstance(rows, dict):
                    rows = [rows]
                
                for row in rows:
                    # Calculate minimum promoter contribution if not provided
                    min_contrib = row.get("Minimum promoter contribution", "")
                    if not min_contrib or min_contrib == "Not specified":
                        try:
                            contrib = row.get("Promotor Contribution", "")
                            cost = row.get("Project Cost", "")
                            if contrib and cost:
                                # Extract numeric values
                                import re
                                contrib_num = re.findall(r'[\d.]+', contrib.replace(",", ""))
                                cost_num = re.findall(r'[\d.]+', cost.replace(",", ""))
                                if contrib_num and cost_num:
                                    percentage = (float(contrib_num[0]) / float(cost_num[0])) * 100
                                    min_contrib = f"{percentage:.2f}%"
                        except:
                            pass
                    
                    # Store company data
                    company_info = {
                        "Company Name": row.get("Company Name", f"Company from {uploaded_file.name}"),
                        "Project Number": row.get("Project Number", ""),
                        "Loan amount": row.get("Loan amount", ""),
                        "Project Type & Sector": row.get("Project Type & Sector", ""),
                        "Grade": row.get("Grade", ""),
                        "Interest": row.get("Interest", ""),
                        "Project Cost": row.get("Project Cost", ""),
                        "Promotor Contribution": row.get("Promotor Contribution", ""),
                        "Minimum promoter contribution": min_contrib,
                        "Debt Equity Ratio": row.get("Debt Equity Ratio", ""),
                        "Average DSCR (new clients)": row.get("Average DSCR (new clients)", ""),
                        "Average DSCR requirement": row.get("Average DSCR requirement", ""),
                        "Average Asset Coverage ratio": row.get("Average Asset Coverage ratio", ""),
                        "Contingent Liability": row.get("Contingent Liability", ""),
                        "Moratorium/grace period": row.get("Moratorium/grace period", "")
                    }
                    all_companies_data.append(company_info)
                    
            except Exception as e:
                # Only use st.error when not in background mode
                if ui_refs is not None:
                    st.error(f"⚠️ Could not parse response for {uploaded_file.name}: {e}")
                    st.text(f"Raw response: {response.text[:500]}")
                # Add empty company data if parsing fails
                all_companies_data.append({
                    "Company Name": f"Unknown Company from {uploaded_file.name}",
                    "Project Number": "", "Loan amount": "", "Project Type & Sector": "", "Grade": "",
                    "Interest": "", "Project Cost": "", "Promotor Contribution": "", 
                    "Minimum promoter contribution": "", "Debt Equity Ratio": "",
                    "Average DSCR (new clients)": "", "Average DSCR requirement": "",
                    "Average Asset Coverage ratio": "", "Contingent Liability": "", "Moratorium/grace period": ""
                })
            
        except Exception as e:
            # Only use st.error when not in background mode
            if ui_refs is not None:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            continue
        
        # Update progress
        progress = int(i / total_files * 100)
        if prog_ph:
            prog_ph.progress(progress)
    
    # Create consolidated horizontal format DataFrame
    if all_companies_data:
        particulars = [
            "Project Number", "Loan amount", "Project Type & Sector", "Grade",
            "Interest", "Project Cost", "Promotor Contribution", "Minimum promoter contribution",
            "Debt Equity Ratio", "Average DSCR (new clients)", "Average DSCR requirement", 
            "Average Asset Coverage ratio", "Contingent Liability", "Moratorium/grace period"
        ]
        
        # Create base DataFrame with S.No and Particulars
        consolidated_data = {
            "S.No": list(range(1, len(particulars) + 1)),
            "Particulars": particulars
        }
        
        # Add each company as a column
        for company_info in all_companies_data:
            company_name = company_info["Company Name"]
            company_values = [company_info[particular] for particular in particulars]
            consolidated_data[company_name] = company_values
        
        df = pd.DataFrame(consolidated_data)
        extracted_data = {"consolidated_data": df}
    else:
        # Create empty format if no data
        particulars = [
            "Project Number", "Loan amount", "Project Type & Sector", "Grade",
            "Interest", "Project Cost", "Promotor Contribution", "Minimum promoter contribution",
            "Debt Equity Ratio", "Average DSCR (new clients)", "Average DSCR requirement", 
            "Average Asset Coverage ratio", "Contingent Liability", "Moratorium/grace period"
        ]
        consolidated_data = {
            "S.No": list(range(1, len(particulars) + 1)),
            "Particulars": particulars,
            "No Data": [""] * len(particulars)
        }
        df = pd.DataFrame(consolidated_data)
        extracted_data = {"consolidated_data": df}
    
    # Final update
    if status_ph:
        status_ph.markdown("- ✅ **PDF Data Extraction** — Completed")
    if prog_ph:
        prog_ph.progress(100)
    
    return extracted_data


def check_pdf_extraction_requirements():
    """Check if all requirements for PDF extraction are met"""
    missing_requirements = []
    
    if not cohere:
        missing_requirements.append("cohere library")
    
    try:
        import pdfplumber
    except ImportError:
        missing_requirements.append("pdfplumber library")
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_requirements.append("python-dotenv library")
    
    if not COHERE_API_KEY:
        missing_requirements.append("COHERE_API_KEY environment variable")
    
    return missing_requirements
