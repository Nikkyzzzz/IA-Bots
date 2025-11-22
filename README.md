# Audit Bots 🤖

An intelligent data analysis and auditing platform that automates financial and operational audits across multiple industry verticals using AI-powered document processing and rule-based analytics.

## 📋 Overview

Audit Bots is a comprehensive Streamlit-based application designed to identify anomalies, compliance issues, and potential fraud indicators in business processes. The system supports multiple industries with specialized audit modules for Manufacturing, Banking, and more.

## ✨ Key Features

### Multi-Industry Support
- **Manufacturing**: P2P (Procure-to-Pay), O2C (Order-to-Cash), and H2R (Hire-to-Retire) processes
- **Banking**: Loan portfolio analysis, NPA detection, provision verification, PDF data extraction
- **Extensible**: Ready for Logistics, Healthcare, Retail, Energy, and Construction

### Intelligent Data Processing
- **PDF Data Extraction**: AI-powered extraction from loan documents using Cohere LLM
- **Excel File Processing**: Automated mapping and validation of multi-sheet workbooks
- **Flexible Column Mapping**: Dynamic field mapping to handle varying data formats
- **Smart Sheet Detection**: Automatic identification of required data sheets

### Advanced Analytics
- **38+ Audit Bots**: Automated detection of various compliance and fraud indicators
- **Real-time Processing**: Interactive progress tracking with detailed status updates
- **Consolidated Reporting**: Multi-format export options (Excel, CSV)
- **Visual Dashboards**: Interactive charts and comparative analysis

## 🏗️ Architecture

### Core Components

```
25bots/
├── run_app.py              # Application entry point
├── zeropage.py             # Industry selection router
├── firstpage.py            # Manufacturing module router
├── secondpage.py           # File upload & sheet mapping
├── thirdpage.py            # Column mapping interface
├── fourthpage.py           # Bot selection page
├── processpage.py          # Processing engine
├── fifthpage.py            # Results dashboard
├── logic.py                # Main data preparation & orchestration
├── logic6.py               # Manufacturing audit rules (25+ bots)
├── b1.py - b7.py           # Banking module pages
├── blogic.py               # Banking logic (legacy)
├── blogic6.py              # Banking audit rules (12 bots)
├── pdf_extraction.py       # AI-powered PDF processing
├── charts.py               # Visualization utilities
├── compare_chart.py        # Comparative analytics
├── requirements.txt        # Python dependencies
└── Input/                  # Sample input data folders
    ├── Input Banking/
    └── Input Manufacturing/
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Cohere API key (for PDF extraction)

### Setup

1. **Clone or download the repository**
   ```bash
   cd c:\Users\CAP-035\Desktop\25bots
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   Create a `.env` file in the root directory:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run run_app.py
   ```
   Or directly:
   ```bash
   streamlit run zeropage.py
   ```

## 📊 Audit Bots by Category

### P2P (Procure-to-Pay) - 16 Bots
- **P2P1**: Missing Vendor Master Fields
- **P2P2**: PO-GRN-Invoice Mismatches
- **P2P3**: Invalid Data Entries
- **P2P4**: Quantity Discrepancies
- **P2P5**: Duplicate Vendor Detection
- **P2P6**: GSTIN Validation
- **P2P8**: PO Approval Bypass
- **P2P10**: Excessive Emergency Purchases
- **P2P11**: Vendor Concentration Risk
- **P2P13**: Invoice vs GRN Date Gap
- **P2P14**: Over Receipt Detection
- **P2P15**: Payment Term Adherence
- **P2P16**: Duplicate Invoice Detection
- **P2P17**: Invoices to Inactive Vendors
- **P2P18**: Non-PO Invoices
- **P2P20**: Round-Sum Invoices

### O2C (Order-to-Cash) - 7 Bots
- **O2C1**: Overdue Deliveries
- **O2C2**: Dispatch Without Invoice
- **O2C3**: Missing Customer Master Data
- **O2C24**: Sales Without Sales Order
- **O2C29**: Multiple Invoices per SO
- **O2C32**: Zero/Missing Invoice Amount
- **O2C39**: Excessive Small-Value Sales

### H2R (Hire-to-Retire) - 3 Bots
- **H2R1**: Ghost Employees
- **H2R2**: Attendance After Exit
- **H2R44**: Duplicate Employees

### Banking - 12 Bots
- **Bank1**: Zero/Null ROI Loans
- **Bank2**: Standard Accounts with URI Zero
- **Bank3**: Provision Verification (Substandard NPA)
- **Bank4**: Restructured Standard Accounts
- **Bank5**: Provision Verification (Doubtful-3 NPA)
- **Bank6**: NPA FB Accounts Overdue
- **Bank7**: Negative Outstanding Amounts
- **Bank8**: Standard Accounts Overdue Details
- **Bank9**: Standard Accounts with Odd Interest
- **Bank10**: AGRI-0 Sector Over Limit
- **Bank11**: Misaligned Scheme for Facilities
- **Bank12**: Loan Book Comparison & Asset Classification

## 💡 Usage

### Manufacturing Workflow

1. **Select Industry**
   - Launch application and choose "Manufacturing" from the industry selector

2. **Upload Data Files**
   - Upload Master_Data file (all processes combined), OR
   - Upload individual files: P2P_Master, O2C_Master, H2R_Master
   - Supported formats: XLSX, XLS

3. **Map Data Sheets**
   - Select which sheet corresponds to each required data type
   - System auto-detects sheets with matching names

4. **Map Columns**
   - Match your data columns to required fields
   - Required fields are clearly marked
   - Optional AI assistance for complex mappings

5. **Select Audit Bots**
   - Choose specific bots or select all by category
   - View bot descriptions and requirements

6. **Process & Review**
   - Monitor real-time processing progress
   - Review flagged exceptions by bot
   - Export results in multiple formats

### Banking Workflow

1. **Select Industry**
   - Choose "Banking" from the industry selector

2. **Upload Data**
   - **Traditional Analysis**: Upload loan book Excel files
   - **PDF Extraction**: Upload PDF documents for AI-powered extraction

3. **PDF Processing** (if applicable):
   - Processing starts automatically in the background
   - Progress monitored in "Data Extraction" tab
   - No user intervention required

4. **Process & Analyze**
   - Automated NPA detection
   - Provision verification
   - Loan portfolio analytics

5. **View Results**
   - Traditional banking analysis in Analysis/Output/Report tabs
   - PDF extraction results in "Data Extraction" tab

6. **Export Reports**
   - Download consolidated results
   - Compare across multiple periods

## 📁 Data Requirements

### P2P Module Required Sheets:
- **P2P Sample**: Transaction-level purchase data
  - Fields: Vendor Name, PO No, PO Date, PO Quantity, PO Amount, PO Created By, PO Approved By, GRN No, GRN Date, GRN Quantity, Invoice Date, Invoice Quantity, Invoice Amount, Creator ID
- **Vendor Master**: Vendor registration details
  - Fields: Vendor Name, GST No, PAN No, Bank Account, Creator ID
- **Employee Master**: Employee information
  - Fields: Employee ID, Employee Name, Department, Creator ID, Designation
- **Authority Matrix**: Approval hierarchies and limits
  - Fields: Department, Role, Creation Authority, Creation Amount Limit, Approval Authority, Approval Amount Limit

### O2C Module Required Sheets:
- **O2C Sample**: Sales and delivery transactions
  - Fields: SO Date, Delivery Date, Invoice No, SO No, Invoice Amount, Taxable Amount
- **Customer Master**: Customer registration details
  - Fields: GST No, PAN No, Credit Limit

### H2R Module Required Sheets:
- **Employee Master**: Employee records with exit dates
  - Fields: Employee ID, Employee Name, Exit Date, Status, Bank Account, PAN No
- **Attendance Register**: Monthly attendance data (with D1-D31 columns)
  - Fields: Employee ID, Employee Name, Month, D1-D31

### Banking Module Required Sheets:
- **Loan Book**: Detailed loan portfolio data
- **Additional data sheets** as per specific bot requirements

## 🤖 AI Features

### PDF Data Extraction (Banking)
- Powered by Cohere's Command-R-Plus model
- Extracts 14+ key financial parameters from loan documents
- Supports multi-page documents with complex tables
- Automatic field detection and validation
- Background processing with progress tracking

**Extracted Fields:**
1. Project Number
2. Loan Amount
3. Project Type & Sector
4. Grade/Rating
5. Interest Rate
6. Project Cost
7. Promoter Contribution
8. Minimum Promoter Contribution %
9. Debt Equity Ratio
10. Average DSCR (new clients)
11. Average DSCR Requirement
12. Average Asset Coverage Ratio
13. Contingent Liability
14. Moratorium/Grace Period

### AI-Assisted Column Mapping (Optional)
- Optional AI suggestions for column mappings in Manufacturing module
- Contextual recommendations based on column names
- Reduces manual mapping effort

## 📦 Dependencies

**Core:**
- streamlit >= 1.48.0
- pandas >= 2.3.2
- numpy >= 2.3.3
- openpyxl >= 3.1.5
- xlsxwriter >= 3.2.5

**PDF Processing:**
- pdfplumber >= 0.11.8
- pdfminer.six >= 20251107
- pypdfium2 >= 5.0.0

**AI/ML:**
- cohere >= 5.20.0
- openai >= 2.8.0

**Visualization:**
- plotly >= 6.4.0
- altair >= 5.5.0
- matplotlib-inline >= 0.1.7

**Utilities:**
- python-dotenv >= 1.2.1
- streamlit-antd-components >= 0.3.2
- reportlab >= 4.4.3

*See `requirements.txt` for complete list*

## 📈 Output Formats

Results can be exported in multiple formats:
- **Excel (.xlsx)**: Consolidated workbook with separate sheets per bot
- **CSV**: Individual CSV files per bot
- **Interactive Dashboard**: Real-time filtering and visualization
- **Comparative Analysis**: Multi-period trend analysis

## 🔒 Security & Privacy

- **Local Processing**: All data processed locally (except PDF AI extraction via Cohere API)
- **API Keys**: Stored securely in `.env` file (not committed to version control)
- **No Data Persistence**: Session-based processing only
- **Secure File Handling**: Files processed in memory, not saved to disk

## 🛠️ Development

### Project Structure Highlights

**Routing System:**
- `zeropage.py`: Main industry selector and router
- `firstpage.py`: Manufacturing module entry and routing
- `b1.py-b7.py`: Banking module pages

**Logic Layers:**
- `logic.py`: Data preparation, field mapping, bot orchestration
- `logic6.py`: Manufacturing audit rule implementations (25+ bots)
- `blogic6.py`: Banking audit rule implementations (12 bots)

**UI Components:**
- Consistent header/branding across all pages
- Responsive design with custom CSS
- Progress tracking with visual feedback
- Tabbed results interface

### Adding New Bots

1. Implement bot logic in `logic6.py` (Manufacturing) or `blogic6.py` (Banking)
2. Add bot metadata and description
3. Register bot in `run_all_bots_with_mappings()` function in `logic.py`
4. Add bot code to process status dictionary
5. Update documentation

### Adding New Industries

1. Create new page modules (e.g., `l1.py` for Logistics)
2. Implement industry-specific logic module
3. Add industry route in `zeropage.py`
4. Define required data sheets and field mappings
5. Implement industry-specific bots

## 🐛 Troubleshooting

**Common Issues:**

1. **Import Errors**
   - Solution: Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

2. **PDF Extraction Fails**
   - Verify Cohere API key in `.env` file
   - Check API quota and connectivity
   - Ensure PDF files are not password-protected
   - Check Cohere library installed: `pip install cohere`

3. **File Upload Issues**
   - Verify file format (XLSX, XLS only for Excel; PDF for documents)
   - Check file size limits
   - Ensure sheets are properly named in Excel files

4. **Bot Processing Errors**
   - Check data completeness in required fields
   - Verify column mappings are correct
   - Review error messages in processing logs
   - Ensure date columns are in proper format

5. **Streamlit Not Starting**
   - Verify Streamlit installed: `pip install streamlit`
   - Check port 8501 is not in use
   - Try: `streamlit run zeropage.py --server.port 8502`

## 📝 License

[Specify your license here]

## 👥 Contributors

[Your team information here]

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [Your contact information]

## 🗺️ Roadmap

**Upcoming Features:**
- Additional industry modules (Healthcare, Retail, Energy, Construction, Logistics)
- Enhanced AI capabilities for anomaly detection
- Real-time data streaming support
- Multi-user collaboration features
- Advanced visualization dashboards
- API integration for enterprise systems
- Scheduled automated audits
- Custom bot builder interface
- Multi-language support
- Mobile-responsive interface

---

**Version:** 1.0  
**Last Updated:** November 2025  
**Built with:** Streamlit, Pandas, Cohere AI, pdfplumber