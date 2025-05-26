# 🔧 Pure Financial Document ETL Pipeline

**Simple tool to extract and structure financial document data for analysis**

## 🚀 **Quick Start (Non-Technical Users)**

### **Step 1: Download the Tool**
1. Go to: https://github.com/daleparr/boe-etl
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file to your computer

### **Step 2: Launch the Tool**

#### **Windows Users:**
- Double-click **`launch_etl.bat`**
- The tool will install itself and open in your web browser

#### **Mac/Linux Users:**
- Double-click **`launch_etl.sh`**
- If it doesn't work, open Terminal and run: `bash launch_etl.sh`

### **Step 3: Use the Tool**
1. **Web interface opens** at http://localhost:8501
2. **Enter institution name** (e.g., "JPMorgan Chase")
3. **Select quarter** (e.g., "Q1 2025")
4. **Drag and drop files** (PDF, Excel, Text)
5. **Click "Extract & Structure Data"**
6. **Download your CSV file** when complete

## 📁 **What Files Can I Upload?**

- **PDF Files**: Earnings transcripts, presentations, reports
- **Excel Files**: Financial supplements, data sheets
- **Text Files**: Transcript files, documents

## 📊 **What Do I Get?**

A **CSV file** with structured data containing:
- **Text sentences** from your documents
- **Financial terms** found in each sentence
- **Financial figures** (numbers, percentages, amounts)
- **Document type** (earnings call, presentation, etc.)
- **Speaker information** (when available)
- **Processing metadata**

## 🎯 **Perfect For:**

- **Financial analysts** processing earnings documents
- **Research teams** structuring financial data
- **Compliance teams** organizing regulatory documents
- **Anyone** who needs to convert financial PDFs to structured data

## ❓ **Need Help?**

### **Common Issues:**

**Tool won't start?**
- Make sure Python is installed: https://python.org
- Try running: `pip install streamlit pandas PyPDF2 openpyxl`

**Can't upload files?**
- Check file formats: PDF, Excel (.xlsx, .xls), Text (.txt)
- File size limit: 200MB per file

**No data extracted?**
- Check if PDF has selectable text (not scanned image)
- Try a different file format

### **Contact:**
- Create an issue on GitHub: https://github.com/daleparr/boe-etl/issues

## 🔧 **For Technical Users**

### **Manual Installation**
```bash
# Clone repository
git clone https://github.com/daleparr/boe-etl.git
cd boe-etl

# Install dependencies
pip install streamlit pandas PyPDF2 openpyxl

# Launch application
streamlit run pure_etl_frontend.py
```

### **What This Tool Does**
- **Pure ETL approach**: Extract and structure data without analytical assumptions
- **Multi-format support**: PDF, Excel, text file processing
- **Raw feature extraction**: Financial terms, figures, temporal indicators
- **Document classification**: Automatic detection of document types
- **Missing value handling**: Clean datasets ready for analysis
- **Web interface**: User-friendly drag-and-drop processing

### **Output Schema**
```csv
source_file,institution,quarter,sentence_id,speaker_raw,text,source_type,
all_financial_terms,financial_figures,temporal_indicators,
has_financial_terms,has_financial_figures,has_temporal_language,
word_count,char_count,processing_date
```

### **Supported Document Types**
- `earnings_call` - Transcript files with speaker patterns
- `earnings_presentation` - Slide decks and presentations
- `financial_supplement` - Financial data and statements
- `financial_report` - Quarterly/annual reports
- `press_release` - Press releases and announcements
- `financial_data` - Excel files with financial data

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

---

**🎯 Extract financial data in minutes, not hours!** 🔧
