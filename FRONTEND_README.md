# 📊 Financial ETL Pipeline - Web Frontend

A professional web interface for processing financial documents from multiple institutions through the ETL pipeline.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_frontend.txt
```

### 2. Launch the Frontend
```bash
python start_frontend.py
```

The web interface will open at: **http://localhost:8501**

## 🎯 Features

### 📁 **Document Upload & Processing**
- **Multi-format support**: PDF presentations, Excel supplements, text transcripts
- **Drag & drop interface**: Easy file upload
- **Real-time progress**: Live processing status with progress bars
- **Batch processing**: Handle multiple documents simultaneously

### 🏛️ **Institution Management**
- **Multiple institutions**: Process different banks back-to-back
- **Quarter tracking**: Organize by reporting periods (Q1-Q4)
- **Automatic organization**: Files organized by institution and quarter

### 📚 **Processing History**
- **Complete audit trail**: Track all processing runs
- **Status monitoring**: Success/failure tracking
- **File inventory**: Record of all processed documents
- **Record counts**: Number of extracted records per run

### 📥 **Export & Download**
- **CSV exports**: Download structured datasets
- **Multiple formats**: Core NLP dataset, financial subset
- **Instant download**: One-click file downloads
- **Organized outputs**: Files named by institution and quarter

## 🖥️ Interface Overview

### **Sidebar - Processing Controls**
- **Institution Name**: Enter bank/financial institution
- **Quarter Selection**: Choose reporting period
- **File Upload**: Drag & drop documents
- **Process Button**: Start ETL pipeline

### **Main Area - Status & Results**
- **Processing Status**: Real-time progress updates
- **Download Section**: Access generated CSV files
- **Error Handling**: Clear error messages and guidance

### **History Panel**
- **Recent Runs**: Last 10 processing sessions
- **Expandable Details**: Click to see run details
- **Quick Reference**: Institution, quarter, file count, status

## 📊 Output Files

### **Core Dataset**: `{Institution}_{Quarter}_nlp_core_dataset.csv`
- Complete NLP-ready dataset
- All extracted sentences with metadata
- Speaker identification and roles
- Topic labels and entity recognition

### **Financial Dataset**: `{Institution}_{Quarter}_financial_dataset.csv`
- Financial content only
- Filtered for financial terms and metrics
- Optimized for financial analysis

## 🔧 Technical Details

### **ETL Pipeline Integration**
- **Document Parsing**: PDF, Excel, text processing
- **NLP Enhancement**: Entity recognition, topic modeling
- **Schema Compliance**: Standardized output format
- **Quality Assurance**: Automated validation and error handling

### **File Organization**
```
uploads/
├── {Institution}/
│   └── {Quarter}/
│       ├── presentation.pdf
│       ├── supplement.xlsx
│       └── transcript.txt
outputs/
├── {Institution}_{Quarter}_{Timestamp}/
│   ├── nlp_core_dataset.csv
│   └── financial_dataset.csv
```

### **Processing History**
- Stored in `processing_history.json`
- Persistent across sessions
- Includes timestamps, file lists, status

## 🎯 Usage Examples

### **Example 1: JPMorgan Chase Q1 2025**
1. Enter "JPMorgan Chase" as institution
2. Select "Q1 2025" quarter
3. Upload earnings presentation PDF
4. Upload financial supplement Excel
5. Upload earnings call transcript
6. Click "Process Documents"
7. Download generated CSV files

### **Example 2: Multiple Institutions**
1. Process JPMorgan Chase Q1 2025
2. Process Bank of America Q1 2025
3. Process Wells Fargo Q1 2025
4. Compare outputs across institutions
5. History panel shows all processing runs

## 🛠️ Troubleshooting

### **Common Issues**

**"Please install requirements"**
- Run: `pip install -r requirements_frontend.txt`

**"No records extracted"**
- Check file formats (PDF, Excel, TXT supported)
- Ensure files contain readable text
- Try smaller files first

**"Processing failed"**
- Check file permissions
- Ensure sufficient disk space
- Review error message in interface

### **Performance Tips**
- **File size**: Keep individual files under 50MB
- **Batch size**: Process 3-5 files at once for optimal performance
- **File types**: PDFs with text (not scanned images) work best

## 📋 Requirements

### **System Requirements**
- Python 3.8+
- 4GB RAM minimum
- 1GB free disk space

### **Python Packages**
- streamlit (web interface)
- pandas (data processing)
- scikit-learn (NLP features)
- openpyxl (Excel support)
- PyPDF2 (PDF processing)

## 🔒 Security & Privacy

### **Data Handling**
- **Local processing**: All data stays on your machine
- **No cloud uploads**: Documents processed locally
- **Temporary storage**: Uploaded files stored temporarily
- **Clean up**: Option to clear processing history

### **File Management**
- Uploaded files stored in `uploads/` directory
- Output files stored in `outputs/` directory
- Processing history in `processing_history.json`
- Manual cleanup available through interface

## 🎉 Success Workflow

1. **📁 Upload** → Drag & drop financial documents
2. **🏛️ Configure** → Set institution and quarter
3. **🚀 Process** → Click to start ETL pipeline
4. **📊 Monitor** → Watch real-time progress
5. **📥 Download** → Get structured CSV datasets
6. **📚 Track** → View in processing history
7. **🔄 Repeat** → Process next institution

## 🆘 Support

For technical issues or questions:
1. Check the troubleshooting section above
2. Review processing history for error details
3. Ensure all requirements are installed
4. Try with smaller test files first

---

**Ready to transform your financial documents into structured NLP datasets!** 🚀