# 📊 **Document Status Classification Guide**

This document explains the **Status Column Classifications** used in the BoE Supervisor Dashboard's Document Analysis Summary.

---

## 🎯 **Status Classifications Overview**

The dashboard automatically assigns one of three status classifications to each uploaded document based on its **Quality Score**:

| Status | Icon | Quality Score Range | Meaning |
|--------|------|-------------------|---------|
| **✅ Ready** | Green checkmark | **> 70%** | Document is ready for analysis |
| **⚠️ Review** | Orange warning | **40% - 70%** | Document needs review before analysis |
| **❌ Poor** | Red X | **< 40%** | Document quality is insufficient for reliable analysis |

---

## 🔍 **Quality Score Calculation**

The Quality Score is calculated using a **weighted formula** that considers three key factors:

### **Formula Components**
```
Quality Score = (File Count Factor × 0.4) + (Type Diversity Factor × 0.3) + (Size Adequacy Factor × 0.3)
```

### **1. File Count Factor (40% weight)**
- **Calculation**: `min(1.0, file_count / 8)`
- **Optimal**: 8 or more files = 100% score
- **Purpose**: Ensures sufficient document coverage for comprehensive analysis

### **2. Type Diversity Factor (30% weight)**
- **Calculation**: `min(1.0, type_diversity / 3)`
- **Optimal**: 3 or more different document types = 100% score
- **Document Types Recognized**:
  - Earnings Transcript
  - Financial Statement
  - Financial Data
  - Other Document
- **Purpose**: Ensures diverse information sources for balanced analysis

### **3. Size Adequacy Factor (30% weight)**
- **Calculation**: `min(1.0, total_size / 10MB)`
- **Optimal**: 10MB or more total content = 100% score
- **Purpose**: Ensures sufficient content volume for meaningful analysis

---

## 📋 **Status Classification Logic**

### **✅ Ready (Quality Score > 70%)**
**Interpretation**: Document set is **ready for comprehensive analysis**

**Characteristics**:
- Sufficient number of documents (typically 6+ files)
- Good document type diversity (2-3 different types)
- Adequate content volume (7+ MB total)
- High confidence in analysis results

**Supervisor Action**: ✅ **Proceed with analysis** - Results will be reliable

---

### **⚠️ Review (Quality Score 40% - 70%)**
**Interpretation**: Document set **needs review** before proceeding

**Characteristics**:
- Moderate document coverage (3-5 files)
- Limited document type diversity (1-2 types)
- Moderate content volume (4-7 MB total)
- Analysis possible but with limitations

**Supervisor Action**: 🔍 **Review and consider**:
- Adding more documents if available
- Proceeding with analysis but noting limitations
- Supplementing with additional data sources

---

### **❌ Poor (Quality Score < 40%)**
**Interpretation**: Document quality is **insufficient for reliable analysis**

**Characteristics**:
- Very few documents (1-2 files)
- Single document type only
- Low content volume (< 4 MB total)
- High risk of unreliable analysis results

**Supervisor Action**: ⚠️ **Do not proceed** without:
- Adding more documents
- Obtaining different document types
- Ensuring adequate content coverage

---

## 🎯 **Optimization Recommendations**

### **To Achieve ✅ Ready Status**

1. **Upload Multiple Documents** (aim for 8+)
   - Earnings transcripts
   - Financial statements
   - Press releases
   - Regulatory filings

2. **Ensure Document Type Diversity** (aim for 3+ types)
   - Mix of transcripts, financial data, and other documents
   - Different quarters/periods if available

3. **Provide Adequate Content Volume** (aim for 10+ MB)
   - Full documents rather than excerpts
   - Complete quarterly reports
   - Comprehensive transcripts

### **Quality Improvement Tips**

- **For Low File Count**: Add more documents from the same institution/period
- **For Low Type Diversity**: Include different types of financial documents
- **For Low Size**: Ensure documents are complete and not truncated

---

## 📊 **Example Quality Score Calculations**

### **Example 1: ✅ Ready Status (85% Quality Score)**
- **Files**: 10 documents
- **Types**: Earnings Transcript, Financial Statement, Financial Data (3 types)
- **Size**: 15 MB total
- **Calculation**: 
  - File Count: min(1.0, 10/8) × 0.4 = 1.0 × 0.4 = 0.40
  - Type Diversity: min(1.0, 3/3) × 0.3 = 1.0 × 0.3 = 0.30
  - Size Adequacy: min(1.0, 15/10) × 0.3 = 1.0 × 0.3 = 0.30
  - **Total**: 0.40 + 0.30 + 0.30 = **1.00 (100%)**

### **Example 2: ⚠️ Review Status (55% Quality Score)**
- **Files**: 4 documents
- **Types**: Earnings Transcript, Other Document (2 types)
- **Size**: 6 MB total
- **Calculation**:
  - File Count: min(1.0, 4/8) × 0.4 = 0.5 × 0.4 = 0.20
  - Type Diversity: min(1.0, 2/3) × 0.3 = 0.67 × 0.3 = 0.20
  - Size Adequacy: min(1.0, 6/10) × 0.3 = 0.6 × 0.3 = 0.18
  - **Total**: 0.20 + 0.20 + 0.18 = **0.58 (58%)**

### **Example 3: ❌ Poor Status (25% Quality Score)**
- **Files**: 1 document
- **Types**: Other Document (1 type)
- **Size**: 2 MB total
- **Calculation**:
  - File Count: min(1.0, 1/8) × 0.4 = 0.125 × 0.4 = 0.05
  - Type Diversity: min(1.0, 1/3) × 0.3 = 0.33 × 0.3 = 0.10
  - Size Adequacy: min(1.0, 2/10) × 0.3 = 0.2 × 0.3 = 0.06
  - **Total**: 0.05 + 0.10 + 0.06 = **0.21 (21%)**

---

## 🔧 **Technical Implementation**

### **Code Reference**
The status classification is implemented in [`boe_supervisor_dashboard.py`](boe_supervisor_dashboard.py) at line 839:

```python
'Status': '✅ Ready' if quality > 0.7 else '⚠️ Review' if quality > 0.4 else '❌ Poor'
```

### **Quality Assessment Function**
The quality score calculation is performed by the `_assess_document_quality()` method (lines 844-858).

---

## 📈 **Impact on Analysis Results**

### **✅ Ready Status Documents**
- **High confidence** in risk scores and sentiment analysis
- **Comprehensive** topic coverage and trend detection
- **Reliable** regulatory flags and peer comparisons
- **Robust** statistical significance in findings

### **⚠️ Review Status Documents**
- **Moderate confidence** in analysis results
- **Limited** topic coverage - may miss important themes
- **Reduced** statistical power for trend detection
- **Potential gaps** in regulatory risk assessment

### **❌ Poor Status Documents**
- **Low confidence** in analysis reliability
- **High risk** of missing critical information
- **Insufficient data** for meaningful trend analysis
- **Not recommended** for supervisory decision-making

---

## 🎯 **Best Practices for Supervisors**

1. **Always aim for ✅ Ready status** before proceeding with analysis
2. **Review document mix** to ensure comprehensive coverage
3. **Consider temporal coverage** - multiple quarters when available
4. **Validate document completeness** - avoid truncated files
5. **Document any limitations** when proceeding with ⚠️ Review status
6. **Never rely solely on ❌ Poor status** analysis for critical decisions

This classification system ensures **supervisory-grade reliability** and **transparency** in document quality assessment for regulatory decision-making.