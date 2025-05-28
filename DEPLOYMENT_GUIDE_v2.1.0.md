# Bank of England Mosaic Lens v2.1.0 Deployment Guide

## 🚀 Production Deployment Instructions

**Version**: 2.1.0  
**Release Date**: May 28, 2025  
**Repository**: https://github.com/daleparr/Bank-of-England-Mosaic-Lens

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for package installation

### Required Dependencies
```bash
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.0.0
scipy>=1.9.0
scikit-learn>=1.1.0
```

## 🔧 Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/daleparr/Bank-of-England-Mosaic-Lens.git
cd Bank-of-England-Mosaic-Lens
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv boe_mosaic_env

# Activate virtual environment
# Windows:
boe_mosaic_env\Scripts\activate
# Linux/Mac:
source boe_mosaic_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"
```

## 🏛️ Production Deployment

### Primary Dashboard (Port 8505)
```bash
cd data_science
streamlit run boe_supervisor_dashboard.py --server.port 8505
```

### Sandbox Environment (Port 8512)
```bash
cd data_science
streamlit run sandbox_integrated_dashboard.py --server.port 8512
```

### Custom Port Configuration
```bash
# Use custom port
streamlit run boe_supervisor_dashboard.py --server.port YOUR_PORT

# Enable external access
streamlit run boe_supervisor_dashboard.py --server.port 8505 --server.address 0.0.0.0
```

## 🔒 Security Configuration

### Network Security
```bash
# Internal network only (recommended for production)
streamlit run boe_supervisor_dashboard.py --server.port 8505 --server.address 127.0.0.1

# Specific network interface
streamlit run boe_supervisor_dashboard.py --server.port 8505 --server.address YOUR_INTERNAL_IP
```

### Environment Variables
```bash
# Set environment variables for production
export STREAMLIT_SERVER_PORT=8505
export STREAMLIT_SERVER_ADDRESS=127.0.0.1
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

## 📊 Feature Verification

### 1. Main Dashboard Verification
- [ ] Dashboard loads without errors
- [ ] Document upload functionality works
- [ ] Risk analysis completes successfully
- [ ] All tabs are accessible (Risk Analysis, Technical Validation, Supervisor Dashboard, Reports & Export)

### 2. Technical Validation Verification
- [ ] Technical validation tab loads
- [ ] Settings configuration works
- [ ] Statistical validation runs successfully
- [ ] Results display correctly
- [ ] Export functionality works

### 3. Data Quality Checks
- [ ] Risk scores extract correctly from analysis
- [ ] Statistical tests complete without errors
- [ ] Confidence intervals calculate properly
- [ ] Model performance metrics display

## 🔍 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
netstat -ano | findstr :8505  # Windows
lsof -i :8505                 # Linux/Mac

# Kill process if needed
taskkill /PID <PID> /F        # Windows
kill -9 <PID>                # Linux/Mac
```

#### Import Errors
```bash
# Verify all dependencies installed
pip list | grep streamlit
pip list | grep pandas
pip list | grep plotly

# Reinstall if needed
pip install --upgrade streamlit pandas plotly scipy scikit-learn
```

#### Memory Issues
```bash
# Monitor memory usage
# Increase system memory or reduce data size if needed
# Consider processing smaller batches of documents
```

### Error Logs
```bash
# Check Streamlit logs
streamlit run boe_supervisor_dashboard.py --logger.level debug

# Python logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 📈 Performance Optimization

### Production Settings
```bash
# Optimize for production
streamlit run boe_supervisor_dashboard.py \
  --server.port 8505 \
  --server.maxUploadSize 200 \
  --server.maxMessageSize 200 \
  --browser.gatherUsageStats false
```

### Memory Management
- **Document Size**: Limit uploads to 200MB per file
- **Batch Processing**: Process documents in smaller batches for large datasets
- **Session Management**: Clear session state periodically for long-running sessions

## 🔄 Updates and Maintenance

### Version Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
# Stop current processes and restart with new version
```

### Backup Procedures
```bash
# Backup configuration files
cp -r data_science/scripts/ backup/scripts_$(date +%Y%m%d)/
cp -r config/ backup/config_$(date +%Y%m%d)/

# Backup processed data (if applicable)
cp -r data/ backup/data_$(date +%Y%m%d)/
```

## 📋 Health Checks

### Automated Health Check Script
```bash
#!/bin/bash
# health_check.sh

# Check if dashboard is responding
curl -f http://localhost:8505/_stcore/health || exit 1

# Check if technical validation is working
python -c "
from data_science.scripts.statistical_validation.statistical_validation_engine import StatisticalValidationEngine
engine = StatisticalValidationEngine()
print('Technical validation engine: OK')
"

echo "Health check passed"
```

### Monitoring
- **Response Time**: Monitor dashboard load times
- **Memory Usage**: Track memory consumption during analysis
- **Error Rates**: Monitor for processing errors
- **User Sessions**: Track concurrent user sessions

## 🆘 Support

### Technical Support
- **Repository Issues**: https://github.com/daleparr/Bank-of-England-Mosaic-Lens/issues
- **Documentation**: See README.md and release notes
- **Logs**: Check Streamlit logs for detailed error information

### Emergency Procedures
1. **Service Down**: Restart dashboard services
2. **Memory Issues**: Reduce document batch sizes
3. **Data Corruption**: Restore from backup
4. **Performance Issues**: Check system resources and optimize

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] System requirements verified
- [ ] Dependencies installed
- [ ] Network configuration set
- [ ] Security settings configured
- [ ] Backup procedures in place

### Post-Deployment
- [ ] Dashboard accessibility verified
- [ ] All features tested
- [ ] Performance benchmarks met
- [ ] Monitoring systems active
- [ ] Support procedures documented

### Production Readiness
- [ ] Load testing completed
- [ ] Security review passed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Rollback procedures tested

---

**For additional support or questions, please refer to the repository documentation or open an issue on GitHub.**

**Repository**: https://github.com/daleparr/Bank-of-England-Mosaic-Lens  
**Version**: 2.1.0  
**Deployment Guide Version**: 1.0