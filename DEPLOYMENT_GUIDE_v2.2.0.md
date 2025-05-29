# Bank of England Mosaic Lens v2.2.0 Deployment Guide

## 🚀 Market Intelligence & G-SIB Monitoring Release

**Version**: 2.2.0  
**Release Date**: May 29, 2025  
**Repository**: https://github.com/daleparr/Bank-of-England-Mosaic-Lens

## 📋 Pre-Deployment Checklist

### ✅ System Requirements
- **Python**: 3.8+ (Recommended: 3.11)
- **Memory**: Minimum 8GB RAM (Recommended: 16GB for G-SIB monitoring)
- **Storage**: 10GB free space for market data caching
- **Network**: Stable internet connection for Yahoo Finance API
- **OS**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+

### ✅ Dependencies Verification
```bash
# Core dependencies
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install plotly>=5.15.0
pip install yfinance>=0.2.18
pip install scikit-learn>=1.3.0
pip install pandas-market-calendars>=4.3.0
```

## 🔧 Installation Steps

### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/daleparr/Bank-of-England-Mosaic-Lens.git
cd Bank-of-England-Mosaic-Lens

# Verify version
cat VERSION  # Should show 2.2.0
```

### 2. Environment Configuration
```bash
# Create virtual environment (recommended)
python -m venv boe_env
source boe_env/bin/activate  # Linux/macOS
# OR
boe_env\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 3. Market Intelligence Setup
```bash
# Verify market intelligence components
python -c "from src.market_intelligence import gsib_monitor; print('G-SIB Monitor: OK')"
python -c "from src.market_intelligence import yahoo_finance_client; print('Yahoo Finance: OK')"
python -c "from src.market_intelligence import sentiment_market_correlator; print('Sentiment Correlator: OK')"
```

## 🚀 Deployment Options

### Option 1: Production Deployment
```bash
# Launch main dashboard with market intelligence
streamlit run main_dashboard.py --server.port 8514 --server.headless true
```

### Option 2: Development Deployment
```bash
# Launch with debug mode
streamlit run main_dashboard.py --server.port 8514 --server.runOnSave true
```

### Option 3: Standalone Market Intelligence
```bash
# Launch market intelligence only
python launch_market_intelligence_dashboard.py
```

## 📊 Configuration

### Market Intelligence Settings
Create `config/market_intelligence_config.yaml`:
```yaml
yahoo_finance:
  rate_limit: 2000  # requests per hour
  timeout: 30  # seconds
  retry_attempts: 3

gsib_monitoring:
  update_interval: 300  # seconds (5 minutes)
  correlation_window: 252  # trading days (1 year)
  clustering_method: "kmeans"
  n_clusters: 4

earnings_analysis:
  pre_days: 5
  post_days: 3
  auto_detect_institutions: true
  auto_detect_dates: true
```

### G-SIB Institution Configuration
The system automatically monitors all 33 G-SIB institutions:
- **US Banks**: JPM, BAC, C, WFC, GS, MS, BK, STT
- **European Banks**: HSBC, BCS, DB, CS, UBS, BNP, SAN, ING, etc.
- **Asian Banks**: 8306.T, 8316.T, 8411.T, 3988.HK, 1398.HK, etc.
- **Canadian Banks**: RY, TD, BNS, BMO, CM

## 🔍 Verification Steps

### 1. System Health Check
```bash
# Run comprehensive system test
python test_market_intelligence_standalone.py
```

Expected output:
```
✅ G-SIB Monitor: Operational
✅ Yahoo Finance Client: Connected
✅ Market Data: 33 institutions loaded
✅ Correlation Analysis: Working
✅ Systemic Clustering: 4 clusters identified
✅ Earnings Impact Analysis: Functional
✅ Institution Auto-Detection: Active
```

### 2. Dashboard Verification
1. Navigate to `http://localhost:8514`
2. Click on "Market Intelligence" tab
3. Verify "Market Intelligence data refreshed successfully" message
4. Check that institution dropdown shows auto-detected institutions
5. Confirm earnings date auto-population works

### 3. Data Quality Verification
```bash
# Test market data quality
python -c "
from src.market_intelligence.gsib_monitor import get_gsib_monitor
monitor = get_gsib_monitor()
data = monitor.track_global_gsib_movements()
print(f'Market data quality: {len(data)} institutions, {sum(len(df) for df in data.values())} total data points')
"
```

## 🛡️ Security Configuration

### API Security
- Yahoo Finance API requires no authentication but has rate limits
- Implement IP whitelisting for production environments
- Use HTTPS in production deployments

### Data Security
```bash
# Set appropriate file permissions
chmod 600 config/*.yaml
chmod 700 data/
chmod 755 src/
```

### Network Security
- Configure firewall to allow only necessary ports
- Use reverse proxy (nginx/Apache) for production
- Implement SSL/TLS certificates

## 📈 Performance Optimization

### Memory Optimization
```python
# Add to main_dashboard.py
import streamlit as st
st.set_page_config(
    page_title="BoE Supervisor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable caching for market data
@st.cache_data(ttl=300)  # 5-minute cache
def load_market_data():
    # Market data loading logic
    pass
```

### CPU Optimization
- Enable multiprocessing for correlation calculations
- Use vectorized operations for large datasets
- Implement lazy loading for market data

## 🔧 Troubleshooting

### Common Issues

#### 1. Yahoo Finance Connection Issues
```bash
# Test Yahoo Finance connectivity
python -c "import yfinance as yf; print(yf.download('AAPL', period='1d'))"
```

**Solution**: Check internet connection and Yahoo Finance service status

#### 2. Memory Issues with Large Datasets
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

**Solution**: Increase system RAM or implement data pagination

#### 3. Timezone Issues
```bash
# Verify timezone handling
python -c "
import pandas as pd
print(f'System timezone: {pd.Timestamp.now().tz}')
print(f'Market timezone: America/New_York')
"
```

**Solution**: All timestamp arithmetic uses `pd.Timedelta()` - verify no `timedelta()` usage

### Error Resolution

#### StreamlitDuplicateElementId
- Ensure all Streamlit components have unique keys
- Check for duplicate widget IDs in market intelligence components

#### Import Errors
```bash
# Verify all market intelligence imports
python -c "
from src.market_intelligence import *
print('All market intelligence modules imported successfully')
"
```

## 📊 Monitoring & Maintenance

### Health Monitoring
```bash
# Create monitoring script
cat > monitor_health.py << 'EOF'
import time
import logging
from src.market_intelligence.gsib_monitor import get_gsib_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def health_check():
    try:
        monitor = get_gsib_monitor()
        data = monitor.track_global_gsib_movements(period="1d")
        logger.info(f"Health check passed: {len(data)} institutions monitored")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    while True:
        health_check()
        time.sleep(300)  # Check every 5 minutes
EOF
```

### Log Management
```bash
# Set up log rotation
mkdir -p logs
touch logs/market_intelligence.log
touch logs/dashboard.log
touch logs/errors.log
```

### Backup Strategy
```bash
# Backup critical data
tar -czf backup_$(date +%Y%m%d).tar.gz \
    config/ \
    data/metadata/ \
    logs/ \
    VERSION \
    RELEASE_NOTES_v2.2.0.md
```

## 🔄 Update Procedures

### From v2.1.0 to v2.2.0
```bash
# Backup current installation
cp -r . ../boe_backup_v2.1.0/

# Pull latest changes
git fetch origin
git checkout v2.2.0

# Update dependencies
pip install -r requirements.txt --upgrade

# Verify market intelligence components
python test_market_intelligence_standalone.py

# Restart services
pkill -f streamlit
streamlit run main_dashboard.py --server.port 8514
```

## 📞 Support

### Technical Support
- **GitHub Issues**: https://github.com/daleparr/Bank-of-England-Mosaic-Lens/issues
- **Documentation**: See README.md and technical documentation
- **Community**: GitHub Discussions for community support

### Emergency Contacts
- **Critical Issues**: Create GitHub issue with "urgent" label
- **Security Issues**: Follow responsible disclosure guidelines
- **Performance Issues**: Include system specifications and logs

---

**Deployment Guide Version**: 2.2.0  
**Last Updated**: May 29, 2025  
**Next Review**: June 29, 2025