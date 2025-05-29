#!/bin/bash

# Bank of England Mosaic Lens v2.2.0 Release Script
# This script prepares and creates the v2.2.0 release

set -e  # Exit on any error

echo "🚀 Bank of England Mosaic Lens v2.2.0 Release Preparation"
echo "=========================================================="

# Check if we're in the right directory
if [ ! -f "VERSION" ]; then
    echo "❌ Error: VERSION file not found. Please run from project root."
    exit 1
fi

# Verify current version
CURRENT_VERSION=$(cat VERSION)
echo "📋 Current version: $CURRENT_VERSION"

if [ "$CURRENT_VERSION" != "2.2.0" ]; then
    echo "❌ Error: VERSION file should contain 2.2.0, found: $CURRENT_VERSION"
    exit 1
fi

echo "✅ Version verification passed"

# Check if git is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: Working directory is not clean. Uncommitted changes detected."
    echo "   Continuing with release preparation..."
fi

# Verify critical files exist
echo "📁 Verifying release files..."
REQUIRED_FILES=(
    "RELEASE_NOTES_v2.2.0.md"
    "DEPLOYMENT_GUIDE_v2.2.0.md" 
    "CHANGELOG_v2.2.0.md"
    "requirements_v2.2.0.txt"
    "VERSION"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Verify market intelligence components
echo "🔍 Verifying market intelligence components..."
MARKET_INTEL_FILES=(
    "src/market_intelligence/__init__.py"
    "src/market_intelligence/gsib_monitor.py"
    "src/market_intelligence/yahoo_finance_client.py"
    "src/market_intelligence/sentiment_market_correlator.py"
    "src/market_intelligence/market_indicators.py"
    "src/market_intelligence/market_intelligence_dashboard.py"
)

for file in "${MARKET_INTEL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Test market intelligence functionality
echo "🧪 Testing market intelligence components..."
if python -c "from src.market_intelligence import gsib_monitor; print('G-SIB Monitor: OK')" 2>/dev/null; then
    echo "✅ G-SIB Monitor import test passed"
else
    echo "❌ G-SIB Monitor import test failed"
    exit 1
fi

if python -c "from src.market_intelligence import yahoo_finance_client; print('Yahoo Finance Client: OK')" 2>/dev/null; then
    echo "✅ Yahoo Finance Client import test passed"
else
    echo "❌ Yahoo Finance Client import test failed"
    exit 1
fi

# Create git tag and commit
echo "📝 Preparing git operations..."

# Add all release files
git add VERSION
git add RELEASE_NOTES_v2.2.0.md
git add DEPLOYMENT_GUIDE_v2.2.0.md
git add CHANGELOG_v2.2.0.md
git add requirements_v2.2.0.txt
git add prepare_v2.2.0_release.sh
git add src/market_intelligence/

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "📝 Committing release changes..."
    git commit -m "Release v2.2.0: Market Intelligence & G-SIB Monitoring

🚀 Major Features:
- Complete Market Intelligence dashboard with G-SIB monitoring
- Real-time tracking of 33 Global Systemically Important Banks
- Yahoo Finance API integration for live market data
- Advanced earnings impact analysis with sentiment correlation
- Institution auto-detection from ETL processing runs
- Intelligent earnings date suggestions

🔧 Technical Improvements:
- Resolved all timezone arithmetic errors
- Enhanced error handling and data validation
- Optimized performance for real-time market analysis
- Production-ready regulatory intelligence capabilities

🏦 Market Intelligence Components:
- G-SIB Monitor: Real-time systemic risk monitoring
- Yahoo Finance Client: Live market data integration
- Sentiment Correlator: NLP-market correlation analysis
- Market Indicators: Advanced technical analysis
- Intelligence Dashboard: Integrated supervisory interface

Closes: Market Intelligence implementation
Fixes: All timezone and timestamp arithmetic issues"
fi

# Create release tag
echo "🏷️  Creating release tag..."
if git tag -l | grep -q "^v2.2.0$"; then
    echo "⚠️  Tag v2.2.0 already exists. Deleting and recreating..."
    git tag -d v2.2.0
fi

git tag -a v2.2.0 -m "Bank of England Mosaic Lens v2.2.0

🚀 Market Intelligence & G-SIB Monitoring Release

Major Features:
✅ Real-time tracking of 33 Global Systemically Important Banks
✅ Yahoo Finance API integration for live market data feeds
✅ Advanced earnings impact analysis with sentiment correlation
✅ Institution auto-detection from ETL processing runs
✅ Intelligent earnings date suggestions based on quarter metadata
✅ Systemic risk clustering using machine learning algorithms
✅ Cross-market correlation analysis and contagion risk assessment

Technical Achievements:
✅ Complete resolution of all timezone arithmetic errors
✅ Production-grade error handling and data validation
✅ Scalable architecture for real-time market surveillance
✅ Comprehensive regulatory intelligence capabilities
✅ Enhanced performance for large-scale financial data processing

Repository: https://github.com/daleparr/Bank-of-England-Mosaic-Lens
Release Notes: RELEASE_NOTES_v2.2.0.md
Deployment Guide: DEPLOYMENT_GUIDE_v2.2.0.md"

echo "✅ Release tag v2.2.0 created successfully"

# Display next steps
echo ""
echo "🎉 Release v2.2.0 preparation completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Review the commit and tag:"
echo "   git log --oneline -1"
echo "   git show v2.2.0"
echo ""
echo "2. Push to repository:"
echo "   git push origin main"
echo "   git push origin v2.2.0"
echo ""
echo "3. Create GitHub release:"
echo "   - Go to: https://github.com/daleparr/Bank-of-England-Mosaic-Lens/releases"
echo "   - Click 'Create a new release'"
echo "   - Select tag: v2.2.0"
echo "   - Title: Bank of England Mosaic Lens v2.2.0 - Market Intelligence & G-SIB Monitoring"
echo "   - Copy content from RELEASE_NOTES_v2.2.0.md"
echo "   - Attach: DEPLOYMENT_GUIDE_v2.2.0.md and CHANGELOG_v2.2.0.md"
echo "   - Mark as 'Latest release'"
echo ""
echo "4. Verify deployment:"
echo "   streamlit run main_dashboard.py --server.port 8514"
echo ""
echo "🚀 Release v2.2.0 is ready for deployment!"