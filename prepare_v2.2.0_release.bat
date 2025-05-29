@echo off
REM Bank of England Mosaic Lens v2.2.0 Release Script (Windows)
REM This script prepares and creates the v2.2.0 release

echo 🚀 Bank of England Mosaic Lens v2.2.0 Release Preparation
echo ==========================================================

REM Check if we're in the right directory
if not exist "VERSION" (
    echo ❌ Error: VERSION file not found. Please run from project root.
    exit /b 1
)

REM Verify current version
set /p CURRENT_VERSION=<VERSION
echo 📋 Current version: %CURRENT_VERSION%

if not "%CURRENT_VERSION%"=="2.2.0" (
    echo ❌ Error: VERSION file should contain 2.2.0, found: %CURRENT_VERSION%
    exit /b 1
)

echo ✅ Version verification passed

REM Verify critical files exist
echo 📁 Verifying release files...

if exist "RELEASE_NOTES_v2.2.0.md" (
    echo ✅ RELEASE_NOTES_v2.2.0.md
) else (
    echo ❌ Missing: RELEASE_NOTES_v2.2.0.md
    exit /b 1
)

if exist "DEPLOYMENT_GUIDE_v2.2.0.md" (
    echo ✅ DEPLOYMENT_GUIDE_v2.2.0.md
) else (
    echo ❌ Missing: DEPLOYMENT_GUIDE_v2.2.0.md
    exit /b 1
)

if exist "CHANGELOG_v2.2.0.md" (
    echo ✅ CHANGELOG_v2.2.0.md
) else (
    echo ❌ Missing: CHANGELOG_v2.2.0.md
    exit /b 1
)

if exist "requirements_v2.2.0.txt" (
    echo ✅ requirements_v2.2.0.txt
) else (
    echo ❌ Missing: requirements_v2.2.0.txt
    exit /b 1
)

REM Verify market intelligence components
echo 🔍 Verifying market intelligence components...

if exist "src\market_intelligence\__init__.py" (
    echo ✅ src\market_intelligence\__init__.py
) else (
    echo ❌ Missing: src\market_intelligence\__init__.py
    exit /b 1
)

if exist "src\market_intelligence\gsib_monitor.py" (
    echo ✅ src\market_intelligence\gsib_monitor.py
) else (
    echo ❌ Missing: src\market_intelligence\gsib_monitor.py
    exit /b 1
)

if exist "src\market_intelligence\yahoo_finance_client.py" (
    echo ✅ src\market_intelligence\yahoo_finance_client.py
) else (
    echo ❌ Missing: src\market_intelligence\yahoo_finance_client.py
    exit /b 1
)

REM Test market intelligence functionality
echo 🧪 Testing market intelligence components...
python -c "from src.market_intelligence import gsib_monitor; print('G-SIB Monitor: OK')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ G-SIB Monitor import test passed
) else (
    echo ❌ G-SIB Monitor import test failed
    exit /b 1
)

python -c "from src.market_intelligence import yahoo_finance_client; print('Yahoo Finance Client: OK')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Yahoo Finance Client import test passed
) else (
    echo ❌ Yahoo Finance Client import test failed
    exit /b 1
)

echo.
echo 🎉 Release v2.2.0 preparation completed successfully!
echo.
echo 📋 Next Steps:
echo 1. Review the files and commit changes:
echo    git add .
echo    git commit -m "Release v2.2.0: Market Intelligence & G-SIB Monitoring"
echo.
echo 2. Create and push release tag:
echo    git tag -a v2.2.0 -m "Bank of England Mosaic Lens v2.2.0"
echo    git push origin main
echo    git push origin v2.2.0
echo.
echo 3. Create GitHub release:
echo    - Go to: https://github.com/daleparr/Bank-of-England-Mosaic-Lens/releases
echo    - Click 'Create a new release'
echo    - Select tag: v2.2.0
echo    - Title: Bank of England Mosaic Lens v2.2.0 - Market Intelligence & G-SIB Monitoring
echo    - Copy content from RELEASE_NOTES_v2.2.0.md
echo.
echo 4. Verify deployment:
echo    streamlit run main_dashboard.py --server.port 8514
echo.
echo 🚀 Release v2.2.0 is ready for deployment!

pause