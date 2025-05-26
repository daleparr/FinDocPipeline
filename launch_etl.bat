@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting Pure ETL Frontend...
streamlit run pure_etl_frontend.py

pause