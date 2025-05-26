#!/usr/bin/env python3
"""
ETL Pipeline Runner for Raw Data Directory

This script runs the ETL pipeline specifically on the raw data in the data/raw directory
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_etl_on_raw_data():
    """Run ETL pipeline on the raw data directory."""
    try:
        from src.etl.etl_pipeline import ETLPipeline
        from src.etl.config import ConfigManager
        
        logger.info("🚀 Starting ETL Pipeline on Raw Data")
        logger.info("=" * 60)
        
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Initialize ETL pipeline with raw data directory
        raw_data_dir = Path("data/raw")
        processed_data_dir = Path("data/processed")
        
        logger.info(f"Raw data directory: {raw_data_dir}")
        logger.info(f"Processed data directory: {processed_data_dir}")
        
        # Create ETL pipeline instance
        pipeline = ETLPipeline()
        
        # Discover files in raw data directory
        logger.info("🔍 Discovering files in raw data directory...")
        
        # Process Q1 2025 data
        q1_2025_dir = raw_data_dir / "Q1_2025"
        if q1_2025_dir.exists():
            logger.info(f"📁 Processing Q1 2025 data from {q1_2025_dir}")
            files = list(q1_2025_dir.glob("*"))
            logger.info(f"Found {len(files)} files: {[f.name for f in files]}")
            
            for file_path in files:
                if file_path.is_file():
                    # Determine document type from filename
                    filename = file_path.name.lower()
                    if 'transcript' in filename:
                        doc_type = 'transcript'
                    elif 'presentation' in filename:
                        doc_type = 'presentation'
                    elif 'supplement' in filename:
                        doc_type = 'supplement'
                    elif 'results' in filename:
                        doc_type = 'results'
                    else:
                        doc_type = 'other'
                    
                    logger.info(f"📄 Processing {file_path.name} as {doc_type}")
                    
                    try:
                        pipeline._process_discovered_file(
                            bank_name="RawDataBank",
                            quarter="Q1_2025", 
                            file_path=file_path,
                            doc_type=doc_type
                        )
                        logger.info(f"✅ Successfully processed {file_path.name}")
                    except Exception as e:
                        logger.error(f"❌ Error processing {file_path.name}: {e}")
        
        # Process Q4 2024 data
        q4_2024_dir = raw_data_dir / "Q4_2024"
        if q4_2024_dir.exists():
            logger.info(f"📁 Processing Q4 2024 data from {q4_2024_dir}")
            files = list(q4_2024_dir.glob("*"))
            logger.info(f"Found {len(files)} files: {[f.name for f in files]}")
            
            for file_path in files:
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.xlsx', '.xls', '.txt', '.json']:
                    # Determine document type from filename
                    filename = file_path.name.lower()
                    if 'transcript' in filename:
                        doc_type = 'transcript'
                    elif 'presentation' in filename:
                        doc_type = 'presentation'
                    elif 'supplement' in filename:
                        doc_type = 'supplement'
                    elif 'results' in filename:
                        doc_type = 'results'
                    else:
                        doc_type = 'other'
                    
                    logger.info(f"📄 Processing {file_path.name} as {doc_type}")
                    
                    try:
                        pipeline._process_discovered_file(
                            bank_name="RawDataBank",
                            quarter="Q4_2024", 
                            file_path=file_path,
                            doc_type=doc_type
                        )
                        logger.info(f"✅ Successfully processed {file_path.name}")
                    except Exception as e:
                        logger.error(f"❌ Error processing {file_path.name}: {e}")
        
        logger.info("🎉 ETL Pipeline processing complete!")
        logger.info("📊 Check the following locations for results:")
        logger.info(f"  • Processed data: {processed_data_dir}")
        logger.info(f"  • Metadata: data/metadata/versions/")
        logger.info("💡 Run 'python view_processed_data.py' to review the results")
        
    except Exception as e:
        logger.error(f"💥 ETL Pipeline failed: {e}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    success = run_etl_on_raw_data()
    if success:
        print("\n🎯 ETL Pipeline completed successfully!")
        print("Run 'python view_processed_data.py' to review your structured data tables.")
    else:
        print("\n❌ ETL Pipeline failed. Check the logs above for details.")
        sys.exit(1)