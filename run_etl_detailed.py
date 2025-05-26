import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Add project root to Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        logger.info("=" * 80)
        logger.info("STARTING ETL PIPELINE")
        logger.info("=" * 80)
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Import required modules
        try:
            logger.info("Importing ETLPipeline...")
            from src.etl.etl_pipeline import ETLPipeline
            logger.info("Successfully imported ETLPipeline")
            
            # Initialize the pipeline
            logger.info("Initializing ETL pipeline...")
            pipeline = ETLPipeline()
            logger.info("ETL pipeline initialized successfully")
            
            # Process all banks
            logger.info("Starting to process all banks...")
            pipeline.process_all_banks()
            logger.info("Successfully processed all banks")
            
        except Exception as e:
            logger.error(f"Error during ETL pipeline execution: {str(e)}", exc_info=True)
            return 1
            
        logger.info("=" * 80)
        logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
