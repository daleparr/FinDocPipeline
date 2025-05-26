import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting ETL pipeline test...")
        
        # Add project root to Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Python version: {sys.version}")
        
        # Try to import ETLPipeline
        try:
            from src.etl.etl_pipeline import ETLPipeline
            logger.info("Successfully imported ETLPipeline")
            
            # Try to initialize the pipeline
            logger.info("Initializing ETL pipeline...")
            pipeline = ETLPipeline()
            logger.info("ETL pipeline initialized successfully")
            
            # Try to process banks
            logger.info("Starting to process banks...")
            pipeline.process_all_banks()
            logger.info("Successfully processed all banks")
            
        except Exception as e:
            logger.error(f"Error during ETL pipeline execution: {str(e)}", exc_info=True)
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
