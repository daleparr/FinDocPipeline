import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Run the pipeline
from src.etl.etl_pipeline import ETLPipeline

if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.process_all_banks()
