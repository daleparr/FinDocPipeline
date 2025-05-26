import os
import sys
import traceback

def main():
    try:
        # Add project root to Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        print("=" * 80)
        print("STARTING ETL PIPELINE DEBUG")
        print("=" * 80)
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Project root: {project_root}")
        
        # Try to import ETLPipeline
        try:
            print("\nAttempting to import ETLPipeline...")
            from src.etl.etl_pipeline import ETLPipeline
            print("Successfully imported ETLPipeline")
            
            # Initialize the pipeline
            print("\nInitializing ETL pipeline...")
            pipeline = ETLPipeline()
            print("ETL pipeline initialized successfully")
            
            # Process all banks
            print("\nStarting to process all banks...")
            pipeline.process_all_banks()
            print("\nSuccessfully processed all banks")
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("\nTRACEBACK:")
            traceback.print_exc()
            return 1
            
        print("\n" + "=" * 80)
        print("ETL PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("\nTRACEBACK:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
