#!/usr/bin/env python3
"""
Extract Processed Data to DataFrame

This script extracts all the processed data from the ETL pipeline and loads it into a pandas DataFrame for review.
"""

import sys
import pandas as pd
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_all_processed_data():
    """Extract all processed data from the ETL pipeline into a single DataFrame."""
    
    try:
        from src.etl.etl_pipeline import ETLPipeline
        from src.etl.config import ConfigManager
        from src.etl.parsers.pdf_parser import PDFParser
        from src.etl.parsers.excel_parser import ExcelParser
        from src.etl.schema_transformer import SchemaTransformer
        
        logger.info("🔄 Extracting processed data from ETL pipeline...")
        
        # Initialize components
        config_manager = ConfigManager()
        config = config_manager.get_config()
        schema_transformer = SchemaTransformer()
        
        # Raw data directory
        raw_data_dir = Path("data/raw")
        
        all_records = []
        
        # Process Q1 2025 data
        q1_2025_dir = raw_data_dir / "Q1_2025"
        if q1_2025_dir.exists():
            logger.info(f"📁 Processing Q1 2025 data from {q1_2025_dir}")
            
            for file_path in q1_2025_dir.glob("*"):
                if file_path.is_file():
                    logger.info(f"📄 Processing {file_path.name}")
                    
                    try:
                        # Determine parser based on file extension
                        if file_path.suffix.lower() == '.pdf':
                            parser = PDFParser("RawDataBank", "Q1_2025")
                            parsed_data = parser.parse(file_path)
                        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                            parser = ExcelParser("RawDataBank", "Q1_2025")
                            parsed_data = parser.parse(file_path)
                        else:
                            logger.warning(f"Unsupported file type: {file_path.suffix}")
                            continue
                        
                        # Transform to NLP schema
                        nlp_records = schema_transformer.transform_parsed_data(
                            parsed_data, "RawDataBank", "Q1_2025"
                        )
                        
                        # Add source file info
                        for record in nlp_records:
                            record['source_file'] = file_path.name
                            record['quarter_period'] = 'Q1_2025'
                        
                        all_records.extend(nlp_records)
                        logger.info(f"✅ Extracted {len(nlp_records)} records from {file_path.name}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {file_path.name}: {e}")
        
        # Process Q4 2024 data
        q4_2024_dir = raw_data_dir / "Q4_2024"
        if q4_2024_dir.exists():
            logger.info(f"📁 Processing Q4 2024 data from {q4_2024_dir}")
            
            for file_path in q4_2024_dir.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.xlsx', '.xls']:
                    logger.info(f"📄 Processing {file_path.name}")
                    
                    try:
                        # Determine parser based on file extension
                        if file_path.suffix.lower() == '.pdf':
                            parser = PDFParser("RawDataBank", "Q4_2024")
                            parsed_data = parser.parse(file_path)
                        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                            parser = ExcelParser("RawDataBank", "Q4_2024")
                            parsed_data = parser.parse(file_path)
                        else:
                            continue
                        
                        # Transform to NLP schema
                        nlp_records = schema_transformer.transform_parsed_data(
                            parsed_data, "RawDataBank", "Q4_2024"
                        )
                        
                        # Add source file info
                        for record in nlp_records:
                            record['source_file'] = file_path.name
                            record['quarter_period'] = 'Q4_2024'
                        
                        all_records.extend(nlp_records)
                        logger.info(f"✅ Extracted {len(nlp_records)} records from {file_path.name}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {file_path.name}: {e}")
        
        # Convert to DataFrame
        if all_records:
            df = pd.DataFrame(all_records)
            logger.info(f"📊 Created DataFrame with {len(df)} total records")
            
            # Reorder columns for better readability
            column_order = [
                'source_file', 'quarter_period', 'bank_name', 'quarter', 'sentence_id',
                'speaker_norm', 'analyst_utterance', 'text', 'word_count', 'sentence_length',
                'source_type', 'call_id', 'sentiment_score', 'sentiment_label',
                'topic_labels', 'named_entities', 'key_phrases', 'processing_date'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in df.columns]
            remaining_columns = [col for col in df.columns if col not in available_columns]
            final_columns = available_columns + remaining_columns
            
            df = df[final_columns]
            
            return df
        else:
            logger.warning("No records found to process")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"💥 Error extracting data: {e}", exc_info=True)
        return pd.DataFrame()

def main():
    """Main function to extract and display processed data."""
    print("🔍 EXTRACTING PROCESSED DATA FROM ETL PIPELINE")
    print("=" * 60)
    
    # Extract data
    df = extract_all_processed_data()
    
    if df.empty:
        print("❌ No data found or extraction failed")
        return
    
    # Display summary
    print(f"\n📊 DATAFRAME SUMMARY")
    print("=" * 60)
    print(f"Total Records: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Show data types
    print(f"\n📋 COLUMN INFORMATION")
    print("=" * 60)
    print(df.dtypes)
    
    # Show value counts for key columns
    if 'source_file' in df.columns:
        print(f"\n📁 RECORDS BY SOURCE FILE")
        print("=" * 60)
        print(df['source_file'].value_counts())
    
    if 'speaker_norm' in df.columns:
        print(f"\n🎤 TOP 10 SPEAKERS")
        print("=" * 60)
        print(df['speaker_norm'].value_counts().head(10))
    
    if 'quarter_period' in df.columns:
        print(f"\n📅 RECORDS BY QUARTER")
        print("=" * 60)
        print(df['quarter_period'].value_counts())
    
    # Display first 50 rows
    print(f"\n📋 FIRST 50 RECORDS")
    print("=" * 60)
    
    # Select key columns for display
    display_columns = [
        'source_file', 'quarter_period', 'sentence_id', 'speaker_norm', 
        'analyst_utterance', 'text', 'word_count'
    ]
    display_columns = [col for col in display_columns if col in df.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', None)
    
    print(df[display_columns].head(50))
    
    # Save to CSV for further analysis
    output_file = "processed_data_complete.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Complete dataset saved to: {output_file}")
    
    # Save summary to separate file
    summary_file = "processed_data_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PROCESSED DATA SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Total Columns: {len(df.columns)}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")
        
        if 'source_file' in df.columns:
            f.write("RECORDS BY SOURCE FILE:\n")
            f.write(str(df['source_file'].value_counts()) + "\n\n")
        
        if 'speaker_norm' in df.columns:
            f.write("TOP SPEAKERS:\n")
            f.write(str(df['speaker_norm'].value_counts().head(15)) + "\n\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    
    return df

if __name__ == "__main__":
    df = main()
    print(f"\n✅ Data extraction complete!")
    print(f"📊 DataFrame available as 'df' variable with {len(df)} records")