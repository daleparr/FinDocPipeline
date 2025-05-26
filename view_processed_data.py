#!/usr/bin/env python3
"""
Data Viewer Script for ETL Pipeline Processed Data

This script helps you review the structured data tables created by the ETL pipeline.
It shows the NLP-ready data in various formats and provides analysis capabilities.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_processed_data():
    """Find all processed data files and metadata."""
    data_dir = Path("data")
    
    # Find metadata files
    metadata_files = []
    if (data_dir / "metadata" / "versions").exists():
        for metadata_file in (data_dir / "metadata" / "versions").rglob("processing_metadata.json"):
            metadata_files.append(metadata_file)
    
    # Find processed data files
    processed_files = []
    if (data_dir / "processed").exists():
        for processed_file in (data_dir / "processed").rglob("*.parquet"):
            processed_files.append(processed_file)
        for processed_file in (data_dir / "processed").rglob("*.csv"):
            processed_files.append(processed_file)
        for processed_file in (data_dir / "processed").rglob("*.json"):
            processed_files.append(processed_file)
    
    return metadata_files, processed_files

def view_metadata(metadata_file):
    """Display metadata information."""
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 METADATA: {metadata_file}")
        print("=" * 60)
        print(f"Bank: {metadata.get('bank_name', 'Unknown')}")
        print(f"Quarter: {metadata.get('quarter', 'Unknown')}")
        print(f"Processing Date: {metadata.get('processing_date', 'Unknown')}")
        print(f"Number of Records: {metadata.get('num_records', 0)}")
        print(f"Pipeline Version: {metadata.get('processing_pipeline', 'Unknown')}")
        
        # Topic distribution
        topic_dist = metadata.get('topic_distribution', {})
        if topic_dist:
            print(f"\n🏷️ Topic Distribution:")
            for topic, count in topic_dist.items():
                print(f"  {topic}: {count}")
        
        # Speaker distribution
        speaker_dist = metadata.get('speaker_distribution', {})
        if speaker_dist:
            print(f"\n🎤 Speaker Distribution:")
            for speaker, count in speaker_dist.items():
                print(f"  {speaker}: {count}")
        
        # Cleaning parameters
        cleaning_params = metadata.get('cleaning_parameters', {})
        if cleaning_params:
            print(f"\n🧹 Text Cleaning Parameters:")
            for param, value in cleaning_params.items():
                print(f"  {param}: {value}")
        
        return metadata
        
    except Exception as e:
        print(f"Error reading metadata {metadata_file}: {e}")
        return None

def view_processed_file(processed_file):
    """Display processed data file contents."""
    try:
        print(f"\n📄 PROCESSED DATA: {processed_file}")
        print("=" * 60)
        
        # Determine file type and read accordingly
        if processed_file.suffix == '.parquet':
            df = pd.read_parquet(processed_file)
        elif processed_file.suffix == '.csv':
            df = pd.read_csv(processed_file)
        elif processed_file.suffix == '.json':
            with open(processed_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                print(f"JSON structure: {json.dumps(data, indent=2)[:500]}...")
                return
        else:
            print(f"Unsupported file format: {processed_file.suffix}")
            return
        
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\n📋 Sample Data (first 3 rows):")
        print(df.head(3).to_string())
        
        # Show NLP schema fields if present
        nlp_fields = [
            'bank_name', 'quarter', 'call_id', 'source_type', 'speaker_norm', 
            'analyst_utterance', 'sentence_id', 'text', 'sentiment_score', 
            'sentiment_label', 'topic_labels', 'named_entities'
        ]
        
        present_nlp_fields = [field for field in nlp_fields if field in df.columns]
        if present_nlp_fields:
            print(f"\n🧠 NLP Schema Fields Present: {present_nlp_fields}")
            
            # Show text samples
            if 'text' in df.columns:
                print(f"\n📝 Text Samples:")
                for i, text in enumerate(df['text'].head(3)):
                    speaker = df.iloc[i].get('speaker_norm', 'Unknown')
                    print(f"  {i+1}. [{speaker}]: {text[:100]}...")
        
        # Show data types
        print(f"\n📊 Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        return df
        
    except Exception as e:
        print(f"Error reading processed file {processed_file}: {e}")
        return None

def create_sample_nlp_data():
    """Create a sample of what the NLP-ready data should look like."""
    print(f"\n🔬 CREATING SAMPLE NLP DATA")
    print("=" * 60)
    
    try:
        from src.etl.parsers.text_parser import TextParser
        from src.etl.schema_transformer import SchemaTransformer
        import tempfile
        
        # Create sample transcript content
        sample_content = """John Smith: Welcome to our Q1 2025 earnings call. I'm pleased to report strong financial results.
Jane Doe: Thank you John. Our revenue increased by 15% to $1.2 billion this quarter.
Analyst Mike: Can you provide more details on the revenue growth drivers?
Jane Doe: Certainly. The growth was primarily driven by our new product launches and improved market penetration."""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_transcript.txt', delete=False) as f:
            f.write(sample_content)
            temp_file = Path(f.name)
        
        try:
            # Parse and transform
            parser = TextParser('SampleBank', 'Q1_2025')
            parsed_data = parser.parse(temp_file)
            
            transformer = SchemaTransformer()
            nlp_records = transformer.transform_parsed_data(parsed_data, 'SampleBank', 'Q1_2025')
            
            # Convert to DataFrame for display
            df = pd.DataFrame(nlp_records)
            
            print(f"Generated {len(nlp_records)} NLP records")
            print(f"Schema fields: {list(df.columns)}")
            
            # Show sample records
            print(f"\n📋 Sample NLP Records:")
            for i, record in enumerate(nlp_records[:2]):
                print(f"\nRecord {i+1}:")
                print(f"  Bank: {record['bank_name']}")
                print(f"  Quarter: {record['quarter']}")
                print(f"  Speaker: {record['speaker_norm']}")
                print(f"  Text: {record['text']}")
                print(f"  Sentence ID: {record['sentence_id']}")
                print(f"  Word Count: {record['word_count']}")
                print(f"  Source Type: {record['source_type']}")
                print(f"  Analyst Utterance: {record['analyst_utterance']}")
            
            # Save sample data
            sample_file = Path("sample_nlp_data.csv")
            df.to_csv(sample_file, index=False)
            print(f"\n💾 Sample data saved to: {sample_file}")
            
            return df
            
        finally:
            temp_file.unlink()  # Clean up
            
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return None

def analyze_data_structure():
    """Analyze the overall data structure and provide insights."""
    print(f"\n🔍 DATA STRUCTURE ANALYSIS")
    print("=" * 60)
    
    metadata_files, processed_files = find_processed_data()
    
    print(f"Found {len(metadata_files)} metadata files")
    print(f"Found {len(processed_files)} processed data files")
    
    # Analyze metadata
    total_records = 0
    banks = set()
    quarters = set()
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            total_records += metadata.get('num_records', 0)
            banks.add(metadata.get('bank_name', 'Unknown'))
            quarters.add(metadata.get('quarter', 'Unknown'))
            
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")
    
    print(f"\n📈 Summary:")
    print(f"  Total Records Processed: {total_records}")
    print(f"  Banks: {sorted(banks)}")
    print(f"  Quarters: {sorted(quarters)}")
    
    # Show file locations
    if metadata_files:
        print(f"\n📁 Metadata Locations:")
        for f in metadata_files[-3:]:  # Show last 3
            print(f"  {f}")
    
    if processed_files:
        print(f"\n📁 Processed Data Locations:")
        for f in processed_files[-3:]:  # Show last 3
            print(f"  {f}")

def main():
    """Main function to run the data viewer."""
    print("🔍 ETL PIPELINE DATA VIEWER")
    print("=" * 60)
    print("This script helps you review the structured data tables created by the ETL pipeline.")
    
    # Find all data
    metadata_files, processed_files = find_processed_data()
    
    if not metadata_files and not processed_files:
        print("\n⚠️ No processed data found. Let me create a sample to show you the expected structure.")
        create_sample_nlp_data()
        return
    
    # Show overall analysis
    analyze_data_structure()
    
    # Show metadata
    print(f"\n" + "="*60)
    print("METADATA FILES")
    print("="*60)
    
    for metadata_file in metadata_files[-3:]:  # Show last 3
        view_metadata(metadata_file)
    
    # Show processed files
    if processed_files:
        print(f"\n" + "="*60)
        print("PROCESSED DATA FILES")
        print("="*60)
        
        for processed_file in processed_files[-2:]:  # Show last 2
            df = view_processed_file(processed_file)
    
    # Create sample data to show expected structure
    print(f"\n" + "="*60)
    print("SAMPLE NLP DATA STRUCTURE")
    print("="*60)
    create_sample_nlp_data()
    
    print(f"\n✅ Data review complete!")
    print(f"\n📍 Key Locations:")
    print(f"  • Metadata: data/metadata/versions/")
    print(f"  • Processed Data: data/processed/")
    print(f"  • Sample Data: sample_nlp_data.csv")

if __name__ == "__main__":
    main()