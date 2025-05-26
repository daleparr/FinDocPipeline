#!/usr/bin/env python3
"""
Test script for the schema transformation functionality.
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

def test_schema_transformer():
    """Test the schema transformer with sample data."""
    try:
        from src.etl.schema_transformer import SchemaTransformer
        
        logger.info("Testing Schema Transformer...")
        
        # Create transformer
        transformer = SchemaTransformer()
        
        # Test with sample PDF-like data
        pdf_data = {
            'bank': 'TestBank',
            'quarter': 'Q1_2025',
            'file_path': '/test/path/transcript.pdf',
            'document_type': 'transcript',
            'content': {
                'pages': [
                    {
                        'page_number': 1,
                        'text': 'John Smith: Welcome to our Q1 earnings call. Jane Doe: Thank you John.'
                    }
                ]
            },
            'metadata': {'page_count': 1}
        }
        
        # Transform to NLP schema
        nlp_records = transformer.transform_parsed_data(pdf_data, 'TestBank', 'Q1_2025')
        
        logger.info(f"Generated {len(nlp_records)} NLP records")
        
        # Verify schema compliance
        if nlp_records:
            sample_record = nlp_records[0]
            required_fields = [
                'bank_name', 'quarter', 'call_id', 'source_type', 'file_path',
                'speaker_norm', 'analyst_utterance', 'sentence_id', 'text',
                'sentence_length', 'word_count', 'processing_date', 'processing_version'
            ]
            
            missing_fields = [field for field in required_fields if field not in sample_record]
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                return False
            
            logger.info("✅ Schema transformation test passed!")
            logger.info(f"Sample record: {sample_record}")
            return True
        else:
            logger.error("No records generated")
            return False
            
    except Exception as e:
        logger.error(f"Schema transformer test failed: {e}", exc_info=True)
        return False

def test_excel_parser():
    """Test the Excel parser."""
    try:
        from src.etl.parsers.excel_parser import ExcelParser
        import tempfile
        import pandas as pd
        
        logger.info("Testing Excel Parser...")
        
        # Create a test Excel file
        df = pd.DataFrame({
            'Metric': ['Revenue', 'Net Income', 'EPS'],
            'Q1_2024': [1000, 100, 1.50],
            'Q1_2025': [1150, 120, 1.75]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df.to_excel(tmp.name, index=False)
            test_file = Path(tmp.name)
        
        try:
            # Parse the file
            parser = ExcelParser('TestBank', 'Q1_2025')
            result = parser.parse(test_file)
            
            logger.info(f"✅ Excel parser test passed!")
            logger.info(f"Parsed {len(result['content']['tables'])} sheets")
            return True
            
        finally:
            test_file.unlink()  # Clean up
            
    except Exception as e:
        logger.error(f"Excel parser test failed: {e}", exc_info=True)
        return False

def test_text_parser():
    """Test the Text parser."""
    try:
        from src.etl.parsers.text_parser import TextParser
        import tempfile
        
        logger.info("Testing Text Parser...")
        
        # Create a test transcript file
        transcript_content = """John Smith: Welcome to our Q1 2025 earnings call.
Jane Doe: Thank you John. Our revenue increased by 15% this quarter.
Analyst Mike: Can you provide more details on the growth drivers?
Jane Doe: The growth was primarily driven by our new product launches."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_transcript.txt', delete=False) as tmp:
            tmp.write(transcript_content)
            test_file = Path(tmp.name)
        
        try:
            # Parse the file
            parser = TextParser('TestBank', 'Q1_2025')
            result = parser.parse(test_file)
            
            logger.info(f"✅ Text parser test passed!")
            logger.info(f"Found {len(result['speakers'])} speakers: {result['speakers']}")
            logger.info(f"Generated {len(result['segments'])} segments")
            return True
            
        finally:
            test_file.unlink()  # Clean up
            
    except Exception as e:
        logger.error(f"Text parser test failed: {e}", exc_info=True)
        return False

def test_end_to_end():
    """Test end-to-end processing with schema transformation."""
    try:
        from src.etl.parsers.text_parser import TextParser
        from src.etl.schema_transformer import SchemaTransformer
        import tempfile
        
        logger.info("Testing End-to-End Processing...")
        
        # Create test data
        transcript_content = """John Smith: Welcome to our Q1 2025 earnings call.
Jane Doe: Our revenue increased by 15% this quarter to $1.2 billion."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_transcript.txt', delete=False) as tmp:
            tmp.write(transcript_content)
            test_file = Path(tmp.name)
        
        try:
            # Step 1: Parse the file
            parser = TextParser('TestBank', 'Q1_2025')
            parsed_data = parser.parse(test_file)
            
            # Step 2: Transform to NLP schema
            transformer = SchemaTransformer()
            nlp_records = transformer.transform_parsed_data(parsed_data, 'TestBank', 'Q1_2025')
            
            # Step 3: Verify results
            if nlp_records and len(nlp_records) > 0:
                logger.info(f"✅ End-to-end test passed!")
                logger.info(f"Generated {len(nlp_records)} NLP records from transcript")
                
                # Show sample record
                sample = nlp_records[0]
                logger.info(f"Sample record - Speaker: {sample['speaker_norm']}, Text: {sample['text'][:50]}...")
                return True
            else:
                logger.error("No NLP records generated")
                return False
                
        finally:
            test_file.unlink()  # Clean up
            
    except Exception as e:
        logger.error(f"End-to-end test failed: {e}", exc_info=True)
        return False

def main():
    """Run all tests."""
    logger.info("Starting ETL Implementation Tests...")
    
    tests = [
        ("Schema Transformer", test_schema_transformer),
        ("Excel Parser", test_excel_parser),
        ("Text Parser", test_text_parser),
        ("End-to-End Processing", test_end_to_end),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! ETL implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())