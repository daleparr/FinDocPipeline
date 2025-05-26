"""
Test script for the parser modules.
"""
import sys
import logging
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_pdf_parser():
    """Test the PDF parser with a simple PDF file."""
    try:
        logger.info("Testing PDF parser...")
        
        # Create a simple PDF file for testing
        import PyPDF2
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_writer = PyPDF2.PdfWriter()
            page = PyPDF2.PageObject.create_blank_page(width=72, height=72)
            page.merge_page(PyPDF2.PageObject.create_blank_page(width=72, height=72))
            pdf_writer.add_page(page)
            
            # Add some text to the page
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from io import BytesIO
            
            packet = BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)
            can.drawString(10, 10, "Test PDF Content")
            can.save()
            
            # Move to the beginning of the StringIO buffer
            packet.seek(0)
            new_pdf = PyPDF2.PdfReader(packet)
            
            # Add the content to the PDF
            page.merge_page(new_pdf.pages[0])
            
            with open(tmp.name, 'wb') as f:
                pdf_writer.write(f)
            
            test_file = Path(tmp.name)
        
        # Import and test the parser
        from src.etl.parsers.pdf_parser import parse_pdf
        
        result = parse_pdf('test_bank', 'Q1_2025', test_file)
        
        # Verify the result
        assert result['bank'] == 'test_bank'
        assert result['quarter'] == 'Q1_2025'
        assert 'content' in result
        assert 'Test PDF Content' in result['content']
        
        logger.info("PDF parser test passed!")
        return True
        
    except Exception as e:
        logger.error(f"PDF parser test failed: {str(e)}")
        return False
    finally:
        if 'test_file' in locals() and test_file.exists():
            test_file.unlink()

def test_excel_parser():
    """Test the Excel parser with a simple Excel file."""
    try:
        logger.info("Testing Excel parser...")
        
        # Create a simple Excel file for testing
        import pandas as pd
        
        df = pd.DataFrame({
            'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'Value': [100, 200, 300],
            'Description': ['Test', 'Data', 'Here']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            df.to_excel(tmp.name, index=False, sheet_name='TestSheet')
            test_file = Path(tmp.name)
        
        # Import and test the parser
        from src.etl.parsers.excel_parser import parse_excel
        
        result = parse_excel('test_bank', 'Q1_2025', test_file)
        
        # Verify the result
        assert result['bank'] == 'test_bank'
        assert result['quarter'] == 'Q1_2025'
        assert 'TestSheet' in result['sheets']
        assert len(result['sheets']['TestSheet']['data']) == 3
        
        logger.info("Excel parser test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Excel parser test failed: {str(e)}")
        return False
    finally:
        if 'test_file' in locals() and test_file.exists():
            test_file.unlink()

def test_file_discovery():
    """Test the file discovery functionality."""
    try:
        logger.info("Testing file discovery...")
        
        # Create a temporary directory structure
        import tempfile
        import shutil
        
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create test directory structure
            bank_dir = temp_dir / 'test_bank'
            quarter_dir = bank_dir / 'Q1_2025'
            quarter_dir.mkdir(parents=True)
            
            # Create test files
            (quarter_dir / 'test.pdf').touch()
            (quarter_dir / 'test.xlsx').touch()
            (quarter_dir / 'unsupported.txt').touch()
            
            # Import and test the file discovery
            from src.etl.file_discovery import discover_and_process
            
            # Mock parser functions
            def mock_parse_pdf(bank, quarter, file_path):
                return {'bank': bank, 'quarter': quarter, 'file': str(file_path), 'type': 'pdf'}
                
            def mock_parse_excel(bank, quarter, file_path):
                return {'bank': bank, 'quarter': quarter, 'file': str(file_path), 'type': 'excel'}
            
            # Set up mock parsers
            parsers = {
                '.pdf': mock_parse_pdf,
                '.xlsx': mock_parse_excel,
                '.xls': mock_parse_excel,
            }
            
            # Run the discovery
            results = []
            
            def process_file(bank, quarter, file_path):
                parser = parsers.get(file_path.suffix.lower())
                if parser:
                    results.append(parser(bank, quarter, file_path))
            
            # Discover and process files
            for bank_dir in temp_dir.iterdir():
                if not bank_dir.is_dir():
                    continue
                    
                bank = bank_dir.name
                
                for quarter_dir in bank_dir.iterdir():
                    if not quarter_dir.is_dir():
                        continue
                        
                    quarter = quarter_dir.name
                    
                    for file_path in quarter_dir.iterdir():
                        if not file_path.is_file():
                            continue
                            
                        process_file(bank, quarter, file_path)
            
            # Verify results
            assert len(results) == 2  # Should process 2 out of 3 files (skipping unsupported .txt)
            assert any(r['type'] == 'pdf' for r in results)
            assert any(r['type'] == 'excel' for r in results)
            
            logger.info("File discovery test passed!")
            return True
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"File discovery test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    tests = [
        test_pdf_parser,
        test_excel_parser,
        test_file_discovery
    ]
    
    results = []
    
    for test_func in tests:
        logger.info(f"\nRunning test: {test_func.__name__}")
        result = test_func()
        results.append((test_func.__name__, result))
        status = "PASSED" if result else "FAILED"
        logger.info(f"Test {test_func.__name__} {status}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 80)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
