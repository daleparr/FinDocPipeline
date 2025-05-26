"""Tests for the Excel parser."""
import unittest
import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.etl.parsers.excel_parser import ExcelParser
except ImportError as e:
    print(f"Error importing ExcelParser: {e}")
    raise

class TestExcelParser(unittest.TestCase):
    """Test cases for the Excel parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.bank = "test_bank"
        self.quarter = "Q1_2025"
        self.parser = ExcelParser(self.bank, self.quarter)
        
        # Create a sample Excel file for testing
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        self.test_file = Path(self.temp_file.name)
        
        # Create a sample Excel file with two sheets
        with pd.ExcelWriter(self.test_file, engine='openpyxl') as writer:
            # Sheet 1: Simple data
            df1 = pd.DataFrame({
                'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
                'Value': [100, 200, 150],
                'Category': ['A', 'B', 'A']
            })
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            
            # Sheet 2: Different structure
            df2 = pd.DataFrame({
                'Metric': ['Revenue', 'Expenses', 'Profit'],
                'Q1_2025': [1000, 700, 300],
                'Q4_2024': [950, 650, 300],
                'Q3_2024': [900, 600, 300]
            })
            df2.to_excel(writer, sheet_name='Financials', index=False)
    
    def tearDown(self):
        """Clean up after tests."""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_parse_basic_structure(self):
        """Test basic parsing of an Excel file."""
        result = self.parser.parse(self.test_file)
        
        # Check top-level structure
        self.assertEqual(result['bank'], self.bank)
        self.assertEqual(result['quarter'], self.quarter)
        self.assertEqual(result['file_path'], str(self.test_file))
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        
        # Check content structure
        content = result['content']
        self.assertIn('tables', content)
        self.assertIn('metadata', content)
        
        # Check sheet names in metadata
        self.assertIn('sheet_names', content['metadata'])
        self.assertEqual(set(content['metadata']['sheet_names']), {'Sheet1', 'Financials'})
    
    def test_sheet_parsing(self):
        """Test parsing of individual sheets."""
        result = self.parser.parse(self.test_file)
        tables = result['content']['tables']
        
        # Check Sheet1
        self.assertIn('Sheet1', tables)
        sheet1 = tables['Sheet1']
        self.assertEqual(sheet1['shape']['rows'], 3)
        self.assertEqual(sheet1['shape']['columns'], 3)
        self.assertEqual(len(sheet1['columns']), 3)
        self.assertEqual(len(sheet1['data']), 3)
        
        # Check Financials sheet
        self.assertIn('Financials', tables)
        financials = tables['Financials']
        self.assertEqual(financials['shape']['rows'], 3)
        self.assertEqual(financials['shape']['columns'], 4)
        self.assertEqual(len(financials['columns']), 4)
        self.assertEqual(len(financials['data']), 3)
    
    def test_metadata_extraction(self):
        """Test extraction of metadata."""
        result = self.parser.parse(self.test_file)
        metadata = result['content']['metadata']
        
        # Check basic metadata
        self.assertIn('sheet_names', metadata)
        self.assertIsInstance(metadata['sheet_names'], list)
        self.assertEqual(len(metadata['sheet_names']), 2)
        
        # Check table metadata
        tables = result['content']['tables']
        for sheet_name, table in tables.items():
            self.assertIn('metadata', table)
            self.assertIsInstance(table['metadata'], dict)
    
    def test_document_type_inference(self):
        """Test document type inference from filename."""
        # Test with results in filename
        result_path = Path("test_results_Q1_2025.xlsx")
        doc_type = self.parser._infer_document_type(result_path.name)
        self.assertEqual(doc_type, 'results')
        
        # Test with supplement in filename
        supp_path = Path("Q1_2025_supplement.xlsx")
        doc_type = self.parser._infer_document_type(supp_path.name)
        self.assertEqual(doc_type, 'supplement')
        
        # Test with financial in filename
        fin_path = Path("financial_statements_Q1_2025.xlsx")
        doc_type = self.parser._infer_document_type(fin_path.name)
        self.assertEqual(doc_type, 'financial_statements')
        
        # Test with unknown type
        other_path = Path("other_document.xlsx")
        doc_type = self.parser._infer_document_type(other_path.name)
        self.assertEqual(doc_type, 'other')


if __name__ == '__main__':
    unittest.main()
