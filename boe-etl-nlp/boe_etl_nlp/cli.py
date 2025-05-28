#!/usr/bin/env python3
"""
Command Line Interface for BoE ETL NLP Extension
===============================================

This module provides a command-line interface for the BoE ETL NLP extension package.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from . import NLPProcessor, TopicModeler, SentimentAnalyzer, FinancialClassifier
from . import __version__


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def process_file(
    input_file: Path,
    output_file: Optional[Path] = None,
    config: Optional[dict] = None
) -> None:
    """
    Process a single file with NLP features.
    
    Args:
        input_file: Path to input CSV/Parquet file
        output_file: Path to output file (optional)
        config: Configuration dictionary (optional)
    """
    logger = logging.getLogger(__name__)
    
    # Read input file
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    elif input_file.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    logger.info(f"Loaded {len(df)} records from {input_file}")
    
    # Process with NLP features
    processor = NLPProcessor(config)
    enhanced_df = processor.add_nlp_features(df)
    
    logger.info(f"Added NLP features to {len(enhanced_df)} records")
    
    # Save output
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_nlp_enhanced{input_file.suffix}"
    
    if output_file.suffix.lower() == '.csv':
        enhanced_df.to_csv(output_file, index=False)
    elif output_file.suffix.lower() == '.parquet':
        enhanced_df.to_parquet(output_file, index=False)
    else:
        # Default to CSV
        output_file = output_file.with_suffix('.csv')
        enhanced_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved enhanced data to {output_file}")


def analyze_topics(
    input_file: Path,
    bank_name: str,
    quarter: str,
    output_file: Optional[Path] = None,
    config: Optional[dict] = None
) -> None:
    """
    Analyze topics in the input data.
    
    Args:
        input_file: Path to input file with text data
        bank_name: Name of the bank
        quarter: Quarter identifier
        output_file: Path to output file (optional)
        config: Configuration dictionary (optional)
    """
    logger = logging.getLogger(__name__)
    
    # Read input file
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    elif input_file.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    # Convert to records format
    records = df.to_dict('records')
    
    # Analyze topics
    modeler = TopicModeler(config)
    topic_results = modeler.process_batch(records, bank_name, quarter)
    
    # Convert back to DataFrame
    results_df = pd.DataFrame(topic_results)
    
    # Save output
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_topics{input_file.suffix}"
    
    if output_file.suffix.lower() == '.csv':
        results_df.to_csv(output_file, index=False)
    elif output_file.suffix.lower() == '.parquet':
        results_df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved topic analysis results to {output_file}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BoE ETL NLP Extension CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add NLP features to a CSV file
  boe-etl-nlp process data.csv --output enhanced_data.csv
  
  # Analyze topics in financial data
  boe-etl-nlp topics data.csv --bank "JPMorgan" --quarter "Q1_2025"
  
  # Process with verbose logging
  boe-etl-nlp process data.csv --verbose
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'boe-etl-nlp {__version__}'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Add NLP features to financial data'
    )
    process_parser.add_argument(
        'input_file',
        type=Path,
        help='Input CSV or Parquet file'
    )
    process_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (optional)'
    )
    
    # Topics command
    topics_parser = subparsers.add_parser(
        'topics',
        help='Analyze topics in financial data'
    )
    topics_parser.add_argument(
        'input_file',
        type=Path,
        help='Input CSV or Parquet file'
    )
    topics_parser.add_argument(
        '--bank', '-b',
        required=True,
        help='Bank name'
    )
    topics_parser.add_argument(
        '--quarter', '-q',
        required=True,
        help='Quarter identifier (e.g., Q1_2025)'
    )
    topics_parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'process':
            process_file(args.input_file, args.output)
        elif args.command == 'topics':
            analyze_topics(args.input_file, args.bank, args.quarter, args.output)
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()