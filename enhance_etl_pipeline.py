#!/usr/bin/env python3
"""
Enhanced ETL Pipeline with Advanced NLP Features

This script enhances the existing ETL pipeline with:
1. Role identification (CEO/CFO/Analyst)
2. Named entity recognition
3. Financial term tagging
4. Enhanced speaker detection
"""

import pandas as pd
import re
import spacy
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

class EnhancedNLPProcessor:
    """Enhanced NLP processor with financial domain expertise."""
    
    def __init__(self):
        """Initialize the enhanced NLP processor."""
        self.financial_terms = self._load_financial_terms()
        self.role_patterns = self._load_role_patterns()
        self.nlp = None
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for NER."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy model for NER")
        except OSError:
            print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _load_financial_terms(self) -> Dict[str, List[str]]:
        """Load financial terms dictionary."""
        return {
            'metrics': [
                'revenue', 'income', 'profit', 'loss', 'earnings', 'ebitda', 'eps',
                'roe', 'roa', 'rotce', 'cet1', 'tier 1', 'capital ratio', 'leverage ratio',
                'net interest margin', 'nim', 'efficiency ratio', 'cost of credit',
                'provision', 'allowance', 'charge-off', 'npa', 'non-performing'
            ],
            'financial_instruments': [
                'loan', 'deposit', 'bond', 'security', 'derivative', 'swap',
                'credit card', 'mortgage', 'commercial loan', 'consumer loan'
            ],
            'business_segments': [
                'investment banking', 'wealth management', 'retail banking',
                'commercial banking', 'markets', 'trading', 'advisory',
                'asset management', 'private banking'
            ],
            'currencies': [
                'usd', 'eur', 'gbp', 'jpy', 'cad', 'aud', 'chf',
                'dollar', 'euro', 'pound', 'yen'
            ],
            'regulatory': [
                'basel', 'dodd-frank', 'mifid', 'ccar', 'stress test',
                'regulatory capital', 'liquidity coverage', 'nsfr'
            ]
        }
    
    def _load_role_patterns(self) -> Dict[str, List[str]]:
        """Load role identification patterns."""
        return {
            'ceo': [
                r'chief executive officer', r'\bceo\b', r'chief exec',
                r'jane fraser', r'fraser'
            ],
            'cfo': [
                r'chief financial officer', r'\bcfo\b', r'chief finance',
                r'mark mason', r'mason'
            ],
            'analyst': [
                r'analyst', r'research', r'equity research', r'bank analyst',
                r'financial analyst', r'sell-side', r'buy-side'
            ],
            'operator': [
                r'operator', r'moderator', r'host', r'conference'
            ],
            'investor_relations': [
                r'investor relations', r'\bir\b', r'jennifer landis', r'landis'
            ]
        }
    
    def identify_speaker_role(self, speaker_name: str, text: str = "") -> str:
        """Identify speaker role based on name and context."""
        if not speaker_name or speaker_name == 'UNKNOWN':
            return 'unknown'
        
        speaker_lower = speaker_name.lower()
        text_lower = text.lower()
        combined_text = f"{speaker_lower} {text_lower}"
        
        # Check each role pattern
        for role, patterns in self.role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return role
        
        # Default classification
        if any(word in speaker_lower for word in ['jane', 'fraser']):
            return 'ceo'
        elif any(word in speaker_lower for word in ['mark', 'mason']):
            return 'cfo'
        elif 'operator' in speaker_lower:
            return 'operator'
        elif any(word in speaker_lower for word in ['landis', 'jennifer']):
            return 'investor_relations'
        else:
            return 'other_speaker'
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        if not self.nlp:
            return {'entities': [], 'types': []}
        
        doc = self.nlp(text)
        entities = []
        entity_types = []
        
        for ent in doc.ents:
            entities.append(ent.text)
            entity_types.append(ent.label_)
        
        # Group by entity type
        entity_dict = {}
        for ent, ent_type in zip(entities, entity_types):
            if ent_type not in entity_dict:
                entity_dict[ent_type] = []
            if ent not in entity_dict[ent_type]:
                entity_dict[ent_type].append(ent)
        
        return entity_dict
    
    def tag_financial_terms(self, text: str) -> Dict[str, List[str]]:
        """Tag financial terms in text."""
        text_lower = text.lower()
        found_terms = {}
        
        for category, terms in self.financial_terms.items():
            found_in_category = []
            for term in terms:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text_lower):
                    found_in_category.append(term)
            
            if found_in_category:
                found_terms[category] = found_in_category
        
        return found_terms
    
    def extract_financial_figures(self, text: str) -> List[Dict[str, str]]:
        """Extract financial figures and amounts."""
        figures = []
        
        # Pattern for monetary amounts
        money_patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|k)?',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|k)?\s*dollars?',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*%'
        ]
        
        for pattern in money_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                figures.append({
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else '',
                    'context': text[max(0, match.start()-20):match.end()+20]
                })
        
        return figures

def enhance_processed_data():
    """Enhance the processed data with advanced NLP features."""
    
    print("🚀 ENHANCING ETL PIPELINE WITH ADVANCED NLP FEATURES")
    print("=" * 70)
    
    # Load existing processed data
    df = pd.read_csv('processed_data_complete.csv')
    print(f"📊 Loaded {len(df)} records for enhancement")
    
    # Initialize enhanced NLP processor
    nlp_processor = EnhancedNLPProcessor()
    
    # Create enhanced columns
    enhanced_df = df.copy()
    
    print("🔍 Adding role identification...")
    enhanced_df['speaker_role'] = enhanced_df.apply(
        lambda row: nlp_processor.identify_speaker_role(
            row['speaker_norm'], 
            row['text']
        ), axis=1
    )
    
    print("🏷️ Extracting named entities...")
    entity_results = []
    for idx, row in enhanced_df.iterrows():
        entities = nlp_processor.extract_named_entities(row['text'])
        entity_results.append(json.dumps(entities) if entities else '')
        
        if idx % 500 == 0:
            print(f"   Processed {idx}/{len(enhanced_df)} records...")
    
    enhanced_df['named_entities_enhanced'] = entity_results
    
    print("💰 Tagging financial terms...")
    financial_tags = []
    financial_figures = []
    
    for idx, row in enhanced_df.iterrows():
        # Tag financial terms
        fin_terms = nlp_processor.tag_financial_terms(row['text'])
        financial_tags.append(json.dumps(fin_terms) if fin_terms else '')
        
        # Extract financial figures
        figures = nlp_processor.extract_financial_figures(row['text'])
        financial_figures.append(json.dumps(figures) if figures else '')
        
        if idx % 500 == 0:
            print(f"   Processed {idx}/{len(enhanced_df)} records...")
    
    enhanced_df['financial_terms'] = financial_tags
    enhanced_df['financial_figures'] = financial_figures
    
    # Create enhanced speaker categories
    def enhanced_speaker_category(role, is_analyst):
        if role == 'ceo':
            return 'CEO'
        elif role == 'cfo':
            return 'CFO'
        elif role == 'analyst' or is_analyst:
            return 'Analyst'
        elif role == 'operator':
            return 'Operator'
        elif role == 'investor_relations':
            return 'Investor Relations'
        elif role == 'unknown':
            return 'Document Text'
        else:
            return 'Other Speaker'
    
    enhanced_df['speaker_category_enhanced'] = enhanced_df.apply(
        lambda row: enhanced_speaker_category(row['speaker_role'], row['analyst_utterance']),
        axis=1
    )
    
    # Add enhancement metadata
    enhanced_df['enhancement_date'] = datetime.now().isoformat()
    enhanced_df['enhancement_version'] = 'v1.0_enhanced'
    
    # Save enhanced data
    enhanced_file = 'processed_data_enhanced.csv'
    enhanced_df.to_csv(enhanced_file, index=False, encoding='utf-8-sig')
    
    # Create enhanced validation export
    print("📋 Creating enhanced validation export...")
    
    # Select key columns for validation
    validation_columns = [
        'Review_ID', 'source_file', 'quarter_period', 'sentence_id',
        'speaker_norm', 'speaker_role', 'speaker_category_enhanced',
        'analyst_utterance', 'text', 'word_count',
        'named_entities_enhanced', 'financial_terms', 'financial_figures',
        'processing_date', 'enhancement_date'
    ]
    
    # Add Review_ID if not present
    if 'Review_ID' not in enhanced_df.columns:
        enhanced_df.insert(0, 'Review_ID', range(1, len(enhanced_df) + 1))
    
    # Create validation subset
    validation_df = enhanced_df[validation_columns].copy()
    
    # Create text preview for readability
    validation_df['text_preview'] = validation_df['text'].str[:150] + '...'
    
    # Save enhanced validation export
    enhanced_validation_file = 'enhanced_validation_export.csv'
    validation_df.to_csv(enhanced_validation_file, index=False, encoding='utf-8-sig')
    
    # Generate enhancement summary
    print("📊 Generating enhancement summary...")
    
    role_distribution = enhanced_df['speaker_role'].value_counts()
    category_distribution = enhanced_df['speaker_category_enhanced'].value_counts()
    
    # Count records with financial content
    records_with_entities = len(enhanced_df[enhanced_df['named_entities_enhanced'] != ''])
    records_with_financial_terms = len(enhanced_df[enhanced_df['financial_terms'] != ''])
    records_with_figures = len(enhanced_df[enhanced_df['financial_figures'] != ''])
    
    # Create summary report
    summary_file = 'enhancement_summary_report.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ENHANCED ETL PIPELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ENHANCEMENT FEATURES ADDED:\n")
        f.write("-" * 30 + "\n")
        f.write("✅ Role identification (CEO/CFO/Analyst)\n")
        f.write("✅ Named entity recognition\n")
        f.write("✅ Financial term tagging\n")
        f.write("✅ Financial figure extraction\n")
        f.write("✅ Enhanced speaker categorization\n\n")
        
        f.write("ROLE IDENTIFICATION RESULTS:\n")
        f.write("-" * 30 + "\n")
        for role, count in role_distribution.items():
            percentage = (count / len(enhanced_df)) * 100
            f.write(f"{role}: {count} records ({percentage:.1f}%)\n")
        
        f.write(f"\nENHANCED SPEAKER CATEGORIES:\n")
        f.write("-" * 30 + "\n")
        for category, count in category_distribution.items():
            percentage = (count / len(enhanced_df)) * 100
            f.write(f"{category}: {count} records ({percentage:.1f}%)\n")
        
        f.write(f"\nNLP ENHANCEMENT COVERAGE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Records with Named Entities: {records_with_entities} ({(records_with_entities/len(enhanced_df)*100):.1f}%)\n")
        f.write(f"Records with Financial Terms: {records_with_financial_terms} ({(records_with_financial_terms/len(enhanced_df)*100):.1f}%)\n")
        f.write(f"Records with Financial Figures: {records_with_figures} ({(records_with_figures/len(enhanced_df)*100):.1f}%)\n")
        
        f.write(f"\nFILES CREATED:\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. {enhanced_file} - Complete enhanced dataset\n")
        f.write(f"2. {enhanced_validation_file} - Enhanced validation export\n")
        f.write(f"3. {summary_file} - This summary report\n")
    
    # Print completion summary
    print(f"\n✅ ENHANCEMENT COMPLETE")
    print("=" * 70)
    print(f"📄 Enhanced dataset: {enhanced_file}")
    print(f"   • {len(enhanced_df):,} records enhanced")
    print(f"   • Role identification added")
    print(f"   • Named entity recognition added")
    print(f"   • Financial term tagging added")
    
    print(f"\n📋 Enhanced validation: {enhanced_validation_file}")
    print(f"   • Optimized for validation workflow")
    print(f"   • Includes all enhancement features")
    
    print(f"\n📊 Summary report: {summary_file}")
    print(f"   • Detailed enhancement analysis")
    print(f"   • Feature coverage statistics")
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"   • CEO identified: {role_distribution.get('ceo', 0)} records")
    print(f"   • CFO identified: {role_distribution.get('cfo', 0)} records")
    print(f"   • Analysts identified: {role_distribution.get('analyst', 0)} records")
    print(f"   • Financial content tagged: {records_with_financial_terms} records")
    print(f"   • Named entities extracted: {records_with_entities} records")
    
    return enhanced_df

if __name__ == "__main__":
    enhanced_data = enhance_processed_data()
    print(f"\n🎉 ETL Pipeline enhancement complete!")