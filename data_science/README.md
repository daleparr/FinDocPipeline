# Data Science Components for Financial Risk Monitoring

This directory contains advanced data science components for financial risk monitoring and analysis, built on top of the existing ETL pipeline.

## Overview

The data science components provide sophisticated NLP analysis capabilities for financial documents, focusing on:

- **Advanced Sentiment Analysis** using FinBERT
- **Hybrid Topic Modeling** combining seed-based and emergent discovery
- **Risk-specific Feature Extraction** for financial monitoring
- **Statistical Analysis** for trend detection and anomaly identification

## Directory Structure

```
data_science/
├── README.md                           # This file
├── config/                             # Configuration files
│   ├── risk_monitoring_config.yaml     # Main configuration
│   └── seed_topics.yaml               # Seed topics for hybrid modeling
├── scripts/                           # Analysis scripts
│   ├── sentiment_analysis/            # Sentiment analysis components
│   │   └── finbert_analyzer.py        # FinBERT-based sentiment analyzer
│   ├── topic_modeling/               # Topic modeling components
│   │   └── hybrid_topic_engine.py    # Hybrid topic modeling engine
│   └── test_phase2_nlp.py            # Comprehensive test suite
└── outputs/                          # Analysis outputs (created automatically)
```

## Phase 2: Advanced NLP Components

### 1. FinBERT Sentiment Analyzer

**File**: [`scripts/sentiment_analysis/finbert_analyzer.py`](scripts/sentiment_analysis/finbert_analyzer.py)

Advanced sentiment analysis specifically designed for financial text using the ProsusAI/finbert model.

**Key Features**:
- Financial domain-specific sentiment classification
- Risk escalation language detection
- Hedging and uncertainty scoring
- Tone analysis (formality, complexity)
- Speaker-weighted aggregation
- Confidence scoring and validation

**Usage**:
```python
from sentiment_analysis.finbert_analyzer import FinBERTAnalyzer

analyzer = FinBERTAnalyzer()
result_df = analyzer.analyze_dataframe(df, text_column='text')
```

**Output Fields**:
- `sentiment_label`: Primary sentiment (positive/negative/neutral)
- `sentiment_confidence`: Confidence score for classification
- `sentiment_positive/negative/neutral`: Individual class probabilities
- `hedging_score`: Degree of hedging language
- `uncertainty_score`: Level of uncertainty markers
- `formality_score`: Formality of language
- `complexity_score`: Text complexity measure
- `risk_escalation_score`: Risk escalation language indicator
- `stress_score`: Financial stress indicators
- `confidence_score`: Management confidence indicators

### 2. Hybrid Topic Modeling Engine

**File**: [`scripts/topic_modeling/hybrid_topic_engine.py`](scripts/topic_modeling/hybrid_topic_engine.py)

Combines seed-based topic classification with BERTopic emergent discovery for comprehensive topic analysis.

**Key Features**:
- Seed topic assignment based on financial risk categories
- Emergent topic discovery using BERTopic
- Coherence-based topic validation
- Configurable thresholds and parameters
- Processing metadata and statistics

**Usage**:
```python
from topic_modeling.hybrid_topic_engine import HybridTopicEngine

engine = HybridTopicEngine()
result_df, metadata = engine.process_quarter_data('BankName', 'Q1_2025', sentences_df)
```

**Output Fields**:
- `topic_seed`: Assigned seed topic (if any)
- `topic_emergent`: Discovered emergent topic (if any)
- `final_topic`: Final topic assignment
- `topic_confidence`: Confidence in topic assignment
- `topic_type`: Type of assignment (seed/emergent/misc)
- `topic_keywords`: Keywords associated with topic

**Seed Topics**:
- Credit Risk
- Operational Risk
- Market Risk
- Regulatory Risk
- Liquidity Risk
- Capital Management
- Strategic Risk
- ESG & Sustainability

## Configuration

### Main Configuration (`config/risk_monitoring_config.yaml`)

```yaml
sentiment_analysis:
  model_name: "ProsusAI/finbert"
  batch_size: 32
  confidence_threshold: 0.7
  use_gpu: true

topic_modeling:
  seed_threshold: 3
  emergent_min_cluster_size: 50
  coherence_threshold: 0.4
  max_topics: 50
  embedding_model: "all-MiniLM-L6-v2"

speaker_weights:
  CEO: 1.0
  CFO: 0.9
  CRO: 0.8
  CCO: 0.7
  CTO: 0.6
  Analyst: 0.3
```

### Seed Topics Configuration (`config/seed_topics.yaml`)

Defines financial risk categories with associated keywords, weights, and confidence thresholds.

## Testing

### Comprehensive Test Suite

**File**: [`scripts/test_phase2_nlp.py`](scripts/test_phase2_nlp.py)

Validates all Phase 2 components with comprehensive test scenarios.

**Run Tests**:
```bash
cd data_science
python scripts/test_phase2_nlp.py
```

**Test Coverage**:
- Sentiment analysis accuracy and performance
- Topic modeling seed and emergent discovery
- Integration between components
- Error handling and edge cases
- Performance benchmarking

## Integration with ETL Pipeline

The data science components integrate seamlessly with the existing ETL pipeline:

1. **Input**: Processed sentence-level data from ETL pipeline
2. **Processing**: Advanced NLP analysis using Phase 2 components
3. **Output**: Enhanced data with sentiment and topic features
4. **Storage**: Results stored in versioned data structure

### Example Integration

```python
# Load processed data from ETL
from src.etl.data_versioning import DataVersionManager
version_manager = DataVersionManager()
sentences_df = version_manager.load_version('BankName', 'Q1_2025', 'latest')

# Apply sentiment analysis
from sentiment_analysis.finbert_analyzer import FinBERTAnalyzer
analyzer = FinBERTAnalyzer()
enhanced_df = analyzer.analyze_dataframe(sentences_df)

# Apply topic modeling
from topic_modeling.hybrid_topic_engine import HybridTopicEngine
engine = HybridTopicEngine()
final_df, metadata = engine.process_quarter_data('BankName', 'Q1_2025', enhanced_df)

# Save enhanced results
version_manager.save_enhanced_version(final_df, metadata)
```

## Performance Considerations

### Hardware Requirements

- **CPU**: Multi-core processor recommended for parallel processing
- **Memory**: 8GB+ RAM for large datasets
- **GPU**: Optional but recommended for FinBERT acceleration
- **Storage**: SSD recommended for model loading performance

### Optimization Tips

1. **Batch Processing**: Use appropriate batch sizes for your hardware
2. **GPU Acceleration**: Enable GPU for FinBERT if available
3. **Caching**: Models are cached after first load
4. **Parallel Processing**: Components support parallel execution
5. **Memory Management**: Large datasets processed in chunks

## Monitoring and Logging

All components include comprehensive logging:

- **INFO**: Processing progress and statistics
- **WARNING**: Performance issues and fallbacks
- **ERROR**: Processing failures with detailed traces

Log levels can be configured in the main configuration file.

## Future Enhancements (Phase 3)

Planned statistical analysis components:

1. **Time Series Analysis**: Trend detection and forecasting
2. **Anomaly Detection**: Statistical outlier identification
3. **Risk Scoring**: Composite risk score calculation
4. **Comparative Analysis**: Cross-institution benchmarking
5. **Predictive Modeling**: Risk prediction models

## Dependencies

Key Python packages required:

```
transformers>=4.20.0
torch>=1.12.0
sentence-transformers>=2.2.0
bertopic>=0.15.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
pyyaml>=6.0
```

## Support and Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure proper PyTorch installation for your system
2. **Memory Errors**: Reduce batch sizes or process data in smaller chunks
3. **Model Download**: First run requires internet connection for model downloads
4. **Unicode Errors**: Ensure proper text encoding in input data

### Performance Tuning

1. **Batch Size**: Adjust based on available memory
2. **Model Selection**: Consider lighter models for faster processing
3. **Threshold Tuning**: Adjust confidence thresholds based on your data
4. **Parallel Processing**: Utilize multiple cores for large datasets

## Contributing

When adding new components:

1. Follow the established directory structure
2. Include comprehensive docstrings and type hints
3. Add corresponding test cases
4. Update configuration files as needed
5. Document new features in this README

## License

This project is part of the Financial ETL Pipeline and follows the same licensing terms.