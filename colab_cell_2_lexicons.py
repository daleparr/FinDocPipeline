# CELL 2: FINANCIAL LEXICONS AND DICTIONARIES
# ===========================================
# Copy this entire cell into the second cell of your Google Colab notebook

# Financial domain lexicon for sentiment analysis
financial_sentiment_lexicon = {
    # Positive terms
    'strong': 0.6, 'robust': 0.6, 'solid': 0.4, 'excellent': 0.8,
    'outstanding': 0.7, 'growth': 0.5, 'improved': 0.5, 'stable': 0.3,
    'resilient': 0.5, 'conservative': 0.2, 'prudent': 0.3, 'healthy': 0.4,
    'enhanced': 0.4, 'efficient': 0.4, 'profitable': 0.6, 'successful': 0.6,
    
    # Negative terms
    'challenging': -0.5, 'volatile': -0.4, 'weak': -0.5, 'deteriorating': -0.7,
    'impacted': -0.3, 'disruptions': -0.5, 'turbulence': -0.6, 'losses': -0.6,
    'declined': -0.5, 'decreased': -0.4, 'pressure': -0.3, 'uncertainty': -0.4,
    'risk': -0.2, 'concern': -0.4, 'difficult': -0.5, 'adverse': -0.6,
    
    # Neutral/context-dependent terms
    'maintained': 0.1, 'continued': 0.1, 'remains': 0.0, 'compliance': 0.2,
    'regulatory': 0.0, 'capital': 0.1, 'provisions': -0.1, 'monitoring': 0.1
}

# Topic keywords for classification
topic_keywords = {
    'regulatory_compliance': [
        'regulatory', 'compliance', 'capital', 'basel', 'stress', 'test',
        'requirements', 'standards', 'supervisory', 'authorities', 'cet1',
        'buffers', 'framework', 'guidelines', 'ratios'
    ],
    'financial_performance': [
        'revenue', 'profit', 'earnings', 'growth', 'margin', 'performance',
        'roe', 'returns', 'profitability', 'efficiency', 'cost', 'income',
        'diversification', 'business', 'results', 'quarterly'
    ],
    'credit_risk': [
        'credit', 'loan', 'npl', 'provision', 'default', 'underwriting',
        'portfolio', 'asset', 'quality', 'impairment', 'coverage',
        'exposures', 'losses', 'disciplined', 'conservative'
    ],
    'operational_risk': [
        'operational', 'technology', 'cyber', 'security', 'process',
        'automation', 'digital', 'efficiency', 'resilience', 'disruptions',
        'systems', 'capabilities', 'monitoring', 'infrastructure'
    ],
    'market_risk': [
        'market', 'trading', 'volatility', 'var', 'hedging', 'currency',
        'interest', 'rate', 'foreign', 'exchange', 'risk', 'management',
        'exposure', 'limits', 'appetite', 'conditions'
    ]
}

# Common stop words
stop_words = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
    'a', 'an', 'we', 'our', 'us', 'bank', 'banking', 'financial',
    'institution', 'company', 'business', 'quarter', 'year'
}

print("Financial lexicons and dictionaries loaded successfully")
print(f"- Sentiment lexicon: {len(financial_sentiment_lexicon)} terms")
print(f"- Topic keywords: {len(topic_keywords)} categories")
print(f"- Stop words: {len(stop_words)} words")