# CELL 5: SENTIMENT ANALYSIS FUNCTIONS
# ====================================
# Copy this entire cell into the fifth cell of your Google Colab notebook

def analyze_sentiment(text):
    """Analyze sentiment using financial domain lexicon."""
    words = preprocess_text(text)
    
    sentiment_scores = []
    matched_words = []
    
    for word in words:
        if word in financial_sentiment_lexicon:
            score = financial_sentiment_lexicon[word]
            sentiment_scores.append(score)
            matched_words.append((word, score))
    
    if sentiment_scores:
        # Calculate compound sentiment score
        compound_score = sum(sentiment_scores) / len(sentiment_scores)
        
        # Apply financial context weighting
        financial_keywords = ['capital', 'risk', 'regulatory', 'compliance', 'credit']
        financial_weight = sum(1 for keyword in financial_keywords if keyword in text.lower())
        
        if financial_weight > 0:
            # Amplify sentiment in financial context
            compound_score *= (1 + financial_weight * 0.1)
            compound_score = max(-1, min(1, compound_score))  # Clip to [-1, 1]
    else:
        compound_score = 0.0
    
    # Classify sentiment
    if compound_score >= 0.05:
        classification = 'positive'
    elif compound_score <= -0.05:
        classification = 'negative'
    else:
        classification = 'neutral'
    
    return {
        'compound': compound_score,
        'classification': classification,
        'matched_words': matched_words,
        'word_count': len(matched_words)
    }

# Test sentiment analysis
print("Testing sentiment analysis:")
print("=" * 40)

for i, doc in enumerate(documents[:5]):
    sentiment = analyze_sentiment(doc)
    print(f"\nDocument {i+1}:")
    print(f"  Text: {doc[:60]}...")
    print(f"  Sentiment: {sentiment['classification']} (score: {sentiment['compound']:.3f})")
    print(f"  Matched words: {sentiment['matched_words'][:3]}...")  # Show first 3 matches