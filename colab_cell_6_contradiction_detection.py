# CELL 6: CONTRADICTION DETECTION
# ===============================
# Copy this entire cell into the sixth cell of your Google Colab notebook

def detect_contradictions(text):
    """Detect contradictions between positive language and negative indicators."""
    sentences = re.split(r'[.!?]+', text)
    
    if len(sentences) < 2:
        return {'has_contradiction': False, 'confidence': 0.0, 'details': 'Insufficient sentences'}
    
    sentence_sentiments = []
    
    for sentence in sentences:
        if sentence.strip():
            sentiment = analyze_sentiment(sentence)
            sentence_sentiments.append(sentiment['compound'])
    
    if len(sentence_sentiments) >= 2:
        # Check for opposing sentiments
        max_sentiment = max(sentence_sentiments)
        min_sentiment = min(sentence_sentiments)
        
        # Detect contradiction (strong positive and strong negative)
        sentiment_diff = max_sentiment - min_sentiment
        has_contradiction = (max_sentiment > 0.3 and min_sentiment < -0.3) or sentiment_diff > 0.6
        
        confidence = min(sentiment_diff, 1.0)
        
        return {
            'has_contradiction': has_contradiction,
            'confidence': confidence,
            'sentiment_range': (min_sentiment, max_sentiment),
            'sentence_count': len(sentence_sentiments)
        }
    
    return {'has_contradiction': False, 'confidence': 0.0, 'details': 'Insufficient analysis'}

# Test contradiction detection with known examples
print("Testing contradiction detection:")
print("=" * 50)

test_cases = [
    {
        'text': "We maintain strong credit quality and robust risk management. However, NPL ratios have increased significantly to 8.5% this quarter.",
        'expected': True
    },
    {
        'text': "Excellent operational performance with enhanced efficiency. Technology disruptions caused major service outages affecting customers.",
        'expected': True
    },
    {
        'text': "Capital ratios remain strong at 12.5% CET1. We continue to meet all regulatory requirements comfortably.",
        'expected': False
    }
]

for i, case in enumerate(test_cases):
    result = detect_contradictions(case['text'])
    print(f"\nTest Case {i+1}:")
    print(f"  Text: {case['text'][:80]}...")
    print(f"  Expected contradiction: {case['expected']}")
    print(f"  Detected contradiction: {result['has_contradiction']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Correct: {result['has_contradiction'] == case['expected']}")