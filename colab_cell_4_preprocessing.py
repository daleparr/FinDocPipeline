# CELL 4: TEXT PREPROCESSING FUNCTIONS
# ====================================
# Copy this entire cell into the fourth cell of your Google Colab notebook

def preprocess_text(text):
    """Preprocess text using basic Python string operations."""
    # Convert to lowercase
    text = text.lower()
    
    # Extract words (alphanumeric only)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    
    return words

def classify_topic(text):
    """Classify document topic using keyword matching."""
    words = preprocess_text(text)
    word_set = set(words)
    
    topic_scores = {}
    
    for topic, keywords in topic_keywords.items():
        # Calculate overlap between document words and topic keywords
        overlap = len(word_set.intersection(set(keywords)))
        # Normalize by topic keyword count
        score = overlap / len(keywords) if keywords else 0
        topic_scores[topic] = score
    
    # Return topic with highest score
    if topic_scores:
        best_topic = max(topic_scores, key=topic_scores.get)
        confidence = topic_scores[best_topic]
        return best_topic, confidence, topic_scores
    else:
        return 'unknown', 0.0, {}

# Test the preprocessing and topic classification
print("Testing text preprocessing and topic classification:")
print("=" * 60)

test_doc = documents[0]
print(f"Original text: {test_doc}")
print(f"Preprocessed: {preprocess_text(test_doc)}")

topic, confidence, scores = classify_topic(test_doc)
print(f"Classified topic: {topic} (confidence: {confidence:.3f})")
print(f"All topic scores: {scores}")