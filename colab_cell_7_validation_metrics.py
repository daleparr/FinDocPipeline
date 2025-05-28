# CELL 7: VALIDATION AND METRICS CALCULATION
# ==========================================
# Copy this entire cell into the seventh cell of your Google Colab notebook

def validate_topic_classification(documents, true_labels):
    """Validate topic classification accuracy."""
    print("Validating Topic Classification...")
    start_time = time.time()
    
    predictions = []
    confidences = []
    
    for doc in documents:
        topic, confidence, scores = classify_topic(doc)
        predictions.append(topic)
        confidences.append(confidence)
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    accuracy = correct / len(true_labels)
    
    # Calculate per-topic performance
    topic_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for true, pred in zip(true_labels, predictions):
        topic_performance[true]['total'] += 1
        if true == pred:
            topic_performance[true]['correct'] += 1
    
    processing_time = time.time() - start_time
    
    print(f"Topic Classification Results:")
    print(f"  Overall Accuracy: {accuracy:.3f} ({correct}/{len(true_labels)})")
    print(f"  Average Confidence: {sum(confidences)/len(confidences):.3f}")
    print(f"  Processing Time: {processing_time:.3f} seconds")
    
    print(f"Per-Topic Performance:")
    for topic, perf in topic_performance.items():
        topic_acc = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        print(f"    {topic}: {topic_acc:.3f} ({perf['correct']}/{perf['total']})")
    
    threshold_status = "PASS" if accuracy >= 0.70 else "FAIL"
    print(f"Status (>=70%): {threshold_status}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'confidences': confidences,
        'topic_performance': dict(topic_performance),
        'processing_time': processing_time
    }

def validate_sentiment_analysis(documents, true_labels):
    """Validate sentiment analysis accuracy."""
    print("\nValidating Sentiment Analysis...")
    start_time = time.time()
    
    predictions = []
    scores = []
    
    for doc in documents:
        sentiment = analyze_sentiment(doc)
        predictions.append(sentiment['classification'])
        scores.append(sentiment['compound'])
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    accuracy = correct / len(true_labels)
    
    # Calculate sentiment distribution
    pred_distribution = Counter(predictions)
    true_distribution = Counter(true_labels)
    
    processing_time = time.time() - start_time
    
    print(f"Sentiment Analysis Results:")
    print(f"  Overall Accuracy: {accuracy:.3f} ({correct}/{len(true_labels)})")
    print(f"  Average Score Magnitude: {sum(abs(s) for s in scores)/len(scores):.3f}")
    print(f"  Processing Time: {processing_time:.3f} seconds")
    
    print(f"Prediction Distribution: {pred_distribution}")
    print(f"True Distribution: {true_distribution}")
    
    threshold_status = "PASS" if accuracy >= 0.80 else "FAIL"
    print(f"Status (>=80%): {threshold_status}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'scores': scores,
        'pred_distribution': dict(pred_distribution),
        'true_distribution': dict(true_distribution),
        'processing_time': processing_time
    }

# Run the validations
topic_results = validate_topic_classification(documents, topic_labels)
sentiment_results = validate_sentiment_analysis(documents, sentiment_labels)