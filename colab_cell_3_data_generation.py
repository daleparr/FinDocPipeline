# CELL 3: VALIDATION DATA GENERATION
# ==================================
# Copy this entire cell into the third cell of your Google Colab notebook

def generate_validation_dataset():
    """Generate comprehensive validation dataset with known ground truth."""
    print("Generating validation dataset...")
    
    # Financial documents with known topics and sentiments
    documents = [
        # Regulatory compliance (positive sentiment)
        "Our capital ratios remain strong with CET1 at 12.5%, well above regulatory requirements. We continue to meet all Basel III standards.",
        "Regulatory compliance remains a top priority. We have enhanced our risk management framework and strengthened our capital position.",
        "The bank maintains robust capital buffers and liquidity positions. Our stress testing results demonstrate resilience under adverse scenarios.",
        
        # Financial performance (positive sentiment)
        "Revenue growth of 8% year-over-year driven by strong net interest margin expansion. Operating efficiency improved significantly.",
        "Solid financial performance with ROE of 12.3% and strong profit margins. Revenue diversification continues to support sustainable growth.",
        "Excellent quarterly results with earnings per share up 15%. Our diversified business model continues to deliver consistent returns.",
        
        # Credit risk (mixed sentiment)
        "Credit quality remains stable with NPL ratio at 1.8%. Provision coverage is adequate and we maintain conservative underwriting standards.",
        "Our loan portfolio shows strong performance with low default rates. Credit provisions have been increased prudently given economic uncertainty.",
        "Challenging economic conditions have led to higher provisions, but our diversified portfolio and strong risk management provide resilience.",
        
        # Operational risk (mixed sentiment)
        "Operational efficiency initiatives delivered cost savings of $50M. Technology investments continue to enhance our digital capabilities.",
        "Cyber security remains a priority with enhanced monitoring and threat detection capabilities. Operational risk framework has been strengthened.",
        "Technology disruptions caused temporary service interruptions but our business continuity plans ensured minimal customer impact.",
        
        # Market risk (negative sentiment)
        "Trading revenue was impacted by market volatility but risk management kept losses within acceptable limits. VaR models performed well.",
        "Volatile market conditions resulted in trading losses, but our risk management framework prevented significant exposure to tail risks.",
        "Market turbulence created challenging conditions for our trading business, with revenues declining due to reduced client activity."
    ]
    
    # Ground truth labels
    topic_labels = [
        'regulatory_compliance', 'regulatory_compliance', 'regulatory_compliance',
        'financial_performance', 'financial_performance', 'financial_performance',
        'credit_risk', 'credit_risk', 'credit_risk',
        'operational_risk', 'operational_risk', 'operational_risk',
        'market_risk', 'market_risk', 'market_risk'
    ]
    
    sentiment_labels = [
        'positive', 'positive', 'positive',
        'positive', 'positive', 'positive',
        'positive', 'positive', 'negative',
        'positive', 'positive', 'negative',
        'neutral', 'negative', 'negative'
    ]
    
    print(f"Generated {len(documents)} documents")
    print(f"Topics: {set(topic_labels)}")
    print(f"Sentiment distribution: {Counter(sentiment_labels)}")
    
    return documents, topic_labels, sentiment_labels

# Generate the validation dataset
documents, topic_labels, sentiment_labels = generate_validation_dataset()

# Display sample documents
print("\nSample Documents:")
for i, (doc, topic, sentiment) in enumerate(zip(documents[:3], topic_labels[:3], sentiment_labels[:3])):
    print(f"\nDocument {i+1} ({topic}, {sentiment}):")
    print(f"  {doc[:100]}...")