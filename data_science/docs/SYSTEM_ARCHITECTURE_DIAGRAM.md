# Risk Monitoring System Architecture Diagrams

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        DS1[Earnings Call Transcripts]
        DS2[Regulatory Filings]
        DS3[Press Releases]
        DS4[Analyst Reports]
    end
    
    subgraph "Data Ingestion Layer"
        ETL[ETL Pipeline]
        PARSE[Multi-Format Parser]
        CLEAN[Data Cleaning]
        VALID[Data Validation]
    end
    
    subgraph "Data Storage"
        RAW[(Raw Data Store)]
        PROC[(Processed Data Store)]
        META[(Metadata Store)]
        MODEL[(Model Registry)]
    end
    
    subgraph "NLP Processing Engine"
        TOPIC[Topic Modeling Engine]
        SENT[Sentiment Analysis Engine]
        TONE[Tone Analysis Engine]
        AGG[Data Aggregation]
    end
    
    subgraph "Analytics & Detection"
        STAT[Statistical Analysis]
        ANOM[Anomaly Detection]
        CORR[Cross-Bank Correlation]
        TREND[Trend Analysis]
    end
    
    subgraph "Risk Intelligence"
        SCORE[Risk Scoring]
        ALERT[Alert Generation]
        EXPLAIN[Explainability Engine]
        RANK[Risk Ranking]
    end
    
    subgraph "Output & Monitoring"
        DASH[Risk Dashboard]
        REPORT[Automated Reports]
        API[REST API]
        MONITOR[System Monitoring]
    end
    
    DS1 --> ETL
    DS2 --> ETL
    DS3 --> ETL
    DS4 --> ETL
    
    ETL --> PARSE
    PARSE --> CLEAN
    CLEAN --> VALID
    VALID --> RAW
    
    RAW --> PROC
    PROC --> TOPIC
    PROC --> SENT
    PROC --> TONE
    
    TOPIC --> AGG
    SENT --> AGG
    TONE --> AGG
    AGG --> META
    
    META --> STAT
    STAT --> ANOM
    STAT --> CORR
    STAT --> TREND
    
    ANOM --> SCORE
    CORR --> SCORE
    TREND --> SCORE
    SCORE --> ALERT
    ALERT --> EXPLAIN
    EXPLAIN --> RANK
    
    RANK --> DASH
    RANK --> REPORT
    RANK --> API
    
    MODEL --> TOPIC
    MODEL --> SENT
    MODEL --> ANOM
    
    DASH --> MONITOR
    REPORT --> MONITOR
    API --> MONITOR
```

## 2. Topic Modeling Workflow

```mermaid
graph TD
    START[Input: Sentence-level Data] --> PREPROCESS[Text Preprocessing]
    PREPROCESS --> SEED_CHECK{Seed Topic Match?}
    
    SEED_CHECK -->|Yes| SEED_ASSIGN[Assign Seed Topic]
    SEED_CHECK -->|No| REMAINING[Add to Remaining Pool]
    
    SEED_ASSIGN --> SEED_CONF[Calculate Confidence Score]
    SEED_CONF --> SEED_OUTPUT[Seed Topic Output]
    
    REMAINING --> POOL_SIZE{Pool Size >= Min?}
    POOL_SIZE -->|No| MISC[Label as Miscellaneous]
    POOL_SIZE -->|Yes| EMBED[Generate Embeddings]
    
    EMBED --> CLUSTER[HDBSCAN Clustering]
    CLUSTER --> REPRESENT[Topic Representation]
    REPRESENT --> COHERENCE[Coherence Validation]
    
    COHERENCE --> COHERENCE_CHECK{Coherence >= Threshold?}
    COHERENCE_CHECK -->|No| MISC
    COHERENCE_CHECK -->|Yes| EMERGENT[Emergent Topic Assignment]
    
    EMERGENT --> EMERGENT_OUTPUT[Emergent Topic Output]
    MISC --> MISC_OUTPUT[Miscellaneous Output]
    
    SEED_OUTPUT --> COMBINE[Combine Results]
    EMERGENT_OUTPUT --> COMBINE
    MISC_OUTPUT --> COMBINE
    
    COMBINE --> FINAL[Final Topic Assignments]
    
    style SEED_ASSIGN fill:#e1f5fe
    style EMERGENT fill:#f3e5f5
    style MISC fill:#fff3e0
```

## 3. Sentiment Analysis Pipeline

```mermaid
graph LR
    subgraph "Input Processing"
        TEXT[Input Text]
        PREPROC[Preprocessing]
        BATCH[Batch Formation]
    end
    
    subgraph "FinBERT Analysis"
        FINBERT[FinBERT Model]
        SENTIMENT[Sentiment Classification]
        CONFIDENCE[Confidence Scoring]
    end
    
    subgraph "Tone Analysis"
        HEDGE[Hedging Detection]
        UNCERTAIN[Uncertainty Analysis]
        FORMAL[Formality Scoring]
        COMPLEX[Complexity Analysis]
    end
    
    subgraph "Risk-Specific Features"
        RISK_LANG[Risk Language Detection]
        ESCALATION[Escalation Markers]
        MITIGATION[Mitigation Signals]
    end
    
    subgraph "Aggregation"
        SPEAKER_WEIGHT[Speaker Weighting]
        TOPIC_WEIGHT[Topic Weighting]
        TEMPORAL[Temporal Aggregation]
    end
    
    subgraph "Output"
        SENT_SCORE[Sentiment Scores]
        TONE_FEATURES[Tone Features]
        RISK_INDICATORS[Risk Indicators]
    end
    
    TEXT --> PREPROC
    PREPROC --> BATCH
    BATCH --> FINBERT
    
    FINBERT --> SENTIMENT
    SENTIMENT --> CONFIDENCE
    
    BATCH --> HEDGE
    BATCH --> UNCERTAIN
    BATCH --> FORMAL
    BATCH --> COMPLEX
    
    BATCH --> RISK_LANG
    BATCH --> ESCALATION
    BATCH --> MITIGATION
    
    CONFIDENCE --> SPEAKER_WEIGHT
    HEDGE --> SPEAKER_WEIGHT
    UNCERTAIN --> SPEAKER_WEIGHT
    FORMAL --> SPEAKER_WEIGHT
    COMPLEX --> SPEAKER_WEIGHT
    RISK_LANG --> SPEAKER_WEIGHT
    ESCALATION --> SPEAKER_WEIGHT
    MITIGATION --> SPEAKER_WEIGHT
    
    SPEAKER_WEIGHT --> TOPIC_WEIGHT
    TOPIC_WEIGHT --> TEMPORAL
    
    TEMPORAL --> SENT_SCORE
    TEMPORAL --> TONE_FEATURES
    TEMPORAL --> RISK_INDICATORS
```

## 4. Anomaly Detection Framework

```mermaid
graph TB
    subgraph "Data Preparation"
        HIST[Historical Data]
        CURRENT[Current Quarter Data]
        FEATURES[Feature Engineering]
    end
    
    subgraph "Statistical Analysis"
        TREND[Trend Analysis]
        SEASONAL[Seasonal Decomposition]
        BASELINE[Baseline Calculation]
    end
    
    subgraph "Anomaly Detection Methods"
        ISO[Isolation Forest]
        CHANGE[Change Point Detection]
        OUTLIER[Statistical Outliers]
        CORR[Correlation Analysis]
    end
    
    subgraph "Cross-Bank Analysis"
        BANK_CORR[Bank Correlation Matrix]
        CLUSTER[Bank Clustering]
        CONSENSUS[Consensus Signals]
    end
    
    subgraph "Significance Testing"
        STAT_TEST[Statistical Tests]
        EFFECT_SIZE[Effect Size Calculation]
        POWER[Power Analysis]
        CORRECTION[Multiple Testing Correction]
    end
    
    subgraph "Risk Scoring"
        BASE_SCORE[Base Anomaly Score]
        CROSS_BOOST[Cross-Bank Boost]
        PERSIST[Persistence Factor]
        FINAL_SCORE[Final Risk Score]
    end
    
    subgraph "Alert Generation"
        THRESHOLD[Threshold Checking]
        SEVERITY[Severity Classification]
        EXPLANATION[Generate Explanations]
        ALERT_OUT[Alert Output]
    end
    
    HIST --> FEATURES
    CURRENT --> FEATURES
    FEATURES --> TREND
    FEATURES --> SEASONAL
    FEATURES --> BASELINE
    
    TREND --> ISO
    SEASONAL --> CHANGE
    BASELINE --> OUTLIER
    FEATURES --> CORR
    
    ISO --> BANK_CORR
    CHANGE --> BANK_CORR
    OUTLIER --> BANK_CORR
    CORR --> BANK_CORR
    
    BANK_CORR --> CLUSTER
    CLUSTER --> CONSENSUS
    
    CONSENSUS --> STAT_TEST
    STAT_TEST --> EFFECT_SIZE
    EFFECT_SIZE --> POWER
    POWER --> CORRECTION
    
    CORRECTION --> BASE_SCORE
    BASE_SCORE --> CROSS_BOOST
    CROSS_BOOST --> PERSIST
    PERSIST --> FINAL_SCORE
    
    FINAL_SCORE --> THRESHOLD
    THRESHOLD --> SEVERITY
    SEVERITY --> EXPLANATION
    EXPLANATION --> ALERT_OUT
    
    style ISO fill:#ffebee
    style CHANGE fill:#e8f5e8
    style OUTLIER fill:#fff3e0
    style CORR fill:#e3f2fd
```

## 5. Data Flow Architecture

```mermaid
graph LR
    subgraph "Raw Data Layer"
        US_BANKS[(US Banks Data)]
        EU_BANKS[(EU Banks Data)]
        EXTERNAL[(External Data)]
    end
    
    subgraph "Processing Layer"
        ETL_PROC[ETL Processing]
        NLP_PROC[NLP Processing]
        STAT_PROC[Statistical Processing]
    end
    
    subgraph "Storage Layer"
        SENTENCES[(Sentence Store)]
        TOPICS[(Topic Store)]
        SENTIMENT[(Sentiment Store)]
        ALERTS[(Alert Store)]
        MODELS[(Model Store)]
    end
    
    subgraph "Analytics Layer"
        TOPIC_ENGINE[Topic Engine]
        SENT_ENGINE[Sentiment Engine]
        ANOM_ENGINE[Anomaly Engine]
        RISK_ENGINE[Risk Engine]
    end
    
    subgraph "API Layer"
        REST_API[REST API]
        GRAPHQL[GraphQL API]
        WEBSOCKET[WebSocket API]
    end
    
    subgraph "Presentation Layer"
        DASHBOARD[Risk Dashboard]
        REPORTS[Automated Reports]
        ALERTS_UI[Alert Interface]
        MOBILE[Mobile App]
    end
    
    US_BANKS --> ETL_PROC
    EU_BANKS --> ETL_PROC
    EXTERNAL --> ETL_PROC
    
    ETL_PROC --> NLP_PROC
    NLP_PROC --> STAT_PROC
    
    STAT_PROC --> SENTENCES
    STAT_PROC --> TOPICS
    STAT_PROC --> SENTIMENT
    STAT_PROC --> ALERTS
    
    SENTENCES --> TOPIC_ENGINE
    TOPICS --> SENT_ENGINE
    SENTIMENT --> ANOM_ENGINE
    ALERTS --> RISK_ENGINE
    
    MODELS --> TOPIC_ENGINE
    MODELS --> SENT_ENGINE
    MODELS --> ANOM_ENGINE
    
    TOPIC_ENGINE --> REST_API
    SENT_ENGINE --> GRAPHQL
    ANOM_ENGINE --> WEBSOCKET
    RISK_ENGINE --> REST_API
    
    REST_API --> DASHBOARD
    GRAPHQL --> REPORTS
    WEBSOCKET --> ALERTS_UI
    REST_API --> MOBILE
```

## 6. Alert Generation Workflow

```mermaid
graph TD
    START[Risk Score Calculation] --> THRESHOLD{Score >= Threshold?}
    
    THRESHOLD -->|No| STORE[Store for Monitoring]
    THRESHOLD -->|Yes| SEVERITY[Determine Severity Level]
    
    SEVERITY --> CRITICAL{Critical Level?}
    SEVERITY --> HIGH{High Level?}
    SEVERITY --> MEDIUM{Medium Level?}
    
    CRITICAL -->|Yes| IMMEDIATE[Immediate Notification]
    HIGH -->|Yes| PRIORITY[Priority Queue]
    MEDIUM -->|Yes| STANDARD[Standard Queue]
    
    IMMEDIATE --> EXPLAIN[Generate Explanations]
    PRIORITY --> EXPLAIN
    STANDARD --> EXPLAIN
    
    EXPLAIN --> SHAP[SHAP Analysis]
    EXPLAIN --> CONTEXT[Historical Context]
    EXPLAIN --> PEER[Peer Comparison]
    
    SHAP --> ENRICH[Enrich Alert]
    CONTEXT --> ENRICH
    PEER --> ENRICH
    
    ENRICH --> VALIDATE[Validation Checks]
    VALIDATE --> DUPLICATE{Duplicate Alert?}
    
    DUPLICATE -->|Yes| MERGE[Merge with Existing]
    DUPLICATE -->|No| ROUTE[Route to Recipients]
    
    MERGE --> UPDATE[Update Existing Alert]
    ROUTE --> EMAIL[Email Notification]
    ROUTE --> SLACK[Slack Notification]
    ROUTE --> DASHBOARD_UPDATE[Dashboard Update]
    
    UPDATE --> TRACK[Track Alert Status]
    EMAIL --> TRACK
    SLACK --> TRACK
    DASHBOARD_UPDATE --> TRACK
    
    TRACK --> FEEDBACK[Collect Feedback]
    FEEDBACK --> LEARN[Update Models]
    
    STORE --> MONITOR[Background Monitoring]
    MONITOR --> TREND_CHECK{Trend Change?}
    TREND_CHECK -->|Yes| SEVERITY
    TREND_CHECK -->|No| CONTINUE[Continue Monitoring]
    
    style CRITICAL fill:#ffcdd2
    style HIGH fill:#fff3e0
    style MEDIUM fill:#e8f5e8
    style IMMEDIATE fill:#ff5722,color:#fff
```

## 7. Model Training and Deployment Pipeline

```mermaid
graph LR
    subgraph "Data Preparation"
        RAW_DATA[Raw Training Data]
        CLEAN_DATA[Data Cleaning]
        FEATURE_ENG[Feature Engineering]
        SPLIT[Train/Val/Test Split]
    end
    
    subgraph "Model Development"
        TOPIC_TRAIN[Topic Model Training]
        SENT_TRAIN[Sentiment Model Training]
        ANOM_TRAIN[Anomaly Model Training]
        HYPEROPT[Hyperparameter Optimization]
    end
    
    subgraph "Model Validation"
        CROSS_VAL[Cross Validation]
        METRICS[Performance Metrics]
        BIAS_TEST[Bias Testing]
        COHERENCE[Coherence Testing]
    end
    
    subgraph "Model Registry"
        VERSION[Model Versioning]
        METADATA[Model Metadata]
        ARTIFACTS[Model Artifacts]
        LINEAGE[Data Lineage]
    end
    
    subgraph "Deployment"
        STAGING[Staging Environment]
        A_B_TEST[A/B Testing]
        CANARY[Canary Deployment]
        PRODUCTION[Production Deployment]
    end
    
    subgraph "Monitoring"
        PERFORMANCE[Performance Monitoring]
        DRIFT[Model Drift Detection]
        FEEDBACK[Feedback Loop]
        RETRAIN[Automated Retraining]
    end
    
    RAW_DATA --> CLEAN_DATA
    CLEAN_DATA --> FEATURE_ENG
    FEATURE_ENG --> SPLIT
    
    SPLIT --> TOPIC_TRAIN
    SPLIT --> SENT_TRAIN
    SPLIT --> ANOM_TRAIN
    SPLIT --> HYPEROPT
    
    TOPIC_TRAIN --> CROSS_VAL
    SENT_TRAIN --> CROSS_VAL
    ANOM_TRAIN --> CROSS_VAL
    HYPEROPT --> METRICS
    
    CROSS_VAL --> METRICS
    METRICS --> BIAS_TEST
    BIAS_TEST --> COHERENCE
    
    COHERENCE --> VERSION
    VERSION --> METADATA
    METADATA --> ARTIFACTS
    ARTIFACTS --> LINEAGE
    
    LINEAGE --> STAGING
    STAGING --> A_B_TEST
    A_B_TEST --> CANARY
    CANARY --> PRODUCTION
    
    PRODUCTION --> PERFORMANCE
    PERFORMANCE --> DRIFT
    DRIFT --> FEEDBACK
    FEEDBACK --> RETRAIN
    
    RETRAIN --> TOPIC_TRAIN
```

---

*These diagrams provide a comprehensive visual representation of the risk monitoring system architecture, showing data flow, processing pipelines, and component interactions.*