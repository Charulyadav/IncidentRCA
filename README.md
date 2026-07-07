# Incident Root Cause Analysis (RCA) System

An automated incident diagnosis system using **ML classification** + **LLM reasoning** to help developers and engineers quickly identify root causes and generate actionable resolutions for production incidents.

---

## Overview

This system implements a two-stage pipeline:

1. **Stage 1 - ML Classifier**: An LSTM-based text classifier that categorizes incident descriptions into 12 predefined categories with confidence scores
2. **Stage 2 - LLM Resolution**: A Large Language Model (Llama 3.2) that generates detailed root cause analysis and resolution steps based on the predicted category

---

## Features

- **Automated Classification**: Predicts incident category with 99.3% accuracy on test data
- **Structured Resolutions**: Generates 5-section responses (Root Cause, Immediate Actions, Solutions, Prevention, Escalation)
- **Quality Evaluation**: Built-in evaluation framework with specificity scoring, structure validation, and LLM-as-Judge
- **Interactive UI**: Streamlit-based web interface for easy interaction
- **A/B Testing**: Experiment framework to compare LLM-only vs ML+LLM approaches

---

## Categories

The system classifies incidents into 12 categories:

| ID | Category | Description |
|----|----------|-------------|
| 1 | Configuration Issue | Config file errors, misconfigurations |
| 2 | Database Issue | DB connections, queries, performance |
| 3 | Deployment Issue | Container, CI/CD, release problems |
| 4 | Infrastructure Issue | Server, cloud, hardware issues |
| 5 | Integration Issue | API, service communication problems |
| 6 | Memory Issue | Memory leaks, heap exhaustion |
| 7 | Network Hardware | Physical network equipment |
| 8 | Network Issue | Connectivity, latency, DNS |
| 9 | Other | Miscellaneous issues |
| 10 | Performance Issue | Slow response, high CPU/memory |
| 11 | Security Issue | Auth failures, unauthorized access |
| 12 | Software Error | Application bugs, exceptions |

---

## Project Structure

``` 
IncidentRCA/
├── data/
│   ├── raw/
│   │   └── incidents.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       ├── train_minimal.csv
│       ├── test_minimal.csv
│       └── label_mappings.json
├── src/
│   └── data/
│       ├── explore_data.py
│       └── preprocess_data.py
├── final_run/
│   ├── app.py                        # Streamlit UI application
│   ├── integrated.py                 # ML + LLM integration pipeline
│   ├── LLM_evaluate.py               # Evaluation framework
│   ├── ab_experiment.py              # A/B experiment script
│   ├── test.py                       # ML classifier interface
│   ├── models.py                     # LSTM model class definition
│   ├── config.yaml                   # Model hyperparameters
│   ├── best_lstm_classifier.pt       # Trained LSTM weights
│   ├── vocab.json                    # Vocabulary mapping
│   ├── label_mappings.json           # Category ID to label mapping
│   ├── requirements.txt              # Python dependencies
│   ├── rca_result.json               # Sample output
│   ├── evaluation_result.json        # Sample evaluation output
│   └── ab_experiment_results.json    # A/B experiment results
├── outputs/
│   ├── best_lstm_classifier.pt
│   └── vocab.json
├── classification_metrics.py
├── classification_metrics.jpg
├── config.yaml
├── data.py
├── models.py
├── train.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Installation

### Prerequisites

- Python 3.10+
- Ollama (https://ollama.com/download) for running local LLMs
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for faster inference)

### Step 1: Clone the Repository

git clone https://github.com/grsx394/IncidentRCA.git
cd IncidentRCA

### Step 2: Create Virtual Environment

Windows:
python -m venv .venv
.venv\Scripts\activate

Linux/Mac:
python -m venv .venv
source .venv/bin/activate

### Step 3: Install Dependencies

pip install -r final_run/requirements.txt

### Step 4: Install and Configure Ollama

1. Download Ollama from https://ollama.com/download
2. Install and run the installer
3. Pull the required models:

ollama pull llama3.2:3b
ollama pull mistral:7b

### Step 5: Start Ollama Server

ollama serve

Keep this terminal open while using the application.

---

## Usage

### Run the Streamlit UI

cd final_run
streamlit run app.py

Opens your browser to locally run the app

### Run Single Text Classification

cd final_run
python test.py "Database connection timeout error during query execution"

Example Output:
Input Text: Database connection timeout error during query execution
Predicted Category: Database Issue
Confidence: 92.15%

### Run Full Pipeline (CLI)

cd final_run
python integrated.py

### Run Evaluation

cd final_run
python LLM_evaluate.py

### Run A/B Experiment

cd final_run
python ab_experiment.py

---

## Model Architecture

### LSTM Text Classifier
```
Input Text
    ↓
Embedding Layer (vocab_size → 128 dimensions) + Dropout (0.3)
    ↓
LSTM Layers (2 stacked, 128 → 256 hidden dimensions)
    ↓
Classifier Head: Dropout (0.3) + Linear (256 → 12 classes)
    ↓
Output: 12 class probabilities
```
### Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 128 |
| Hidden Dimension | 256 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Bidirectional | False |
| Vocabulary Size | 2,642 |
| Number of Classes | 12 |

---

## Training Results

### Overall Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 99.30% |
| Macro Precision | 99.36% |
| Macro Recall | 98.32% |
| Macro F1 | 98.79% |
| Weighted F1 | 99.28% |

### Per-Class Performance

| ID | Category | Precision | Recall | F1-Score | Support |
|----|----------|-----------|--------|----------|---------|
| 1 | Configuration Issue | 1.00 | 1.00 | 1.00 | 44 |
| 2 | Database Issue | 0.99 | 1.00 | 0.99 | 99 |
| 3 | Deployment Issue | 0.99 | 1.00 | 0.99 | 86 |
| 4 | Infrastructure Issue | 1.00 | 1.00 | 1.00 | 58 |
| 5 | Integration Issue | 0.96 | 1.00 | 0.98 | 49 |
| 6 | Memory Issue | 1.00 | 1.00 | 1.00 | 48 |
| 7 | Network Hardware | 1.00 | 0.92 | 0.96 | 13 |
| 8 | Network Issue | 0.98 | 1.00 | 0.99 | 61 |
| 9 | Other | 1.00 | 0.88 | 0.93 | 32 |
| 10 | Performance Issue | 1.00 | 1.00 | 1.00 | 49 |
| 11 | Security Issue | 1.00 | 1.00 | 1.00 | 55 |
| 12 | Software Error | 1.00 | 1.00 | 1.00 | 117 |

---

## Evaluation Framework

### 1. Specificity Score (Rule-based)

Measures concrete vs vague language in the response.

| Indicator Type | Examples |
|----------------|----------|
| **Specific (positive)** | Numbers, technical terms, action verbs, commands, file paths |
| **Vague (negative)** | Hedging words ("maybe", "perhaps"), filler phrases |

**Formula:** `score = (specificity_ratio * 70) + density_bonus`

### 2. Structure Validation (Rule-based)

Checks presence of required sections:
- Root Cause Analysis
- Immediate Actions
- Detailed Solutions
- Prevention Recommendations
- Escalation Guidance

### 3. LLM-as-Judge (Model-based)

Mistral 7B rates responses on 6 dimensions (1-10 scale):

| Dimension | Description |
|-----------|-------------|
| Relevance | Does it address the specific incident? |
| Accuracy | Is the technical information correct? |
| Completeness | Are all aspects covered? |
| Actionability | Can someone act on this immediately? |
| Clarity | Is it easy to understand? |
| Technical Depth | Is there sufficient technical detail? |

---
### Overall Score Calculation

Overall = (Specificity × 0.3) + (Structure × 0.2) + (LLM Judge × 10 × 0.5)

---

## A/B Experiment Results

### Overview

We conducted an A/B experiment comparing two approaches:
- **Approach A (LLM-Only)**: LLM receives only issue name, description, and priority
- **Approach B (ML+LLM)**: LLM receives issue details + ML-predicted category + confidence score

### Results Summary (10 Test Cases)

| # | Test Case | ML Prediction | Confidence | LLM-Only | ML+LLM | Winner |
|---|-----------|---------------|------------|----------|--------|--------|
| 1 | JVM Heap Space Exhausted | Database Issue | 45.6% | 88.9 | 87.1 | TIE |
| 2 | Database Query Timeout | Database Issue | 99.8% | 85.8 | 92.9 | **ML+LLM** |
| 3 | Brute Force Attack | Deployment Issue | 73.4% | 88.2 | 85.6 | LLM-Only |
| 4 | DNS Resolution Failures | Other | 81.9% | 88.6 | 86.0 | LLM-Only |
| 5 | K8s Pod CrashLoopBackOff | Deployment Issue | 99.5% | 85.4 | 87.3 | TIE |
| 6 | Redis Cache Miss Spike | Configuration Issue | 55.0% | 84.1 | 84.1 | TIE |
| 7 | SSL Certificate Chain | Deployment Issue | 55.9% | 88.4 | 87.6 | TIE |
| 8 | Disk Space Critical | Integration Issue | 50.4% | 89.2 | 89.3 | TIE |
| 9 | Payment API Failures | Deployment Issue | 60.7% | 87.3 | 84.4 | LLM-Only |
| 10 | NullPointerException | Deployment Issue | 79.9% | 84.1 | 84.8 | TIE |

### Final Tally

| Winner | Count | Percentage |
|--------|-------|------------|
| LLM-Only | 3 | 30% |
| ML+LLM | 1 | 10% |
| TIE | 6 | 60% |

### Key Findings

**Finding 1: Classification Accuracy is Critical**

| ML Confidence | ML+LLM Performance |
|---------------|-------------------|
| 99.8% (Database) | +7.1 points (biggest win) |
| 99.5% (Deployment) | +1.9 points |
| 45-82% (various) | -0.8 to -2.9 points |

**Finding 2: Confidence Threshold Analysis**

| Confidence Range | Cases | ML+LLM Wins | LLM-Only Wins | Ties |
|------------------|-------|-------------|---------------|------|
| ≥90% | 2 | 1 | 0 | 1 |
| 60-89% | 4 | 0 | 3 | 1 |
| <60% | 4 | 0 | 0 | 4 |

**Finding 3: Misclassification Pattern**

The model over-predicts "Deployment Issue" (appeared 6 times, often incorrectly).

### Conclusion

The ML+LLM approach outperforms LLM-Only when the classifier achieves high confidence (≥90%) and correct predictions. When the classifier is wrong, ML+LLM either ties or underperforms.

**Recommendations:**
1. Implement confidence thresholding — only use ML predictions when confidence ≥ 85%
2. Retrain classifier with more data for Security, Network, and Infrastructure categories

---

## Sample Output

### Input

| Field | Value |
|-------|-------|
| Issue Name | Memory Leak in Production |
| Description | Application crashed with OutOfMemoryError. Heap space exhausted after processing large batch job. JVM heap set to 4GB. |
| Priority | P1 - Critical |

### ML Classification Result

| Field | Value |
|-------|-------|
| Predicted Category | Memory Issue |
| Confidence | 98.6% |
| Confidence Level | HIGH |

### LLM Resolution Output

Root Cause Analysis:
- Insufficient Heap Size: JVM heap set to 4GB may not be sufficient
- Inefficient Algorithm: Possible memory-intensive operations
- Resource-Intensive Components: Third-party libraries consuming memory

Immediate Actions:
1. Stop the batch job to prevent further damage
2. Temporarily increase heap size to 8GB

Detailed Solutions:
1. Increase Heap Size Permanently (16GB recommended)
2. Optimize Algorithm or Data Structure
3. Implement Memory Profiling
4. Add Garbage Collection Tuning

Prevention Recommendations:
- Regular memory monitoring
- Implement heap usage alerts
- Code review for memory-intensive operations

Escalation Guidance:
- Escalate to Development Team if solutions don't work
- Contact DevOps for infrastructure changes

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch |
| Model Architecture | 2-layer LSTM with learned embeddings |
| LLM (Resolution) | Llama 3.2 (3B) via Ollama |
| LLM (Judge) | Mistral 7B via Ollama |
| UI Framework | Streamlit |
| Data Processing | Pandas, NumPy, scikit-learn |
| Configuration | YAML |

---

## Dependencies

pandas==2.1.0
| numpy==1.24.0
| PyYAML==6.0.2
| torch==2.5.0
| requests
| streamlit
| scikit-learn

---

## Troubleshooting

### Ollama Connection Error
Error: Cannot connect to Ollama. Make sure it's running (ollama serve).
Solution: Open a new terminal and run ollama serve

### Model Not Found
Error: model 'llama3.2:3b' not found
Solution: Pull the model with ollama pull llama3.2:3b

### Out of Memory
CUDA out of memory
Solution: Use CPU instead or reduce batch size. The models are designed to run on CPU as well.

### Port Already in Use
Error: listen tcp 127.0.0.1:11434: bind: address already in use
Solution: Ollama is already running. You can proceed to use the application.

---

## Future Work

- **Cloud Deployment**: Deploy on GPU-enabled cloud platform (AWS, GCP, or Hugging Face Spaces) or integrate with serverless LLM APIs (Groq, Together AI)
- **Confidence Thresholding**: Only use ML predictions when confidence ≥ 85% to improve ML+LLM approach performance
- **Classifier Improvement**: Retrain with more balanced data for Security, Network, and Infrastructure categories to reduce "Deployment Issue" over-prediction
- **User Feedback Loop**: Allow users to correct predictions to continuously improve the model
- **Historical Analysis**: Store and analyze past incidents to identify recurring patterns
- **Integration**: Connect with ticketing systems (Jira, ServiceNow) for automated incident ingestion

---

## Team

Charul
| Gaurang Rajesh Sathe
| Sanika Sanjay Girase
| Tuhin Subhra De 

---

## License

This project is a part of CS 5130 coursework.

---

## Acknowledgments

- Ollama (https://ollama.com/) for local LLM inference
- Streamlit (https://streamlit.io/) for the UI framework
- PyTorch (https://pytorch.org/) for the ML framework
