# Incident Root Cause Analysis (RCA) System

An automated incident diagnosis system using **ML classification** + **LLM reasoning** to help developers and engineers quickly identify root causes and generate actionable resolutions for production incidents.

---

## Overview

This system implements a two-stage pipeline:

1. **Stage 1 - ML Classifier**: An LSTM-based text classifier that categorizes incident descriptions into 12 predefined categories with confidence scores
2. **Stage 2 - LLM Resolution**: Mistral 7B generates detailed root cause analysis and resolution steps based on the predicted category

Resolution quality is scored by an evaluation framework that uses **Llama 3.2 (3B)** as an independent judge.

---

## What's in it

| RCA generator | Mistral 7B |
| LLM judge | Llama 3.2 (3B) |
| Classifier | BiLSTM |

---

## Features

- **Automated Classification**: Predicts incident category with 87% accuracy on test data
- **Structured Resolutions**: Generates 5-section responses (Root Cause, Immediate Actions, Solutions, Prevention, Escalation)
- **Quality Evaluation**: Built-in evaluation framework with specificity scoring, structure validation, and LLM-as-Judge
- **Interactive UI**: Streamlit-based web interface for easy interaction
- **A/B Testing**: Experiment framework to compare LLM-only vs ML+LLM approaches

---

## Categories

The system classifies incidents into 12 categories:

| ID | Category             | Description                           |
| -- | -------------------- | ------------------------------------- |
| 1  | Configuration Issue  | Config file errors, misconfigurations |
| 2  | Database Issue       | DB connections, queries, performance  |
| 3  | Deployment Issue     | Container, CI/CD, release problems    |
| 4  | Infrastructure Issue | Server, cloud, hardware issues        |
| 5  | Integration Issue    | API, service communication problems   |
| 6  | Memory Issue         | Memory leaks, heap exhaustion         |
| 7  | Network Hardware     | Physical network equipment            |
| 8  | Network Issue        | Connectivity, latency, DNS            |
| 9  | Other                | Miscellaneous issues                  |
| 10 | Performance Issue    | Slow response, high CPU/memory        |
| 11 | Security Issue       | Auth failures, unauthorized access    |
| 12 | Software Error       | Application bugs, exceptions          |

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
│   ├── integrated.py                 # ML + LLM integration pipeline (RCA generator)
│   ├── LLM_evaluate.py               # Evaluation framework (LLM judge)
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
- [Ollama](https://ollama.com/download) for running local LLMs
- 16GB RAM recommended (Mistral 7B and Llama 3.2 3B)
- NVIDIA GPU (optional, for faster inference)

### Step 1: Clone the Repository

```
git clone https://github.com/Charulyadav/IncidentRCA.git
cd IncidentRCA
```

### Step 2: Create Virtual Environment

Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

Linux/Mac:
```
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```
pip install -r final_run/requirements.txt
```

### Step 4: Install and Configure Ollama

1. Download Ollama from https://ollama.com/download
2. Install and run the installer
3. Pull the required models:

```
ollama pull mistral:7b
ollama pull llama3.2:3b
```

### Step 5: Start Ollama Server

```
ollama serve
```

Keep this terminal open while using the application.

---

## Usage

### Run the Streamlit UI

```
cd final_run
streamlit run app.py
```

Opens your browser to locally run the app.

### Run Single Text Classification

```
cd final_run
python test.py "Database connection timeout error during query execution"
```

### Run Full Pipeline (CLI)

```
cd final_run
python integrated.py
```

### Run Evaluation

```
cd final_run
python LLM_evaluate.py
```

### Run A/B Experiment

```
cd final_run
python ab_experiment.py
```

---

## Model Architecture

### Model Configuration

| Parameter           | Value |
| ------------------- | ----- |
| Embedding Dimension | 128   |
| Hidden Dimension    | 256   |
| LSTM Layers         | 2     |
| Dropout             | 0.3   |
| Bidirectional       | True  |
| Vocabulary Size     | 2,642 |
| Number of Classes   | 12    |

---

## Training Results

### Overall Metrics

| Metric          | Score |
| --------------- | ----- |
| Accuracy        | 87.00% |
| Macro Precision | 80.50% |
| Macro Recall    | 88.70% |
| Macro F1        | 85.00% |
| Weighted F1     | 86.00% |

---

## Evaluation Framework

### 1. Specificity Score (Rule-based)

Measures concrete vs vague language in the response.

| Indicator Type          | Examples                                                     |
| ----------------------- | ------------------------------------------------------------ |
| **Specific (positive)** | Numbers, technical terms, action verbs, commands, file paths |
| **Vague (negative)**    | Hedging words ("maybe", "perhaps"), filler phrases           |

**Formula:** `score = (specificity_ratio * 70) + density_bonus`

### 2. Structure Validation (Rule-based)

Checks presence of required sections:

- Root Cause Analysis
- Immediate Actions
- Detailed Solutions
- Prevention Recommendations
- Escalation Guidance

### 3. LLM-as-Judge (Model-based)

Llama 3.2 (3B) rates responses on 6 dimensions (1-10 scale):

| Dimension       | Description                            |
| --------------- | -------------------------------------- |
| Relevance       | Does it address the specific incident? |
| Accuracy        | Is the technical information correct?  |
| Completeness    | Are all aspects covered?               |
| Actionability   | Can someone act on this immediately?   |
| Clarity         | Is it easy to understand?              |
| Technical Depth | Is there sufficient technical detail?  |

### Overall Score Calculation

Overall = (Specificity × 0.3) + (Structure × 0.2) + (LLM Judge × 10 × 0.5)

---

**Recommendations:**

1. Implement confidence thresholding: only use ML predictions when confidence ≥ 85%
2. Retrain classifier with more data for Security, Network, and Infrastructure categories

---

## Tech Stack

| Component          | Technology                           |
| ------------------ | ------------------------------------ |
| ML Framework       | PyTorch                              |
| Model Architecture | 2-layer LSTM with learned embeddings |
| LLM (Resolution)   | Mistral 7B via Ollama                |
| LLM (Judge)        | Llama 3.2 (3B) via Ollama            |
| UI Framework       | Streamlit                            |
| Data Processing    | Pandas, NumPy, scikit-learn          |
| Configuration      | YAML                                 |

---

## Dependencies

```
pandas==2.1.0
numpy==1.24.0
PyYAML==6.0.2
torch==2.5.0
requests
streamlit
scikit-learn
```

---

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the UI framework
- [PyTorch](https://pytorch.org/) for the ML framework
