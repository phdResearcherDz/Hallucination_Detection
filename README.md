# MedHallu Graph-Based Hallucination Detection Pipeline

This repository implements a multi-stage pipeline to detect and categorize hallucinations in biomedical question answering using the **MedHallu** dataset (`UTAustin-AIHealth/MedHallu`).

The system combines:
- LLM-based biomedical entity and relation extraction
- Graph construction and normalization
- Graph/text alignment feature engineering
- Classical ML and deep learning classifiers

---

## Pipeline Overview

### Stage 1 — Graph Construction (`_1_graph_builder.py`)

For each MedHallu sample, this stage:

- Extracts **biomedical entities** from:
  - Question (Q)
  - Knowledge context (K)
  - Hallucinated Answer (A)
- Extracts **relations** between entities using an LLM (Ollama).
- Performs **entity normalization across Q/K/A** using embedding similarity.
- Builds three directed graphs:
  - Question graph (`Gq`)
  - Knowledge graph (`Gk`)
  - Answer graph (`Ga`)
- Stores graphs and embeddings as JSON.

**Output directory**
```
graph_outputs_not_labled/
```

---

### Stage 2 — Feature Extraction (`_2_feature_extractor.py`)

This stage loads graphs from Stage 1 and computes numerical features:

#### Text-level features
- Cosine similarity between answer and question/knowledge embeddings
- Lexical Jaccard overlap

#### Graph-level features
- Mean node embedding similarity
- Graph density, number of nodes/edges
- Edge coherence scores

#### Alignment features
- Node and edge alignment between answer graph and question/knowledge graphs

Features are written in JSONL format.

**Output**
```
features_not_labeled/features.jsonl
```

---

### Stage 3 — Training & Evaluation

You can use multiple scripts:

#### `_3_train_best.py`
- Trains **TabNet** models
- Binary hallucination detection
- Multi-class hallucination category prediction

#### `_3_train_test_classifier_performance.py`
- Trains **XGBoost**
- Evaluates binary + multi-class performance
- Exports top feature importance

#### `4_get_results.py`
- Large-scale benchmarking framework
- Multiple feature subsets:
  - TextOnly
  - GraphLevelOnly
  - AlignmentOnly
  - AllFeatures
- Multiple models:
  - Logistic Regression
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - AdaBoost
  - SVM
  - MLP
  - XGBoost (optional)
  - LightGBM (optional)
  - Torch TabMLP
  - FeatBERT (Transformer over features)
- Threshold tuning and PDF result export

---

## Installation

### System requirements
- Python 3.9+
- Ollama installed and running locally
- (Optional) CUDA-capable GPU

### Ollama model
```bash
ollama pull gemma3:12b
ollama serve
```

### Python dependencies
```bash
pip install -U \
  numpy \
  regex \
  networkx \
  datasets \
  sentence-transformers \
  scikit-learn \
  matplotlib \
  pandas \
  torch \
  xgboost \
  pytorch-tabnet \
  ollama
```

Optional:
```bash
pip install lightgbm
```

---

## How to Run

### Step 1 — Build graphs
```bash
python _1_graph_builder.py
```

Adjust `MAX_SAMPLES` inside the script for quick testing.

---

### Step 2 — Extract features
```bash
python _2_feature_extractor.py
```

---

### Step 3 — Train models

TabNet:
```bash
python _3_train_best.py
```

XGBoost:
```bash
python _3_train_test_classifier_performance.py
```

Full benchmark:
```bash
python 4_get_results.py
```

---

## Directory Structure

```
.
├── _1_graph_builder.py
├── _2_feature_extractor.py
├── _3_train_best.py
├── _3_train_test_classifier_performance.py
├── 4_get_results.py
├── llm_cache_unlabeld/
├── graph_outputs_not_labled/
├── features_not_labeled/
└── results_binary_*/
```

---

## Dataset

- **MedHallu**: `UTAustin-AIHealth/MedHallu`
- Accessed via HuggingFace `datasets`


