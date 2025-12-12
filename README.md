# Early Harmful Content Detection Framework

This project adapts the **MIND** framework, *“MIND: Unsupervised Modeling of Internal States for Hallucination Detection of Large Language Models” (Findings of ACL 2024)*, to perform **early detection of harmful content** in Large Language Models (LLMs) and enable safe, context-aware redirection for child-focused applications.

---

## Overview

Our system monitors the internal hidden states of a Transformer-based LLM during generation to detect when harmful content is likely to be produced, before it appears in the final output.
A classifier is trained on hidden-state features derived from safe and harmful examples, and its predictions drive an adaptive redirection module that generates safe, contextually relevant alternatives.

---

## Getting Started

### Install Environment

#### 1. Create and activate the environment
conda create -n torch-nightly python=3.11.14
conda activate torch-nightly

#### 2. Install all Python dependencies from requirements.txt
pip install -r requirements.txt

---

## Usage Pipeline

### Step 1: Generate Data

Run the script to automatically construct and label data for harmful vs. safe content (you can adjust model names, paths, and hyperparameters inside the script).

python generate_data.py


This produces JSON files under the configured `output_path` (e.g., `./auto-labeled/output/{model_name}/data_{datasplit}.json` with `datasplit` in `{train, valid, test}`) containing original text, modified (harmful) variants, and entity-level annotations.

---

### Step 2: Extract Hidden-State Features

Generate hidden-state features needed for training the classifier.

python generate_hd.py


This creates feature files such as `last_token_mean_{datasplit}.json` and `last_mean_{datasplit}.json`, where each entry stores hidden states for safe (“right”) and harmful (“harmfull”) variants of each base sentence.

---

### Step 3: Train the Harmful Content Classifier

Train a classifier on the extracted features.

The classifier learns to distinguish harmful from safe content based on internal activation patterns, and the best-performing model is saved for deployment.

python ./auto-labeled/train/train.py

### Step 4: Train the Topic (Context) Classifier

Train a topic or context classifier that predicts the user’s topic or intent (e.g., “games”, “school”, “health”) from the input text or its hidden-state representation.

python train_topic_classifier.py

This script should take your labeled topic dataset, encode it (optionally through the same LLM to get hidden states), and train a classifier whose output will later guide the suggestion of safe, contextually relevant alternatives


### Step 5: Run the Suggestor and Demo

Once the harmful-content classifier and topic classifier are trained, use the suggestion module to perform **end‑to‑end detection and redirection** and optionally show an interactive demo.

python suggestor.py

In typical usage, `suggestor.py` should:
- Load the LLM, the harmful-content classifier, and the topic classifier.
- Monitor hidden states for each user input in real time.
- If harmful content is detected, use the topic classifier output to create a safe, context-aware alternative response.
- Optionally launch a simple CLI or web demo that shows the original risky output vs. the redirected safe suggestion for inspection and testing.

This final step demonstrates the full pipeline: early harmful content detection, topic understanding, and adaptive safe content generation in a single interface.


## Acknowledgements

This project is built on top of the original **MIND** framework and codebase: *“MIND: Unsupervised Modeling of Internal States for Hallucination Detection of Large Language Models” (Findings of ACL 2024)*.  
Please consider citing the MIND paper and supporting the original repository if you use this work in your research. 