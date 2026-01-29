# When LLM Meets Fuzzy TOPSIS for Personnel Selection through Automated Profile Analysis

## Overview

This repository presents a hybrid, end-to-end intelligent decision-support framework that integrates Large Language Models (LLMs) with Fuzzy TOPSIS to enable automated, fair, and explainable personnel selection. The system transforms unstructured candidate profiles into structured, quantitative decision inputs and ranks candidates based on multiple evaluation criteria. The core idea is simple yet powerful: combine deep semantic understanding from LLMs with a robust multi-criteria decision-making (MCDM) technique to support expert-level hiring decisions at scale.

---
# Key Contributions

* Automated feature extraction from candidate profiles using **Transformer-based LLMs**
* Column-wise semantic scoring (About, Education, Experience, Skills, Overall)
* Integration of **Fuzzy TOPSIS** for uncertainty-aware decision-making
* Expert-informed evaluation rubric and model comparison
* Clear visual explainability via structured plots and diagrams
---
## System Architecture (Visual Assets)
All high-level visual explanations are organized under the `assets/` directory:

* **Attention Mechanism:** `assets/attention_mechanism/` — Top-level overview of the Attention head.
* **Dataset Overview:** `assets/dataset_overview/` — Professional background distribution of candidates.
* **Exploratory Analysis:** `assets/exploratory_analysis/` — Correlation Heatmap of features and labels.
* **Model Internals:** `assets/model_internals/` — Top-level overview of the Model’s internals.
* **Decision Framework:** `assets/decision_framework/` — Top-level overview of the LLM-Fuzzy TOPSIS.

---

## Repository Structure
* **`assets/`**: Conceptual and analytical visualizations.
* **`notebooks/`**: Model-wise and column-wise experiments.
* **`results/`**: Metrics, plots, tables, and final comparisons.
* **`src/`**: Core source code for modeling.
* **`fuzzy_topsis/`**: Fuzzy TOPSIS implementation.
* **`requirements.txt`**: Project dependencies.

---

## Models and Experiments
Experiments are conducted using multiple LLM  model architectures:
* **DistilRoBERTa-base** 
* **RoBERTa-base** 
* **LastBERT** 

Each model is evaluated column-wise (About, Education, Experience, Skills, Overall) to ensure fine-grained semantic assessment.

---

## Results & Evaluation
The `results/` directory contains:
* Accuracy vs Loss curves.
* Confusion matrices per column and per model.
* Final TOPSIS-based rankings.
* Expert evaluation rubric.
* Comparative performance tables and visual summaries.

Final decision artifacts are consolidated under `results/final_results/`.
![Final Results Overview](results/final_results/topsis_model_ranking%20table.png)

---

## Decision Making with Fuzzy TOPSIS
Fuzzy TOPSIS is employed to:
* **Handle ambiguity** and uncertainty in candidate evaluation.
* **Integrate expert-defined criteria weights**.
* **Produce stable and interpretable** candidate rankings.

---

##  Setup & Requirements
Install dependencies using:
also Key libraries include PyTorch, HuggingFace Transformers, Scikit-learn, and NumPy.
```bash
pip install -r requirements.txt
```

---
##  Use Cases
* Automated recruitment systems.
* Decision-support tools for HR analytics.
* Research on LLM-driven MCDM frameworks.
* Explainable AI in personnel selection.
---

## Final Note
This work demonstrates how **LLMs can move beyond prediction into structured decision-making**, bridging the gap between unstructured human data and formal evaluation frameworks. We hope this repository inspires further research at the intersection of **NLP, explainable AI, and decision science**.

