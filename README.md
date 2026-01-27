# When LLM Meets Fuzzy TOPSIS for Personnel Selection through Automated Profile Analysis

## Overview

This repository presents a **hybrid, end-to-end intelligent decision-support framework** that integrates **Large Language Models (LLMs)** with **Fuzzy TOPSIS** to enable **automated, fair, and explainable personnel selection**. The system transforms unstructured candidate profiles into structured, quantitative decision inputs and ranks candidates based on multiple evaluation criteria.The core idea is simple yet powerful: **combine deep semantic understanding from LLMs with a robust multi-criteria decision-making (MCDM)** technique to support expert-level hiring decisions at scale.
---
# Key Contributions

* Automated feature extraction from candidate profiles using **Transformer-based LLMs**
* Column-wise semantic scoring (About, Education, Experience, Skills, Overall)
* Integration of **Fuzzy TOPSIS** for uncertainty-aware decision-making
* Expert-informed evaluation rubric and model comparison
* Clear visual explainability via structured plots and diagrams
---
## System Architecture (Visual Assets)
[cite_start]All high-level visual explanations are organized under the `assets/` directory[cite: 37]:

* **Attention Mechanism:** `assets/attention_mechanism/` — Top-level overview of the Attention head[cite: 38, 39].
* **Dataset Overview:** `assets/dataset_overview/` — Professional background distribution of candidates[cite: 40, 41].
* **Exploratory Analysis:** `assets/exploratory_analysis/` — Correlation Heatmap of features and labels[cite: 42].
* **Model Internals:** `assets/model_internals/` — Top-level overview of the Model’s internals[cite: 43].
* **Decision Framework:** `assets/decision_framework/` — Top-level overview of the LLM-Fuzzy TOPSIS[cite: 44].

---

##  Repository Structure
* **`assets/`**: Conceptual and analytical visualizations[cite: 55, 62].
* **`notebooks/`**: Model-wise and column-wise experiments[cite: 56, 63].
* **`results/`**: Metrics, plots, tables, and final comparisons[cite: 57, 64].
* **`src/`**: Core source code for modeling[cite: 65].
* **`fuzzy_topsis/`**: Fuzzy TOPSIS implementation[cite: 58, 66].
* **`requirements.txt`**: Project dependencies[cite: 61].

---

##  Models and Experiments
Experiments are conducted using multiple transformer architectures[cite: 68]:
* **DistilRoBERTa-base** [cite: 69]
* **RoBERTa-base** [cite: 70]
* **LastBERT** [cite: 71]

Each model is evaluated column-wise (About, Education, Experience, Skills, Overall) to ensure fine-grained semantic assessment[cite: 72].

---

##  Results & Evaluation
The `results/` directory contains:
* Accuracy vs Loss curves[cite: 88].
* Confusion matrices per column and per model.
* Final TOPSIS-based rankings.
* Expert evaluation rubric.
* Comparative performance tables and visual summaries.

Final decision artifacts are consolidated under `results/final_results/`[cite: 93, 94].

---

## ⚖️ Decision Making with Fuzzy TOPSIS
Fuzzy TOPSIS is employed to:
* **Handle ambiguity** and uncertainty in candidate evaluation.
* [cite_start]**Integrate expert-defined criteria weights**.
* [cite_start]**Produce stable and interpretable** candidate rankings.

---

## 🛠️ Setup & Requirements
Install dependencies using:
also Key libraries include PyTorch, HuggingFace Transformers, Scikit-learn, and NumPy.
```bash
pip install -r requirements.txt 
