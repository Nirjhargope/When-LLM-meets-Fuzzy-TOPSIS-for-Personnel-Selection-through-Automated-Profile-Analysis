# When-LLM-meets-Fuzzy-TOPSIS-for-Personnel-Selection-through-Automated-Profile-Analysis


## Overview
This repository presents a hybrid, end-to-end intelligent decision-support framework that integrates Large Language Models (LLMs) with Fuzzy TOPSIS to enable automated, fair, and explainable personnel selection. The system transforms unstructured candidate profiles into structured, quantitative decision inputs and ranks candidates based on multiple evaluation criteria.
The core idea is simple yet powerful: combine deep semantic understanding from LLMs with a robust multi-criteria decision-making (MCDM) technique to support expert-level hiring decisions at scale.
---
# Key Contributions

* Automated feature extraction from candidate profiles using **Transformer-based LLMs**
* Column-wise semantic scoring (About, Education, Experience, Skills, Overall)
* Integration of **Fuzzy TOPSIS** for uncertainty-aware decision-making
* Expert-informed evaluation rubric and model comparison
* Clear visual explainability via structured plots and diagrams
---
## 🧠 System Architecture (Visual Assets)
[cite_start]All high-level visual explanations are organized under the `assets/` directory[cite: 37]:

* [cite_start]**Attention Mechanism:** `assets/attention_mechanism/` — Top-level overview of the Attention head[cite: 38, 39].
* [cite_start]**Dataset Overview:** `assets/dataset_overview/` — Professional background distribution of candidates[cite: 40, 41].
* [cite_start]**Exploratory Analysis:** `assets/exploratory_analysis/` — Correlation Heatmap of features and labels[cite: 42].
* [cite_start]**Model Internals:** `assets/model_internals/` — Top-level overview of the Model’s internals[cite: 43].
* [cite_start]**Decision Framework:** `assets/decision_framework/` — Top-level overview of the LLM-Fuzzy TOPSIS[cite: 44].

---

## 📂 Repository Structure
* [cite_start]**`assets/`**: Conceptual and analytical visualizations[cite: 55, 62].
* [cite_start]**`notebooks/`**: Model-wise and column-wise experiments[cite: 56, 63].
* [cite_start]**`results/`**: Metrics, plots, tables, and final comparisons[cite: 57, 64].
* [cite_start]**`src/`**: Core source code for modeling[cite: 65].
* [cite_start]**`fuzzy_topsis/`**: Fuzzy TOPSIS implementation[cite: 58, 66].
* **`requirements.txt`**: Project dependencies[cite: 61].

---

## 🔬 Models and Experiments
Experiments are conducted using multiple transformer architectures[cite: 68]:
* [cite_start]**DistilRoBERTa-base** [cite: 69]
* [cite_start]**RoBERTa-base** [cite: 70]
* **LastBERT** [cite: 71]

Each model is evaluated column-wise (About, Education, Experience, Skills, Overall) to ensure fine-grained semantic assessment[cite: 72].

---

## 📊 Results & Evaluation
The `results/` directory contains:
* [cite_start]Accuracy vs Loss curves[cite: 88].
* [cite_start]Confusion matrices per column and per model[cite: 89].
* Final TOPSIS-based rankings[cite: 90].
* [cite_start]Expert evaluation rubric[cite: 91].
* [cite_start]Comparative performance tables and visual summaries[cite: 92].

[cite_start]Final decision artifacts are consolidated under `results/final_results/`[cite: 93, 94].

---

## ⚖️ Decision Making with Fuzzy TOPSIS
Fuzzy TOPSIS is employed to:
* **Handle ambiguity** and uncertainty in candidate evaluation[cite: 98].
* [cite_start]**Integrate expert-defined criteria weights**[cite: 99].
* [cite_start]**Produce stable and interpretable** candidate rankings[cite: 100].

---

## 🛠️ Setup & Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
