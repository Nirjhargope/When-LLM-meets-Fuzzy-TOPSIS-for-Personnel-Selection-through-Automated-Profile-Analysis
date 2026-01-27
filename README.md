# When-LLM-meets-Fuzzy-TOPSIS-for-Personnel-Selection-through-Automated-Profile-Analysis

🚀 When LLM Meets Fuzzy TOPSIS for Personnel Selection
📌 Overview
This repository presents a hybrid, end-to-end intelligent decision-support framework that integrates Large Language Models (LLMs) with Fuzzy TOPSIS. The system enables automated, fair, and explainable personnel selection by transforming unstructured candidate profiles into structured, quantitative decision inputs.


The core idea: Combine deep semantic understanding from LLMs with robust multi-criteria decision-making (MCDM) to support expert-level hiring at scale.

🎯 Key Contributions

Automated Feature Extraction: Utilizing Transformer-based LLMs to parse candidate profiles.


Semantic Scoring: Column-wise evaluation across About, Education, Experience, Skills, and Overall sections.


Uncertainty-Aware Decisions: Integration of Fuzzy TOPSIS to handle ambiguity in hiring.


Expert-Informed Rubric: Model comparisons guided by professional evaluation standards.


Visual Explainability: Structured plots and diagrams for transparent decision-making.

🧠 System Architecture (Visual Assets)
All high-level visual explanations are organized under the assets/ directory:


Attention Mechanism: assets/attention_mechanism/ — Overview of the Attention head.


Dataset Overview: assets/dataset_overview/ — Professional background distribution.


Exploratory Analysis: assets/exploratory_analysis/ — Correlation Heatmaps.


Model Internals: assets/model_internals/ — Top-level overview of model internals.


Decision Framework: assets/decision_framework/ — Flow from LLM to Fuzzy TOPSIS.

📂 Repository Structure
Plaintext
assets/          # Conceptual and analytical visualizations [cite: 62]
notebooks/       # Model-wise and column-wise experiments [cite: 63]
results/         # Metrics, plots, tables, and final comparisons [cite: 64]
src/             # Core source code for modeling [cite: 65]
fuzzy_topsis/    # Fuzzy TOPSIS implementation [cite: 66]
LICENSE          # MIT License [cite: 59]
README.md        # Documentation [cite: 60]
requirements.txt # Dependencies [cite: 61]
🔬 Models and Experiments
Experiments were conducted using multiple transformer architectures:


DistilRoBERTa-base 


RoBERTa-base 


LastBERT 

Each model is evaluated column-wise to ensure fine-grained semantic assessment of candidate data.

📊 Results & Evaluation
The results/ directory contains comprehensive performance data:


Accuracy vs Loss curves 


Confusion matrices (per column and per model) 


Final TOPSIS-based rankings 


Expert evaluation rubrics 

Final artifacts are consolidated in results/final_results/ for publication-ready summaries.

⚖️ Decision Making with Fuzzy TOPSIS
Fuzzy TOPSIS is employed to:


Handle ambiguity and uncertainty in candidate evaluations.


Integrate expert weights for different hiring criteria.


Produce interpretable rankings that are transparent and justifiable.

🛠️ Setup & Requirements
Install the necessary dependencies using pip:

Bash
pip install -r requirements.txt

Key Libraries: PyTorch, HuggingFace Transformers, Scikit-learn, and NumPy.

🚀 Use Cases
Automated recruitment systems.

Decision-support tools for HR analytics.

Research on LLM-driven MCDM frameworks.

Explainable AI (XAI) in personnel selection.

📜 License
This project is released under the MIT License.

✨ Final Note
This work bridges the gap between unstructured human data and formal evaluation frameworks, moving LLMs beyond simple prediction into structured decision-making.

Would you like me to help you generate a requirements.txt file or a specific LICENSE file for this repository?
