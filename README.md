# When-LLM-meets-Fuzzy-TOPSIS-for-Personnel-Selection-through-Automated-Profile-Analysis

Overview
This repository presents a hybrid, end-to-end intelligent decision-support framework that integrates Large Language Models (LLMs) with Fuzzy TOPSIS to enable automated, fair, and explainable personnel selection. The system transforms unstructured candidate profiles into structured, quantitative decision inputs and ranks candidates based on multiple evaluation criteria.
The core idea is simple yet powerful: combine deep semantic understanding from LLMs with a robust multi-criteria decision-making (MCDM) technique to support expert-level hiring decisions at scale.

 Key Contributions
Automated feature extraction from candidate profiles using Transformer-based LLMs
Column-wise semantic scoring (About, Education, Experience, Skills, Overall)
Integration of Fuzzy TOPSIS for uncertainty-aware decision-making
Expert-informed evaluation rubric and model comparison
Clear visual explainability via structured plots and diagrams

 System Architecture (Visual Assets)
All high-level visual explanations are organized under the assets/ directory:
Attention Mechanism
assets/attention_mechanism/Top-level overview of the Attention head.png
Dataset Overview
assets/dataset_overview/Professional background distribution of the candidates.png
Exploratory Analysis
assets/exploratory_analysis/Correlation Heatmap Encoded Features and Labels.png
Model Internals
assets/model_internals/Top-level overview of the Model’s internals.png
Decision Framework
assets/decision_framework/Top-level overview of the LLM-Fuzzy TOPSIS.png
These assets provide an intuitive understanding of how raw data flows through LLMs and ultimately feeds into the Fuzzy TOPSIS decision engine.

 Repository Structure
assets/                 # Conceptual and analytical visualizations
notebooks/              # Model-wise and column-wise experiments
results/                # Metrics, plots, tables, and final comparisons
src/                    # Core source code for modeling
fuzzy_topsis/           # Fuzzy TOPSIS implementation
LICENSE
README.md
requirements.txt


 Models and Experiments
Experiments are conducted using multiple transformer architectures:
DistilRoBERTa-base
RoBERTa-base
LastBERT
Each model is evaluated column-wise using dedicated notebooks (About, Education, Experience, Skills, Overall), ensuring fine-grained semantic assessment.

 Results & Evaluation
The results/ directory contains:
Accuracy vs Loss curves
Confusion matrices per column and per model
Final TOPSIS-based rankings
Expert evaluation rubric
Comparative performance tables and visual summaries
The final decision artifacts are consolidated under:
results/final_results/

providing a clear, publication-ready summary of model and TOPSIS outcomes.

 Decision Making with Fuzzy TOPSIS
Fuzzy TOPSIS is employed to:
Handle ambiguity and uncertainty in candidate evaluation
Integrate expert-defined criteria weights
Produce stable and interpretable candidate rankings
This ensures the framework is not only accurate but also transparent and justifiable in real-world hiring scenarios.

Setup & Requirements
Install dependencies using:
pip install -r requirements.txt

Key libraries include PyTorch, HuggingFace Transformers, Scikit-learn, and NumPy.

 Use Cases
Automated recruitment systems
Decision-support tools for HR analytics
Research on LLM-driven MCDM frameworks
Explainable AI in personnel selection

 License
This project is released under the MIT License, allowing flexible academic and industrial use with proper attribution.

Final Note
This work demonstrates how LLMs can move beyond prediction into structured decision-making, bridging the gap between unstructured human data and formal evaluation frameworks. We hope this repository inspires further research at the intersection of NLP, explainable AI, and decision science.

