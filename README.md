# ⚡ Enterprise ML Benchmarking & Inference Studio

An interactive, end-to-end Machine Learning web application built with Streamlit, Plotly, and Scikit-Learn. It enables automated dataset profiling, interactive EDA, automated preprocessing (Imputation, One-Hot Encoding, Scaling), multi-model benchmarking, interpretability (Feature Importance), and live single-instance inference.

## 📁 Repository Architecture

```text
PROJEXT/
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Automated Cleaning, Encoding & Scaling
│   └── model_trainer.py       # Cross-Validation & Metric Evaluation
├── app.py                     # Main Streamlit UI Engine
├── requirements.txt           # Project Dependencies
└── README.md