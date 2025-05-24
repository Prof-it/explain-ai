# Model Interpretability Demo

This repository contains a Jupyter notebook demonstrating the use of LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) for explaining AI model predictions. It also includes comparisons with inherently interpretable models like decision trees.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

## Contents

- `model_interpretability/model_explanations.ipynb`: Main notebook containing the demonstrations
  - LIME explanations
  - SHAP value analysis
  - Comparison with tree-based models
  - Visualizations and interpretations

## Note

This repository is part of a book project demonstrating practical applications of AI model interpretability techniques. 