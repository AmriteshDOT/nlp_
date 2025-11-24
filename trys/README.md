
# Article Grading Prediction

- This project builds an automated system to predict article quality scores (1–5) using NLP and transformer-based models.
- It aims to ensure consistent and objective grading across essays or written articles.


## Model Overview
- Baseline Models: TF-IDF with classical ML (XgBoost) and BiLSTM for sequential text modeling.
- Transformer Models: Fine-tuned DistilBERT and DistilRoBERTa, combined via ensembling.
## Training
- Used K-Fold Cross Validation for robustness.
- Optuna for hyperparameter tuning and threshold optimization for mapping predictions to discrete grades.
## Results
- Quadratic Weighted Kappa (QWK): 0.82
- Mean Absolute Error (MAE): 0.39


## Tech Stack
- Python · PyTorch · scikit-learn · Hugging Face Transformers · Optuna