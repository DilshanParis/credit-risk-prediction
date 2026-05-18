
# Credit Risk Prediction System

## Live Demo
[Try the app here](https://credit-risk-prediction-27468zx82zfalfio7rufcf.streamlit.app/)

## What It Does
A web app that predicts loan applicant credit risk using a machine learning
model trained on the German Credit Risk dataset (1,000 applicants, 20 features).

## Stack
Python · scikit-learn · Streamlit · pandas · imbalanced-learn

## Key Techniques
- scikit-learn Pipelines + ColumnTransformer
- SMOTE for class imbalance (training data only)
- GridSearchCV with 5-fold cross-validation
- 4-model comparison with ROC-AUC based selection

## Run Locally
```bash
git clone https://github.com/DilshanParis/credit-risk-prediction
cd credit-risk-prediction
pip install -r requirements.txt
streamlit run app.py
```

[![Open App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-prediction-27468zx82zfalfio7rufcf.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dDevv2oipLv00hA56Yg20IBYZosbNKJk?usp=sharing)