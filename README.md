# Diabetes Risk Predictor: Classification App

Machine Learning Classification Application deployed using Streamlit Community Cloud.

**Live app:** https://classificationproject-dhzpa7k3wyuevcswgcsuc7.streamlit.app/

## Overview
This app predicts whether a patient is likely to have diabetes from eight diagnostic
measurements. Users enter patient details with sliders and receive a prediction with
an estimated probability.

## Dataset
**Pima Indians Diabetes**: 768 female patients of Pima Indian heritage, aged 21+.
- 8 features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
  Diabetes Pedigree Function, Age
- Target: Outcome (1 = diabetes, 0 = no diabetes)

## Preprocessing
- Zeros in Glucose, Blood Pressure, Skin Thickness, Insulin and BMI represent missing
  values and are replaced with the training-set median.
- Features are standardised with StandardScaler.
- Data is split 80/20 (stratified) into training and test sets.

## Model
Random Forest Classifier (300 trees, max depth 6, balanced class weights), built as a
scikit-learn Pipeline and saved with joblib as `classification_model.joblib`.

## Results (test set, 154 patients)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.740 |
| Precision | 0.603 |
| Recall    | 0.759 |
| F1-score  | 0.672 |
| ROC-AUC   | 0.818 |

## Input Validation
- Sliders restrict every input to the range seen in the training data.
- The app rejects unrealistic combinations (e.g. too many pregnancies for the patient's age).

## Repository Structure
```
classification-project/
├── classification_app.py
├── classification_model.joblib
├── requirements.txt
├── train_model.py
└── README.md
```

## Run Locally
```
pip install -r requirements.txt
streamlit run classification_app.py
```

Built with Python 3.12 and scikit-learn 1.9.1.

*For educational purposes only. Not a medical diagnosis.*
