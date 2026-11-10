# Diabetes Risk Predictor: Classification App

Machine Learning Classification Application deployed using Streamlit Community Cloud.

**Live app:** https://classificationproject-dhzpa7k3wyuevcswgcsuc7.streamlit.app/

## Overview
This app predicts whether a patient is likely to have diabetes (**Yes / No**) from eight
diagnostic measurements. Users enter patient details with sliders and receive a
prediction along with the estimated probability.

## Dataset
**Pima Indians Diabetes**: 768 female patients of Pima Indian heritage, aged 21+.
- 8 features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
  Diabetes Pedigree Function, Age
- Target: Outcome (1 = diabetes, 0 = no diabetes)
- Class balance: 500 non-diabetic, 268 diabetic

## Preprocessing
- Zeros in Glucose, Blood Pressure, Skin Thickness, Insulin and BMI represent missing
  values and are replaced with the training-set median.
- Features are standardised with StandardScaler.
- Data is split 80/20 (stratified) into training and test sets.
- All preprocessing is inside a scikit-learn Pipeline, so the app passes raw inputs
  directly to the model.

## Model
Random Forest Classifier (300 trees, max depth 6, balanced class weights), saved with
joblib as `classification_model.joblib`.

## Threshold Selection
The decision threshold was tuned using 5-fold cross-validated probabilities on the
training set, so the test set was kept untouched for the final evaluation.

| Threshold | Accuracy | Precision | Recall | F1    |
|-----------|----------|-----------|--------|-------|
| 0.40      | 0.728    | 0.576     | 0.832  | 0.681 |
| **0.45**  | 0.749    | 0.607     | 0.794  | 0.688 |
| 0.50      | 0.759    | 0.626     | 0.766  | 0.689 |
| 0.55      | 0.764    | 0.648     | 0.706  | 0.676 |
| 0.60      | 0.772    | 0.687     | 0.636  | 0.660 |

A threshold of **0.45** was chosen. It matches the best F1-score while catching more
diabetic patients, which is the priority for a screening tool: a missed diagnosis
is more costly than a false alarm.

## Results (test set, 154 patients)

| Metric    | Threshold 0.50 | Threshold 0.45 (used) |
|-----------|----------------|-----------------------|
| Accuracy  | 0.740          | 0.734                 |
| Precision | 0.603          | 0.584                 |
| Recall    | 0.759          | **0.833**             |
| F1-score  | 0.672          | **0.687**             |
| ROC-AUC   | 0.818          | 0.818                 |

**Confusion matrix at threshold 0.45:**

|                  | Predicted No | Predicted Yes |
|------------------|--------------|---------------|
| **Actual No**    | 68           | 32            |
| **Actual Yes**   | 9            | 45            |

At 0.45 the model correctly identifies 45 of 54 diabetic patients (up from 41 at 0.50).

## App Features
- **Yes / No prediction** with the estimated probability
- **Sliders** that restrict each input to the range seen in the training data
- **Input validation** that rejects unrealistic combinations (e.g. too many
  pregnancies for the patient's age)
- **Adjustable decision threshold** in the sidebar (default 0.45)
- Sidebar summary of the model and its test-set performance

## Repository Structure
```
classification-project/
├── classification_app.py        # Streamlit app
├── classification_model.joblib  # trained model pipeline
├── requirements.txt             # dependencies
├── train_model.py               # training, evaluation and threshold tuning
└── README.md
```

## Run Locally
```
pip install -r requirements.txt
streamlit run classification_app.py
```

To retrain the model: `python train_model.py`

Built with Python 3.12 and scikit-learn 1.9.1.

*For educational purposes only. Not a medical diagnosis.*
