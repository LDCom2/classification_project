import pandas as pd

# Load the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, header=None, names=cols)

# Explore the data
print(df.shape)
print(df.head())
print(df["Outcome"].value_counts())


from sklearn.model_selection import train_test_split

# --- Check for hidden missing values ---
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nZeros (missing values) per column:")
print((df[zero_cols] == 0).sum())

# --- Separate features (X) from the target (y) ---
X = df.drop(columns="Outcome")
y = df["Outcome"]

# --- Split into 80% training, 20% testing ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Columns where 0 is a real value, so we leave them as they are
other_cols = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

# --- Preprocessing: replace 0s with the median in the "missing" columns ---
preprocess = ColumnTransformer([
    ("fill_zeros", SimpleImputer(missing_values=0, strategy="median"), zero_cols),
    ("keep", "passthrough", other_cols),
])

# --- Full pipeline: preprocess -> scale -> classify ---
model = Pipeline([
    ("preprocess", preprocess),
    ("scale", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=6,          
        class_weight="balanced",
        random_state=42,
    )),
])

print("\nPipeline built:")
print(model)

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# --- Train the model on the training set only ---
model.fit(X_train, y_train)

# --- Predict on the test set (data the model has never seen) ---
y_pred = model.predict(X_test) 
y_prob = model.predict_proba(X_test)[:, 1]

# --- Evaluate ---
print("\n=== Test Set Evaluation ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.3f}")

print("\nConfusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, y_pred))

print("\n" + classification_report(y_test, y_pred,
                                    target_names=["No Diabetes", "Diabetes"]))



import joblib

# --- Save the trained pipeline ---
joblib.dump(model, "classification_model.joblib")
print("\nModel saved to classification_model.joblib")

# --- Check it loads and predicts correctly ---
loaded = joblib.load("classification_model.joblib")
sample = X_test.iloc[[0]]          # one patient from the test set
print("Sample patient:\n", sample)
print("Prediction:", loaded.predict(sample)[0])
print("Probability of diabetes:", round(loaded.predict_proba(sample)[0][1], 3))