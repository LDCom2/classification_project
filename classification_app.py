import joblib
import pandas as pd
import streamlit as st

# --- Page settings (must be the first Streamlit command) ---
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# --- Load the trained model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    return joblib.load("classification_model.joblib")

model = load_model()

# --- Sidebar: about the model + threshold ---
with st.sidebar:
    st.header("About the Model")
    st.markdown(
        "- **Dataset:** Pima Indians Diabetes (768 patients)\n"
        "- **Model:** Random Forest (300 trees, max depth 6)\n"
        "- **Preprocessing:** median imputation of missing values + standard scaling\n"
        "- **Split:** 80% training / 20% testing"
    )
    st.subheader("Test Set Performance")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", "73.4%")
    c2.metric("ROC-AUC", "0.818")
    c1.metric("Recall", "83.3%")
    c2.metric("Precision", "58.4%")
    st.caption("Evaluated on 154 patients the model never saw during training.")

    st.divider()
    st.subheader("Decision Threshold")
    threshold = st.slider(
        "Predict 'Yes' when probability is at least:",
        min_value=0.30, max_value=0.70, value=0.45, step=0.05,
        help="Lower = catches more diabetic patients but more false alarms. "
             "Higher = fewer false alarms but misses more cases. "
             "0.45 was chosen using cross-validation.",
    )
    st.caption("Performance metrics above are for the default threshold of 0.45.")

# --- Title and description ---
st.title("🩺 Diabetes Risk Predictor")
st.write(
    "This app uses a machine learning classification model trained on the "
    "Pima Indians Diabetes dataset to predict whether a patient is likely "
    "to have diabetes."
)

# --- Instructions ---
with st.expander("How to use this app", expanded=True):
    st.markdown(
        "1. Use the sliders below to enter the patient's details.\n"
        "2. If **Skin Thickness** or **Insulin** were not measured, leave them at **0**. "
        "The model will estimate them.\n"
        "3. Click **Predict** to get a **Yes / No** result.\n"
        "4. Optional: adjust the **decision threshold** in the sidebar."
    )

# --- User inputs (slider limits match the ranges in the training data) ---
with st.form("patient_form"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 1,
                                help="Number of times pregnant")
        glucose = st.slider("Glucose (mg/dL)", 40, 200, 120,
                            help="Plasma glucose 2 hours into an oral glucose tolerance test")
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 20, 130, 72,
                                   help="Diastolic blood pressure")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 0,
                                   help="Triceps skinfold thickness. Leave at 0 if not measured")

    with col2:
        insulin = st.slider("Insulin (µU/mL)", 0, 850, 0,
                            help="2-hour serum insulin. Leave at 0 if not measured")
        bmi = st.slider("BMI (kg/m²)", 15.0, 70.0, 28.0, step=0.1,
                        help="Body mass index: weight (kg) / height (m)²")
        dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.50, 0.40, step=0.01,
                        help="Score of diabetes likelihood based on family history")
        age = st.slider("Age (years)", 21, 81, 33,
                        help="The dataset covers patients aged 21 to 81")

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# --- Validate inputs, then predict ---
if submitted:
    # Validation: block unrealistic combinations
    if pregnancies > age - 12:
        st.error(f"{pregnancies} pregnancies is not realistic for a {age}-year-old patient. "
                 "Please check the inputs.")
        st.stop()

    # Let the user know when values will be estimated
    if skin_thickness == 0:
        st.info("Skin Thickness is 0 (not measured), so the model will estimate it.")
    if insulin == 0:
        st.info("Insulin is 0 (not measured), so the model will estimate it.")

    # Same column names as in training
    patient = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }])

    # Prediction using the chosen threshold
    probability = model.predict_proba(patient)[0][1]
    prediction = "Yes" if probability >= threshold else "No"

    st.subheader("Result")
    if prediction == "Yes":
        st.error("## Diabetes: **Yes**")
    else:
        st.success("## Diabetes: **No**")

    st.write(f"Estimated probability of diabetes: **{probability:.1%}** "
             f"(threshold: {threshold:.2f})")
    st.progress(float(probability))

    with st.expander("Values sent to the model"):
        st.dataframe(patient, hide_index=True)

# --- Disclaimer ---
st.divider()
st.caption("For educational purposes only. This is not a medical diagnosis. "
           "Consult a healthcare professional for medical advice.")
