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

# --- Sidebar: about the model ---
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
    c1.metric("Accuracy", "74.0%")
    c2.metric("ROC-AUC", "0.818")
    c1.metric("Recall", "75.9%")
    c2.metric("Precision", "60.3%")
    st.caption("Evaluated on 154 patients the model never saw during training.")

# --- Title and description ---
st.title("🩺 Diabetes Risk Predictor")
st.write(
    "This app uses a machine learning classification model trained on the "
    "Pima Indians Diabetes dataset to estimate whether a patient is likely "
    "to have diabetes."
)

# --- Instructions ---
with st.expander("How to use this app", expanded=True):
    st.markdown(
        "1. Enter the patient's details in the form below.\n"
        "2. If **Skin Thickness** or **Insulin** were not measured, leave them at **0**. "
        "The model will fill in a typical value.\n"
        "3. Click **Predict** to see the result."
    )


# --- User inputs ---
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
    errors = []
    if pregnancies > age - 12:
        errors.append(f"{pregnancies} pregnancies is not realistic for a {age}-year-old patient.")
    if skin_thickness == 0 and insulin > 0:
        st.info("Skin Thickness is 0, so the model will estimate it.")
    if insulin == 0:
        st.info("Insulin is 0 (not measured), so the model will estimate it.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()   # don't predict on invalid input

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

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"**Diabetes likely**: estimated probability {probability:.0%}")
    else:
        st.success(f"**Diabetes unlikely**: estimated probability {probability:.0%}")
    st.progress(float(probability), text=f"Probability of diabetes: {probability:.1%}")

    with st.expander("Values sent to the model"):
        st.dataframe(patient, hide_index=True)


# --- Disclaimer ---
st.divider()
st.caption("For educational purposes only. This is not a medical diagnosis. "
           "Consult a healthcare professional for medical advice.")