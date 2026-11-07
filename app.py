import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prédiction de crédit", page_icon="🏦")


@st.cache_resource
def load_model():
    with open("model_GB.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("🏦 Prédiction de l'approbation de crédit")
st.write("Renseigne le profil du demandeur pour estimer si le prêt sera approuvé.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Sexe", ["Male", "Female"])
    married = st.selectbox("Marié(e)", ["Yes", "No"])
    dependents = st.selectbox("Personnes à charge", ["0", "1", "2", "3+"])
    education = st.selectbox("Niveau d'études", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Travailleur indépendant", ["No", "Yes"])
    credit_history = st.selectbox(
        "Historique de crédit conforme",
        [1.0, 0.0],
        format_func=lambda x: "Oui" if x == 1.0 else "Non",
    )

with col2:
    applicant_income = st.number_input("Revenu du demandeur", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Revenu du co-emprunteur", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Montant du prêt (en milliers)", min_value=1.0, value=120.0, step=1.0)
    loan_term = st.number_input("Durée du prêt (en mois)", min_value=1.0, value=360.0, step=12.0)
    property_area = st.selectbox("Zone du bien", ["Urban", "Semiurban", "Rural"])

if st.button("Prédire"):
    total_income = applicant_income + coapplicant_income

    # Mêmes transformations que dans le notebook
    row = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Total_Income_log": np.log(total_income + 1),
        "LoanAmount_log": np.log(loan_amount + 1),
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    }])

    # Remet les colonnes dans le même ordre que pendant l'entraînement
    row = row[model.feature_names_in_]

    proba = model.predict_proba(row)[0, 1]
    prediction = model.predict(row)[0]

    st.divider()
    if prediction == 1:
        st.success(f"Prêt **approuvé** (probabilité : {proba:.1%})")
    else:
        st.error(f"Prêt **refusé** (probabilité d'approbation : {proba:.1%})")

    st.caption("Estimation statistique issue d'un modèle de Machine Learning, qui ne remplace pas une décision bancaire.")