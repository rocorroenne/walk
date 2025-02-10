import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger les données
FILE_PATH = "data.xlsx"
df = pd.read_excel(FILE_PATH, sheet_name="data")

# Gérer les valeurs manquantes pour club_feet (au cas où)
df["club_feet"].fillna(df["club_feet"].mode()[0], inplace=True)

# Définition des variables explicatives
X = df[["below_L2", "club_feet", "tOL", "gender"]]

# Entraîner un modèle pour la marche indépendante (walk_indep)
y_indep = df["walk_indep"]
model_indep = RandomForestClassifier(n_estimators=100, random_state=42)
model_indep.fit(X, y_indep)

# Entraîner un modèle pour la marche assistée avec attelle (walk_brace)
y_brace = df["walk_brace"]
model_brace = RandomForestClassifier(n_estimators=100, random_state=42)
model_brace.fit(X, y_brace)

# Interface utilisateur Streamlit
st.title("Ambulation Prediction Following Prenatal Myelomeningocele Repair")

st.write("Enter the following information to estimate the probability (%) of independent or assisted walking at 30 months of age.")

# Widgets pour entrer les variables (identiques pour les deux calculateurs)
below_L2 = st.selectbox("Anatomical level of lesion below Lumbar two level", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
club_feet = st.selectbox("Presence of club feet", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
tOL = st.selectbox("Type of lesion", [1, 2], format_func=lambda x: "Flat lesion" if x == 1 else "Cystic lesion")
gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

# Création d'un DataFrame pour la prédiction
input_data = pd.DataFrame([[below_L2, club_feet, tOL, gender]], columns=X.columns)

# Premier calculateur : Prediction of Independent Walking
st.header("🦵 Prediction of Independent Walking")
if st.button("Predict Independent Walking"):
    prediction_proba_indep = model_indep.predict_proba(input_data)[0, 1]  # Probabilité de marche indépendante
    st.success(f"Probability of independent ambulation at 30 months: {prediction_proba_indep * 100:.1f}%")

# Séparation visuelle entre les deux calculateurs
st.markdown("---")

# Deuxième calculateur : Prediction of Assisted Walking (Brace)
st.header("🦿 Prediction of Independent Walking + Assisted Walking (Brace)")
if st.button("Predict Assisted Walking (Brace)"):
    prediction_proba_brace = model_brace.predict_proba(input_data)[0, 1]  # Probabilité de marche assistée avec attelle
    st.success(f"Probability of assisted ambulation (brace) at 30 months: {prediction_proba_brace * 100:.1f}%")
