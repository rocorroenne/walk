import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger les données
FILE_PATH = "data.xlsx"
df = pd.read_excel(FILE_PATH, sheet_name="data")

# Gérer les valeurs manquantes pour club_feet (au cas où)
df["club_feet"].fillna(df["club_feet"].mode()[0], inplace=True)

# Définition des variables explicatives et cible
X = df[["below_L2", "club_feet", "tOL", "gender"]]
y = df["walk_indep"]

# Entraîner le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Interface utilisateur Streamlit
st.title("Prediction of Independent Ambulation Following Prenatal Myelomeningocele Repair")
st.write("Enter the following information to estimate the probability (%) of independent walking at 30 months of age")

# Widgets pour entrer les variables
below_L2 = st.selectbox("Anatomical level of lesion below Lumbar two level", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
club_feet = st.selectbox("Presence of club feet", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
tOL = st.selectbox("Type of lesion", [1, 2], format_func=lambda x: "Flat lesion" if x == 1 else "Cystic lesion")
gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

# Bouton de prédiction
if st.button("Prediction"):
    input_data = pd.DataFrame([[below_L2, club_feet, tOL, gender]], columns=X.columns)
    prediction_proba = model.predict_proba(input_data)[0, 1]  # Probabilité de marcher indépendamment
    st.success(f"Prediction of independent ambulation at 30 months : {prediction_proba * 100:.1f}%")
