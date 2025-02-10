import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger les données
FILE_PATH = "data.xlsx"
df = pd.read_excel(FILE_PATH, sheet_name="data")

# Gérer les valeurs manquantes
df["mf"].fillna(df["mf"].mode()[0], inplace=True)

# Définition des variables explicatives et cible
X = df[["below_L2", "mf", "tOL"]]
y = df["walk_indep"]

# Entraîner le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Interface utilisateur Streamlit
st.title("Prédiction de la Marche Indépendante")
st.write("Entrez les paramètres pour obtenir une probabilité (%)")

# Widgets pour entrer les variables
below_L2 = st.selectbox("Lésion en dessous de L2 ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
mf = st.selectbox("mf (1 = Oui, 0 = Non)", [0, 1])
tOL = st.selectbox("tOL (1 ou 2)", [1, 2])

# Bouton de prédiction
if st.button("Prédire"):
    input_data = pd.DataFrame([[below_L2, mf, tOL]], columns=X.columns)
    prediction_proba = model.predict_proba(input_data)[0, 1]  # Probabilité de marcher indépendamment
    st.success(f"Probabilité de marcher indépendamment : {prediction_proba * 100:.1f}%")
