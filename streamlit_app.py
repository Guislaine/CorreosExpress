import streamlit as st
import subprocess

# Titre de l'application
st.title("Interface pour Calculate flows and map")

# Bouton pour lancer l'exécution du script
if st.button("Lancer l'exécution du script"):
    # Exécution du script
    subprocess.run(["python", "Calculate_flows_and_map.py"])
    # Message de confirmation
    st.write("L'exécution du script est terminée !")

