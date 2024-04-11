import streamlit as st
import subprocess
import traceback

def run_script():
    try:
        subprocess.run(["python", "Calculate_flows_and_map.py"])
        st.success("Le script a été exécuté avec succès!")
    except Exception as e:
        st.error(f"Une erreur s'est produite: {e}")
        st.exception(traceback.format_exc())

def main():
    st.title("Application pour exécuter le script 'Calculate flows and map'")
    st.write("Cette application permet d'exécuter le script 'Calculate flows and map'.")

    # Bouton pour exécuter le script
    if st.button("Exécuter le script"):
        run_script()

if __name__ == "__main__":
    main()
