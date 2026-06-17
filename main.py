import joblib
import os
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.models import get_trained_models
from src.evaluate import evaluate_and_plot  

def main():
    # Cream folderul pentru modele daca nu exista
    if not os.path.exists('models'): 
        os.makedirs('models')
    
    # 1. Procesare date
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/heart.csv')
    
    # 2. Antrenare
    # Se antreneaza modelele si retinem varianta Random Forest (rf_model)
    _, rf_model = get_trained_models(X_train, y_train)
    
    # --- GENERARE GRAFIC  ---
    evaluate_and_plot(rf_model, X_test, y_test, "Random Forest")
    
    # 3. Salvare resurse
    joblib.dump(rf_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    # Salvam numele coloanelor pentru SHAP din interfață
    feature_names = pd.read_csv('data/heart.csv').drop('target', axis=1).columns.tolist()
    joblib.dump(feature_names, "models/feature_names.joblib")
    
    print("Modelul a fost salvat cu succes în folderul models/!")

if __name__ == "__main__":
    main()