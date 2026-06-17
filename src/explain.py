import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, feature_names):
    print("\n[INFO] Se genereaza explicabilitatea modelului (SHAP)...")
    # Cream un explainer pentru Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Generam un grafic care arata importanta fiecarei trasaturi (varsta, colesterol etc.)
    plt.figure()
    shap.summary_plot(shap_values[1], X_train, feature_names=feature_names, show=False)
    plt.title("Importanta Factorilor Clinici (SHAP)")
    plt.show()