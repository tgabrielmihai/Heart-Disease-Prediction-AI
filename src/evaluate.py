from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_and_plot(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    # Calculam metricile de baza în terminal
    acc = accuracy_score(y_test, y_pred)
    print(f"\nRezultate pentru {model_name}:")
    print(f"Acuratețe: {acc:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Matricea de confuzie
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds') 
    
    plt.title(f'Matrice de Confuzie - {model_name}')
    plt.ylabel('Valoare Reală')
    plt.xlabel('Predicție')
    plt.show()