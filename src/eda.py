import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(filepath):
    df = pd.read_csv(filepath)
    
    # Setam stilul graficelor
    sns.set_theme(style="whitegrid")
    
    # 1. Distributia target-ului (cati sunt bolnavi vs sanatosi)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='magma')
    plt.title('Distribuția Diagnosticului (0=Sănătos, 1=Bolnav)')
    plt.show()

    # 2. Corelatia între variabile (Heatmap)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matricea de Corelație a Factorilor Clinici')
    plt.show()

if __name__ == "__main__":
        run_eda('data/heart.csv')