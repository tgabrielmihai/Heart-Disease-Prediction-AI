import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Incarcam datele
    df = pd.read_csv(filepath)
    
    # Separam variabilele de intrare (X) de rezultatul dorit (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Impartim datele: 80% pentru antrenare, 20% pentru testare
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardizam datele (aducem toate valorile la aceeasi scara)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler