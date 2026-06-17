from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_trained_models(X_train, y_train):
    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return lr, rf