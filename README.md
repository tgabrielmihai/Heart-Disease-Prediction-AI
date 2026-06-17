# Sistem AI pentru Predictia Riscului Cardiovascular

Acest proiect reprezinta o solutie software modulara bazata pe Inteligenta Artificiala, conceputa pentru a asista personalul medical in evaluarea riscului de boli de inima.

## 🧠 Detalii AI
- **Model:** Random Forest Classifier (Ansamblu de 100 de arbori de decizie).
- **Dataset:** Heart Disease Dataset (UCI Cleveland) - 303 esantioane, 14 atribute.
- **Tehnici folosite:** Standardizare (StandardScaler), Stratificare (Train-Test Split), Explicabilitate (SHAP).
- **Performanta:** Acuratete ridicata si Recall de 1.0 (esential pentru evitarea falsilor negativi in medicina).

## 📂 Structura Proiectului
- `main.py`: Scriptul principal care coordoneaza incarcarea datelor, antrenarea si salvarea modelului.
- `src/`: Contine modulele de logica:
    - `preprocess.py`: Curatarea si scalarea datelor.
    - `models.py`: Arhitectura algoritmilor de Machine Learning.
    - `evaluate.py`: Calcularea metricilor de performanta si a matricii de confuzie.
    - `eda.py`: Analiza exploratorie a datelor (corelatii, distributii).
    - `explain.py`: Generarea valorilor SHAP pentru transparenta decizionala.
- `interface.py`: Aplicatia web interactiva realizata in Streamlit.
- `data/`: Contine fisierul sursa `heart.csv`.
- `models/`: Folderul unde sunt salvate modelele antrenate in format `.joblib`.

## 🚀 Instructiuni de Instalare si Rulare

1. **Instalarea dependintelor:**
   ```bash
   pip install -r requirements.txt