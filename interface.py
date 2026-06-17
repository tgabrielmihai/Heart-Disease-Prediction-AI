import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# 1. Configurare pagina si stilizare (CSS)

st.set_page_config(
    page_title="Diagnostic AI",
    page_icon="🩺", 
    layout="centered"
)

st.markdown("""
    <style>
    /* Fundal*/
    .stApp { background-color: #fff5f5 !important; }
    
    /* Titlu principal - Roșu */
    .big-title { 
        font-size: 2.1rem; 
        font-weight: 700; 
        color: #ff4b4b; 
        margin-bottom: 0.2rem; 
    }
    
    .subtitle { font-size: 0.95rem; color: #4a4a4a; margin-bottom: 0.8rem; }
    
    /* Stilizare Tab-uri */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #ff4b4b;
        border: 1px solid #fee2e2;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ff4b4b !important; 
        color: white !important; 
    }

    /* Titluri secțiuni în interiorul tab-urilor */
    .section-title { 
        font-size: 1.05rem; 
        font-weight: 600; 
        color: #ff4b4b; 
        margin-top: 0.5rem; 
        margin-bottom: 0.8rem; 
    }
    
    .section-number { 
        display: inline-block; 
        background-color: #ff4b4b; 
        color: white; 
        border-radius: 999px; 
        padding: 0.1rem 0.6rem; 
        font-size: 0.8rem; 
        margin-right: 0.4rem; 
    }
    
    /* Cardul de rezultat */
    .result-card { 
        background-color: #ffffff; 
        border-radius: 14px; 
        padding: 18px 22px; 
        box-shadow: 0 10px 20px rgba(255, 75, 75, 0.10); 
        margin-top: 18px; 
        border-left: 4px solid #ff4b4b; 
    }
    
    /* Butonul principal roșu */
    .stButton button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 8px !important;
        width: 100%;
        height: 3rem;
        font-weight: 600;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Incarcare model si resurse

@st.cache_resource
def load_resources():
    try:
        model = joblib.load("models/best_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        features = joblib.load("models/feature_names.joblib")
        return model, scaler, features
    except:
        return None, None, None

model, scaler, feature_names = load_resources()

if model is None:
    st.error("❌ Resursele nu au fost găsite! Rulează mai întâi 'python main.py'.")
    st.stop()

# 3. Header

st.markdown("""
    <div>
        <div class="big-title">❤️ – Diagnostic cardiac asistat de AI</div>
        <p class="subtitle">
            Completează informațiile de mai jos pentru o estimare orientativă realizată de Inteligența Artificială.
            Rezultatele <strong>nu înlocuiesc un consult medical</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")


# 4. Organizare pe tab-uri 

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Despre tine", 
    "2. Tensiune & Colesterol", 
    "3. EKG & Efort", 
    "4. Alte informații"
])

with tab1:
    st.markdown('<div class="section-title"><span class="section-number">1</span>Despre tine</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Vârsta (ani)", 18, 100, 50)
    with c2:
        sex_disp = st.radio("Sex biologic", ["Femeie", "Bărbat"], horizontal=True)
    sex_val = 1 if sex_disp == "Bărbat" else 0

with tab2:
    st.markdown('<div class="section-title"><span class="section-number">2</span>Tensiune, colesterol, glicemie</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        trestbps = st.number_input("Tensiunea arterială în repaus (mmHg)", 80, 220, 120)
        chol = st.number_input("Colesterol total (mg/dl)", 100, 600, 230)
    with c2:
        fbs_disp = st.radio("Glicemia peste 120 mg/dl?", ["Nu", "Da"], horizontal=True)
        fbs = 1 if fbs_disp == "Da" else 0
        cp_disp = st.selectbox("Ai dureri în piept?", ["Fără dureri (0)", "Angină tipică (1)", "Angină atipică (2)", "Durere non-anginală (3)"])
    cp = int(cp_disp.split('(')[1].split(')')[0])

with tab3:
    st.markdown('<div class="section-title"><span class="section-number">3</span>Informații din EKG / test de efort</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        restecg_disp = st.selectbox("Rezultat EKG repaus", ["Normal (0)", "Anomalii ST-T (1)", "Hipertrofie (2)"])
        restecg = int(restecg_disp.split('(')[1].split(')')[0])
        thalach = st.number_input("Puls maxim atins (thalach)", 60, 220, 150)
    with c2:
        exang_disp = st.radio("Durere la efort? (exang)", ["Nu", "Da"], horizontal=True)
        exang = 1 if exang_disp == "Da" else 0
        oldpeak = st.number_input("Depresie ST (oldpeak)", 0.0, 10.0, 1.0, step=0.1)

with tab4:
    st.markdown('<div class="section-title"><span class="section-number">4</span>Alte informații tehnice</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        slope = st.selectbox("Panta ST (slope)", [0, 1, 2])
    with c2:
        ca = st.selectbox("Vase majore (ca)", [0, 1, 2, 3])
    with c3:
        thal = st.selectbox("Thalium (thal)", [1, 2, 3])

# 5. Buton si calcul rezultat

st.divider()
if st.button("🔍 CALCULEAZĂ RISCUL ESTIMAT"):
    input_data = pd.DataFrame([[age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=feature_names)
    X_scaled = scaler.transform(input_data)
    
    proba = model.predict_proba(X_scaled)[0, 1]
    
    st.markdown(f"""
        <div class="result-card">
            <div style="font-size: 1.1rem; font-weight: 600; color: #ff4b4b;">Rezultat estimat:</div>
            <div style="font-size: 1.5rem; margin: 10px 0;">
                Probabilitate: <strong>{proba:.1%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP FIXAT
    st.markdown('<div class="section-title">🔎 Ce a influențat acest rezultat?</div>', unsafe_allow_html=True)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled, check_additivity=False)
        influenta = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0,:,1] if len(shap_values.shape)==3 else shap_values[0])
        
        friendly_names = {"age": "Vârsta", "sex": "Sex", "cp": "Tip durere piept", "trestbps": "Tensiune", "chol": "Colesterol", "fbs": "Glicemie", "restecg": "EKG repaus", "thalach": "Puls maxim", "exang": "Durere efort", "oldpeak": "Depresie ST", "slope": "Panta ST", "ca": "Vase majore", "thal": "Thalium"}
        
        impact_df = pd.DataFrame({
            'Indicator': [friendly_names.get(col, col) for col in feature_names],
            'Influență': influenta
        })
        st.table(impact_df.sort_values(by='Influență', ascending=False, key=abs).head(5))
    except:
        st.info("Analiza factorilor se încarcă...")