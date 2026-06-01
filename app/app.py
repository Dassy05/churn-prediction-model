import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, os

# --- Load Model & Preprocessor ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE, "models", "churn_model.pkl"))
preprocessor = joblib.load(os.path.join(BASE, "models", "preprocessor.joblib"))
NUM_FEATS = preprocessor.transformers[0][2]
CAT_FEATS = preprocessor.transformers[1][2]
ALL_FEAT_NAMES = list(NUM_FEATS) + list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATS)
)
explainer = shap.TreeExplainer(model)

# --- Page Config ---
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
        font-weight: 400;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero-card h3 { color: #667eea; margin-top: 0; }
    .hero-card p { color: #555; line-height: 1.7; }
    
    .stat-card {
        background: #f8f9fe;
        border: 1px solid #e8eaf6;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-card .stat-num {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.2rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b22, #ee545422);
        border: 1px solid #ff6b6b66;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-high h2 { color: #e53e3e; margin: 0; font-size: 1.5rem; }
    .risk-high p { color: #c53030; margin: 0.5rem 0 0 0; }
    
    .risk-low {
        background: linear-gradient(135deg, #48bb7822, #38a16922);
        border: 1px solid #48bb7866;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-low h2 { color: #2f855a; margin: 0; font-size: 1.5rem; }
    .risk-low p { color: #276749; margin: 0.5rem 0 0 0; }
    
    .reason-card {
        background: #f8f9fe;
        border-left: 4px solid #667eea;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .reason-card .reason-title {
        font-weight: 600;
        color: #333;
        font-size: 0.95rem;
    }
    .reason-card .reason-detail {
        color: #666;
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }
    
    .prob-display {
        text-align: center;
        padding: 1rem 0;
    }
    .prob-display .prob-num {
        font-size: 3.5rem;
        font-weight: 700;
    }
    .prob-display .prob-label {
        font-size: 0.9rem;
        color: #888;
    }
    
    .feature-section { margin-bottom: 0.5rem; }
    .feature-section label { font-weight: 500 !important; }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fe 0%, #eef0fb 100%);
    }
    div[data-testid="stSidebar"] h1 {
        font-size: 1.3rem;
        color: #667eea;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Predictor</h1>
    <p>Predict & Prevent Customer Loss — Powered by XGBoost & SHAP</p>
</div>
""", unsafe_allow_html=True)

# --- Navigation ---
tab_home, tab_predict = st.tabs(["🏠 Home", "🔮 Predict"])

# ================== HOME TAB ==================
with tab_home:
    st.markdown("""
    <div class="hero-card">
        <h3>🚀 What Does This Project Do?</h3>
        <p>
            This is an end-to-end <b>Customer Churn Prediction</b> system built on the 
            <b>IBM Telco Customer Churn</b> dataset. It uses machine learning to identify 
            customers who are likely to stop using a telecom company's services.
        </p>
        <p>
            Churn costs telecom companies billions annually. By predicting which customers are 
            at risk <i>before</i> they leave, businesses can launch targeted retention campaigns 
            — such as personalized discounts, contract upgrades, or proactive support calls — 
            and significantly reduce customer loss.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="stat-card"><div class="stat-num">7,043</div><div class="stat-label">Customers Analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><div class="stat-num">40</div><div class="stat-label">Engineered Features</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><div class="stat-num">82.6%</div><div class="stat-label">ROC-AUC Score</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><div class="stat-num">XGBoost</div><div class="stat-label">Tuned Model</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline overview
    st.markdown("### 🔧 How It Works")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**1️⃣ Data Cleaning**")
        st.caption("Handle missing values, fix data types, and drop irrelevant columns like customerID.")
    with p2:
        st.markdown("**2️⃣ Feature Engineering**")
        st.caption("Encode categorical features, scale numerics with StandardScaler, and apply One-Hot Encoding.")
    with p3:
        st.markdown("**3️⃣ Model Training**")
        st.caption("Train XGBoost with GridSearchCV (5-fold CV), apply SMOTE to handle class imbalance.")
    with p4:
        st.markdown("**4️⃣ Explainability**")
        st.caption("Use SHAP values to explain every prediction and surface the top reasons driving churn risk.")

    st.markdown("---")

    # Key insights
    st.markdown("### 💡 Key Business Insights")
    st.info("📌 **Month-to-month contracts** are the #1 predictor of churn. Customers without long-term commitments are far more likely to leave.")
    st.info("📌 **New customers (tenure < 6 months)** churn at 3x the rate of long-term customers. Onboarding support is critical.")
    st.info("📌 Customers paying via **Electronic check** and those **without Tech Support or Online Security** addons are significantly more likely to churn.")

# ================== PREDICT TAB ==================
with tab_predict:
    # --- Sidebar Inputs ---
    st.sidebar.markdown("# 🧑‍💼 Customer Profile")
    st.sidebar.markdown("---")
    yn = ["No", "Yes"]

    st.sidebar.markdown("**📅 Account Details**")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, 0.5)
    total = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly), 10.0)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    paperless = st.sidebar.selectbox("Paperless Billing", yn)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**👤 Demographics**")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: yn[x])
    partner = st.sidebar.selectbox("Partner", yn)
    dependents = st.sidebar.selectbox("Dependents", yn)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📞 Services**")
    phone = st.sidebar.selectbox("Phone Service", yn)
    multi = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

    # --- Prediction Logic ---
    if predict_btn:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "tenure": tenure,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        X = pd.DataFrame(preprocessor.transform(row), columns=ALL_FEAT_NAMES)
        prob = float(model.predict_proba(X)[0][1])  # cast to native Python float

        # --- Result Layout ---
        col_result, col_reasons = st.columns([1, 1], gap="large")

        with col_result:
            # Probability display
            prob_color = "#e53e3e" if prob > 0.5 else "#2f855a"
            st.markdown(f"""
            <div class="prob-display">
                <div class="prob-label">CHURN PROBABILITY</div>
                <div class="prob-num" style="color: {prob_color}">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(prob)

            # Risk card
            if prob > 0.5:
                st.markdown(f"""
                <div class="risk-high">
                    <h2>🔴 HIGH CHURN RISK</h2>
                    <p>This customer is likely to leave. Consider offering a contract upgrade, 
                    loyalty discount, or proactive outreach from the Customer Success team.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>🟢 LOW CHURN RISK</h2>
                    <p>This customer shows strong loyalty signals. Focus on cross-selling 
                    additional services and maintaining engagement.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_reasons:
            st.markdown("#### 🧠 Top 3 Reasons for This Prediction")
            st.caption("Driven by SHAP (SHapley Additive exPlanations)")
            st.markdown("")

            sv = explainer(X)
            impacts = pd.Series(sv.values[0], index=ALL_FEAT_NAMES)
            top3 = impacts.abs().nlargest(3).index

            # Friendly feature name mapping
            friendly_names = {
                "tenure": "Tenure (months)",
                "MonthlyCharges": "Monthly Charges",
                "TotalCharges": "Total Charges",
                "Contract_Month-to-month": "Month-to-Month Contract",
                "Contract_One year": "One Year Contract",
                "Contract_Two year": "Two Year Contract",
                "OnlineSecurity_No": "No Online Security",
                "OnlineSecurity_Yes": "Has Online Security",
                "TechSupport_No": "No Tech Support",
                "TechSupport_Yes": "Has Tech Support",
                "InternetService_Fiber optic": "Fiber Optic Internet",
                "InternetService_DSL": "DSL Internet",
                "PaperlessBilling_Yes": "Paperless Billing",
                "PaymentMethod_Electronic check": "Electronic Check Payment",
                "PaymentMethod_Mailed check": "Mailed Check Payment",
            }

            icons = ["1️⃣", "2️⃣", "3️⃣"]
            for i, feat in enumerate(top3):
                val = float(impacts[feat])
                direction = "↑ Increases" if val > 0 else "↓ Decreases"
                dir_color = "#e53e3e" if val > 0 else "#2f855a"
                display_name = friendly_names.get(feat, feat)
                st.markdown(f"""
                <div class="reason-card">
                    <div class="reason-title">{icons[i]} {display_name}</div>
                    <div class="reason-detail">
                        SHAP value: <b>{val:+.3f}</b> — 
                        <span style="color:{dir_color};font-weight:600">{direction} churn risk</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("")
        st.markdown("")
        col_empty = st.columns([1, 2, 1])[1]
        with col_empty:
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color: #aaa;">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">🔮</p>
                <p style="font-size: 1.1rem;">Fill in the <b>Customer Profile</b> on the sidebar<br>and click <b>Predict Churn Risk</b> to get started.</p>
            </div>
            """, unsafe_allow_html=True)
