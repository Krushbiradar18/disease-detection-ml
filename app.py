import streamlit as st
import pandas as pd
import joblib
from src.prediction import DiseasePredictor

# Load preprocessor which contains label_encoder and scaler
preprocessor = joblib.load('models/preprocessor.pkl')
label_encoder = preprocessor['label_encoder']
scaler = preprocessor['scaler']

# Load the trained model
model_path = 'models/best_model.pkl'
predictor = DiseasePredictor(model_path=model_path,
                             label_encoder=label_encoder,
                             scaler=scaler)

# --- UI Layout ---
st.set_page_config(page_title="Disease Detection", layout="centered")
st.title("🧠 Disease Detection System")
st.markdown("""  
Select your symptoms below and click **Predict Disease**.
""")
# Sidebar Branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2784/2784403.png", width=100)
    st.markdown("### 🤖 Built with ML & Streamlit")
    st.markdown("Made by **Krushnali Biradar**")
    st.sidebar.success("Model Accuracy (Validation): 67.92%")

# Define symptoms (match training features)
symptoms = [
    'fever', 'cough', 'headache', 'fatigue', 'nausea',
    'runny_nose', 'sore_throat', 'muscle_ache', 'chills', 'diarrhea'
]

st.subheader("📋 Select your symptoms:")
user_input = [int(st.checkbox(sym.capitalize().replace("_", " "), key=sym)) for sym in symptoms]

# Predict Button
if st.button("🔍 Predict Disease", key="predict_button"):
    selected_symptoms = [sym.replace('_', ' ').title() for sym, val in zip(symptoms, user_input) if val]

    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        st.markdown("📝 **Selected Symptoms:** " + ", ".join(selected_symptoms))
        
        # Create DataFrame
        input_df = pd.DataFrame([user_input], columns=symptoms)

        # Prediction
        disease, confidence = predictor.predict_disease(input_df)
        
        # Output
        st.success(f"🩺 Predicted Disease: **{disease}**")
        st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")