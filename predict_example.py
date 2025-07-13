import pandas as pd
from src.prediction import DiseasePredictor
import joblib

# Load preprocessor components
preprocessor_data = joblib.load("models/preprocessor.pkl")
label_encoder = preprocessor_data['label_encoder']
scaler = preprocessor_data['scaler']
feature_names = preprocessor_data['feature_columns']

# Example symptoms: match the feature_names order
symptoms_values = [1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
symptoms_df = pd.DataFrame([symptoms_values], columns=feature_names)

# Predict
predictor = DiseasePredictor('models/best_model.pkl', label_encoder, scaler)
disease, confidence = predictor.predict_disease(symptoms_df)

# Output
print("Predicted Disease:", disease)
print("Prediction Confidence:", f"{confidence:.2f}")