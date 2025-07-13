import joblib
import numpy as np

class DiseasePredictor:
    def __init__(self, model_path, label_encoder, scaler):
        self.model = joblib.load(model_path)
        self.label_encoder = label_encoder
        self.scaler = scaler

    def predict_disease(self, symptoms_df):
        """Predict disease from a DataFrame of symptoms"""
        symptoms_scaled = self.scaler.transform(symptoms_df)
        prediction = self.model.predict(symptoms_scaled)[0]
        disease_name = self.label_encoder.inverse_transform([prediction])[0]

        # Confidence score
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(symptoms_scaled)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0

        return disease_name, confidence