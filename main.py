from src.data_preprocessing import DataPreprocessor
from src.model_training import ImprovedModelTrainer
from src.evaluation import ModelEvaluator
import os
import joblib
import numpy as np

def main():
    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ImprovedModelTrainer()

    # Load raw data
    df = preprocessor.load_data('data/raw/disease_symptoms.csv')

    # Preprocess data (no manual undersampling now)
    processed = preprocessor.preprocess_data(df)

    X_train = processed['X_train']
    X_val = processed['X_val']
    X_test = processed['X_test']
    y_train = processed['y_train']
    y_val = processed['y_val']
    y_test = processed['y_test']

    # Save test data for later evaluation
    os.makedirs("data", exist_ok=True)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    # Apply SMOTE for balanced training
    X_train_balanced, y_train_balanced = trainer.apply_smote(X_train, y_train)

    # Apply feature selection (optional: pass num_features to select top K)
    X_train_selected = trainer.select_features(X_train_balanced, y_train_balanced)
    X_val_selected = trainer.selector.transform(X_val)
    X_test_selected = trainer.selector.transform(X_test)

    # Train ensemble and base models
    trainer.train_models(X_train_selected, y_train_balanced)

    # Evaluate on validation set
    results = trainer.evaluate_models(X_val_selected, y_val)

    # Save best model
    os.makedirs("models", exist_ok=True)
    best_model_name = trainer.save_best_model(results, 'models/best_model.pkl')

    # Save scaler, label encoder, and selector
    joblib.dump(preprocessor.scaler, 'models/scaler.pkl')
    joblib.dump(preprocessor.label_encoder, 'models/label_encoder.pkl')
    joblib.dump(trainer.selector, 'models/feature_selector.pkl')

    # Save preprocessor (recommended)
    preprocessor.save_preprocessor('models/preprocessor.pkl')

    print("\n🎉 Training complete!")
    print(f"✅ Best model: {best_model_name}")

if __name__ == "__main__":
    main()