import sys
import os
sys.path.append(os.path.abspath('.'))

from src.data_preprocessing import DataPreprocessor

def test_preprocessing():
    """Test the preprocessing pipeline"""

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Path to raw data
    data_path = 'data/raw/disease_symptoms.csv'

    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        print("Please make sure you have 'disease_symptoms.csv' in data/raw/")
        return

    # Load and preprocess
    df = preprocessor.load_data(data_path)
    if df is None:
        return

    processed = preprocessor.preprocess_data(df, balance_method='undersample')

    # Save preprocessor and processed data
    os.makedirs('models', exist_ok=True)
    preprocessor.save_preprocessor('models/preprocessor.pkl')

    import joblib
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(processed, 'data/processed/processed_data.pkl')

    # Print final info
    print("\n✅ Preprocessing completed:")
    print(f"Train shape: {processed['X_train'].shape}")
    print(f"Validation shape: {processed['X_val'].shape}")
    print(f"Test shape: {processed['X_test'].shape}")
    print(f"Target classes: {processed['target_names']}")

if __name__ == "__main__":
    test_preprocessing()