import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'disease'
        self.is_fitted = False
    
    def load_data(self, file_path):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully: {df.shape}")
            print(f"✓ Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """Explore the dataset and print basic statistics"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Number of features: {len(df.columns) - 1}")
        print(f"Number of samples: {len(df)}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n⚠ Missing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("✓ No missing values found")
        
        # Disease distribution
        print(f"\nDisease distribution:")
        disease_counts = df[self.target_column].value_counts()
        for disease, count in disease_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {disease}: {count} samples ({percentage:.1f}%)")
        
        # Check data balance
        min_samples = disease_counts.min()
        max_samples = disease_counts.max()
        balance_ratio = min_samples / max_samples
        print(f"\nData balance ratio: {balance_ratio:.3f}")
        if balance_ratio < 0.8:
            print("⚠ Warning: Dataset is imbalanced. Consider balancing.")
        else:
            print("✓ Dataset is reasonably balanced")
        
        # Feature statistics
        feature_cols = [col for col in df.columns if col != self.target_column]
        print(f"\nFeature statistics:")
        print(f"  Binary features: {len(feature_cols)}")
        
        # Check if all features are binary
        non_binary_features = []
        for col in feature_cols:
            unique_values = df[col].unique()
            if not set(unique_values).issubset({0, 1}):
                non_binary_features.append(col)
        
        if non_binary_features:
            print(f"⚠ Non-binary features found: {non_binary_features}")
        else:
            print("✓ All features are binary")
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\n" + "="*50)
        print("HANDLING MISSING VALUES")
        print("="*50)
        
        # Check for missing values
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"Found {missing_before} missing values")
            
            # For symptom columns, fill with 0 (no symptom)
            feature_cols = [col for col in df.columns if col != self.target_column]
            df[feature_cols] = df[feature_cols].fillna(0)
            
            # For disease column, drop rows with missing values
            df = df.dropna(subset=[self.target_column])
            
            missing_after = df.isnull().sum().sum()
            print(f"✓ Missing values after cleaning: {missing_after}")
        else:
            print("✓ No missing values to handle")
        
        return df
    
    def balance_dataset(self, df, method='undersample'):
        """Balance the dataset using undersampling or oversampling"""
        print("\n" + "="*50)
        print("BALANCING DATASET")
        print("="*50)
        
        disease_counts = df[self.target_column].value_counts()
        print(f"Before balancing: {disease_counts.to_dict()}")
        
        if method == 'undersample':
            # Undersample to the minority class
            min_samples = disease_counts.min()
            balanced_dfs = []
            
            for disease in disease_counts.index:
                disease_df = df[df[self.target_column] == disease]
                balanced_df = resample(disease_df, 
                                     n_samples=min_samples, 
                                     random_state=42)
                balanced_dfs.append(balanced_df)
            
            df_balanced = pd.concat(balanced_dfs, ignore_index=True)
            
        elif method == 'oversample':
            # Oversample to the majority class
            max_samples = disease_counts.max()
            balanced_dfs = []
            
            for disease in disease_counts.index:
                disease_df = df[df[self.target_column] == disease]
                balanced_df = resample(disease_df, 
                                     n_samples=max_samples, 
                                     random_state=42, 
                                     replace=True)
                balanced_dfs.append(balanced_df)
            
            df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the balanced dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"After balancing: {df_balanced[self.target_column].value_counts().to_dict()}")
        print(f"New dataset shape: {df_balanced.shape}")
        
        return df_balanced
    
    def prepare_features_target(self, df):
        """Separate features and target variable"""
        print("\n" + "="*50)
        print("PREPARING FEATURES AND TARGET")
        print("="*50)
        
        # Separate features and target
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Feature columns: {self.feature_columns}")
        
        return X, y
    
    def encode_target(self, y):
        """Encode target variable"""
        print("\n" + "="*50)
        print("ENCODING TARGET VARIABLE")
        print("="*50)
        
        # Fit and transform target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Print encoding mapping
        print("Target encoding mapping:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name} -> {i}")
        
        return y_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        print("\n" + "="*50)
        print("SCALING FEATURES")
        print("="*50)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"✓ Training features scaled: {X_train_scaled.shape}")
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            print(f"✓ Test features scaled: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1, random_state=42):
        """Split data into training, validation, and test sets"""
        print("\n" + "="*50)
        print("SPLITTING DATA")
        print("="*50)
        
        # First split: train + validation vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs validation
        validation_ratio = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"✓ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_data(self, df, balance_method=None):
        """Complete preprocessing pipeline"""
        print("\n" + "🔄 STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Explore data
        self.explore_data(df)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df)
        
        # 3. Balance dataset if requested
        if balance_method:
            df = self.balance_dataset(df, method=balance_method)
        
        # 4. Prepare features and target
        X, y = self.prepare_features_target(df)
        
        # 5. Encode target
        y_encoded = self.encode_target(y)
        
        # 6. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y_encoded)
        
        # 7. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.is_fitted = True
        
        print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_columns,
            'target_names': self.label_encoder.classes_
        }
    
    def save_preprocessor(self, filepath):
        """Save the fitted preprocessor"""
        if not self.is_fitted:
            print("⚠ Warning: Preprocessor not fitted yet")
            return
        
        preprocessor_data = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"✓ Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load a fitted preprocessor"""
        preprocessor_data = joblib.load(filepath)
        
        self.label_encoder = preprocessor_data['label_encoder']
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_column = preprocessor_data['target_column']
        self.is_fitted = True
        
        print(f"✓ Preprocessor loaded from {filepath}")
    
    def transform_new_data(self, X_new):
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call preprocess_data() first.")
        
        # Ensure all feature columns are present
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new[self.feature_columns]
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new)
        
        return X_new_scaled