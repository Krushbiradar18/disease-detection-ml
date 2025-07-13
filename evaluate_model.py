import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from src.evaluation import ModelEvaluator

# Load model
model = joblib.load('models/best_model.pkl')

# Load preprocessor
preprocessor = joblib.load('models/preprocessor.pkl')
label_encoder = preprocessor['label_encoder']
feature_columns = preprocessor['feature_columns']

# Load test data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
evaluator = ModelEvaluator(label_encoder=label_encoder)
evaluator.plot_confusion_matrix(y_test, y_pred)

# Feature Importance
evaluator.feature_importance(model, feature_columns)