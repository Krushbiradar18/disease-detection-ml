from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

class ImprovedModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

        # Base models
        self.models = {
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        }

        # Ensemble model (VotingClassifier: soft voting)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('nb', self.models['naive_bayes']),
                ('rf', self.models['random_forest'])
            ],
            voting='soft'
        )

        self.trained_models = {}
        self.best_model = None
        self.selected_features = None

    def apply_smote(self, X, y):
        print("\n🔄 Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"✅ Balanced dataset shape: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def apply_feature_selection(self, X, y, k=10):
        print(f"\n🔍 Applying SelectKBest (top {k} features)...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        self.selected_features = selector.get_support(indices=True)
        print(f"✅ Selected features indices: {self.selected_features}")
        return X_selected, self.selected_features

    def train_models(self, X_train, y_train):
        print("\n📚 Training base models...")
        for name, model in self.models.items():
            print(f"→ Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

        print("\n🤝 Training VotingClassifier ensemble...")
        self.ensemble_model.fit(X_train, y_train)
        self.trained_models['voting_classifier'] = self.ensemble_model

    def cross_validate_models(self, X, y):
        print("\n📊 5-Fold Cross-Validation Results:")
        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results[name] = scores
            print(f"  {name}: Mean accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")
        return results

    def evaluate_models(self, X_test, y_test):
        print("\n🧪 Evaluating models on validation/test set...")
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=False)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'report': report
            }
            print(f"✅ {name} Accuracy: {accuracy:.4f}")
        return results

    def save_best_model(self, results, file_path):
        best_model_name = max(results, key=lambda name: results[name]['accuracy'])
        self.best_model = self.trained_models[best_model_name]
        joblib.dump(self.best_model, file_path)
        print(f"\n💾 Best model '{best_model_name}' saved to {file_path}")
        return best_model_name