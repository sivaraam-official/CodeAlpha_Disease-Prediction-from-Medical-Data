import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# Set plotting style
sns.set_theme(style="whitegrid")

class DiseasePredictionSystem:
    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        self.results = {}
        self.best_model_name = None
        self.best_model = None

    def load_and_explore_data(self):
        """Loads dataset and performs Exploratory Data Analysis (EDA)."""
        print("--- Loading Dataset ---")
        data = load_breast_cancer(as_frame=True)
        self.dataset = data.frame
        self.X = data.data
        self.y = data.target # 0: Malignant, 1: Benign
        
        print(f"Dataset Shape: {self.dataset.shape}")
        print("\nMissing Values:")
        print(self.dataset.isnull().sum().sum()) # Should be 0 for this dataset
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.X.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        print("Saved: correlation_heatmap.png")

    def preprocess_data(self):
        """Splits and scales the dataset."""
        print("\n--- Preprocessing Data ---")
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for feature importance later
        self.X_train = pd.DataFrame(self.X_train, columns=self.X.columns)
        self.X_test = pd.DataFrame(self.X_test, columns=self.X.columns)
        print("Data split and scaled successfully.")

    def train_and_evaluate(self):
        """Trains models and calculates evaluation metrics."""
        print("\n--- Training and Evaluating Models ---")
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_prob)
            cm = confusion_matrix(self.y_test, y_pred)
            
            self.results[name] = {
                'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
                'F1-Score': f1, 'ROC-AUC': roc_auc, 'Model': model,
                'ConfusionMatrix': cm
            }
            
            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.savefig('roc_curves.png')
        plt.close()
        print("Saved: roc_curves.png")

    def select_best_model(self):
        """Selects the best model based on F1-Score."""
        print("\n--- Model Comparison ---")
        results_df = pd.DataFrame(self.results).T.drop(columns=['Model', 'ConfusionMatrix'])
        print(results_df.to_markdown())
        
        # Plot Model Comparison
        results_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        print("Saved: model_comparison.png")
        
        # Select best based on F1-Score (balance of precision and recall)
        self.best_model_name = results_df['F1-Score'].astype(float).idxmax()
        self.best_model = self.results[self.best_model_name]['Model']
        print(f"\nBest Model Selected: {self.best_model_name}")

    def plot_feature_importance(self):
        """Plots feature importance for tree-based models."""
        if self.best_model_name in ['Random Forest', 'XGBoost']:
            importance = self.best_model.feature_importances_
            feat_imp = pd.Series(importance, index=self.X.columns).sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_imp, y=feat_imp.index, palette='viridis')
            plt.title(f'Top 10 Feature Importances ({self.best_model_name})')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            print("Saved: feature_importance.png")

    def save_artifacts(self):
        """Saves the trained model and the scaler."""
        joblib.dump(self.best_model, 'best_disease_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        print("\n--- Artifacts Saved ---")
        print("Saved model to 'best_disease_model.joblib'")
        print("Saved scaler to 'scaler.joblib'")

    def predict_risk(self, patient_data):
        """
        Accepts patient information and predicts disease risk.
        patient_data should be a dictionary matching feature names.
        """
        df = pd.DataFrame([patient_data])
        scaled_data = self.scaler.transform(df)
        prediction = self.best_model.predict(scaled_data)[0]
        probability = self.best_model.predict_proba(scaled_data)[0][prediction]
        
        # Map 0 and 1 back to standard target labels for Breast Cancer
        label = "Benign (No Malignant Disease)" if prediction == 1 else "Malignant (Disease Risk)"
        
        return {
            "Prediction": label,
            "Confidence": f"{probability * 100:.2f}%"
        }

if __name__ == "__main__":
    # 1. Initialize Pipeline
    pipeline = DiseasePredictionSystem()
    
    # 2. Run Pipeline Steps
    pipeline.load_and_explore_data()
    pipeline.preprocess_data()
    pipeline.train_and_evaluate()
    pipeline.select_best_model()
    pipeline.plot_feature_importance()
    pipeline.save_artifacts()
    
    # 3. Test Prediction Function (Simulating a single patient record)
    print("\n--- Testing Prediction Function ---")
    sample_patient = pipeline.X.iloc[0].to_dict() # Take the first patient in the dataset
    result = pipeline.predict_risk(sample_patient)
    print(f"Patient Assessment: {result['Prediction']} with {result['Confidence']} confidence.")