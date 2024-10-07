import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import shap

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def train_random_forest(self):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

    def train_xgboost(self):
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0]
        }
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Model: {type(self.model).__name__}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    def perform_cross_validation(self, cv=5):
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV score: {np.mean(scores):.4f}")

    def explain_predictions(self, sample_size=100):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test[:sample_size])
        shap.summary_plot(shap_values, self.X_test[:sample_size], plot_type="bar")

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

# Usage
if __name__ == "__main__":
    # Assume X_train, X_test, y_train, y_test are loaded from processed data
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_xgboost()
    trainer.evaluate_model()
    trainer.perform_cross_validation()
    trainer.explain_predictions()
    trainer.save_model('../models/loan_approval_model.joblib')