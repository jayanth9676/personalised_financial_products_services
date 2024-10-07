import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import joblib

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def handle_missing_values(self):
        # Implement strategy for handling missing values
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        self.data[numeric_columns] = numeric_imputer.fit_transform(self.data[numeric_columns])
        self.data[categorical_columns] = categorical_imputer.fit_transform(self.data[categorical_columns])

    def encode_categorical_variables(self):
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        encoded_data = self.encoder.fit_transform(self.data[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(categorical_columns))
        self.data = pd.concat([self.data.drop(columns=categorical_columns), encoded_df], axis=1)

    def normalize_numerical_features(self):
        numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])

    def perform_feature_selection(self, target_column, n_features=15):
        self.y = self.data[target_column]
        self.X = self.data.drop(columns=[target_column])
        
        mi_scores = mutual_info_classif(self.X, self.y)
        mi_scores = pd.Series(mi_scores, index=self.X.columns)
        selected_features = mi_scores.nlargest(n_features).index.tolist()
        self.X = self.X[selected_features]

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def process_data(self, target_column):
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.normalize_numerical_features()
        self.perform_feature_selection(target_column)
        return self.split_data()

    def save_preprocessor(self, file_path):
        joblib.dump({
            'scaler': self.scaler,
            'encoder': self.encoder,
            'selected_features': self.X.columns.tolist()
        }, file_path)
        print(f"Preprocessor saved to {file_path}")

# Usage
if __name__ == "__main__":
    processor = DataProcessor('../data/personalized_loan_recommendation_dataset.csv')
    X_train, X_test, y_train, y_test = processor.process_data('loan_approved')
    processor.save_preprocessor('../models/preprocessor.joblib')
    print("Data processing completed.")