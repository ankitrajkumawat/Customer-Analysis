import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.numerical_columns = [
            'tenure', 'MonthlyCharges', 'TotalCharges'
        ]
        self.transformers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'transformers')
        os.makedirs(self.transformers_dir, exist_ok=True)
        
    def load_data(self):
        """Load and perform initial data cleaning"""
        df = pd.read_csv(self.data_path)
        
        # Convert TotalCharges to numeric, handling any non-numeric values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing values in TotalCharges with median
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Drop customerID as it's not useful for prediction
        df = df.drop('customerID', axis=1)
        
        return df
    
    def fit_encoders_and_scaler(self, df):
        """Fit LabelEncoders for categorical columns and StandardScaler for numerical columns"""
        for column in self.categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            self.label_encoders[column].fit(df[column])
        
        self.scaler.fit(df[self.numerical_columns])

    def encode_categorical_features(self, df):
        """Encode categorical features using fitted LabelEncoder"""
        df_encoded = df.copy()
        for column in self.categorical_columns:
            # Use transform, not fit_transform, as encoders should already be fitted
            df_encoded[column] = self.label_encoders[column].transform(df[column])
        return df_encoded
    
    def scale_numerical_features(self, df):
        """Scale numerical features using fitted StandardScaler"""
        df_scaled = df.copy()
        # Use transform, not fit_transform, as scaler should already be fitted
        df_scaled[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        return df_scaled
    
    def prepare_data_for_svm(self):
        """Prepare data for SVM model"""
        df = self.load_data()
        self.fit_encoders_and_scaler(df) # Fit encoders and scaler during training data prep
        df_encoded = self.encode_categorical_features(df)
        df_scaled = self.scale_numerical_features(df_encoded)
        
        X = df_scaled.drop('Churn', axis=1)
        y = df_scaled['Churn'].map({'Yes': 1, 'No': 0})
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def prepare_data_for_lstm(self):
        """Prepare data for LSTM model"""
        X_train, X_test, y_train, y_test = self.prepare_data_for_svm()
        
        # Reshape data for LSTM (samples, time steps, features)
        # We'll use 1 time step since we don't have temporal data
        X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        return X_train_lstm, X_test_lstm, y_train, y_test
    
    def get_feature_names(self):
        """Get list of feature names after preprocessing"""
        df = self.load_data()
        return [col for col in df.columns if col != 'Churn']
    
    def transform_new_data(self, data):
        """Transform new data for prediction"""
        # Ensure all expected columns are present, fill with default if needed for consistency
        # with training data. For categorical, fill with mode, for numerical, fill with median/mean.
        # This can be more robust, but for now we rely on the UI providing all fields.
        
        # Reorder columns to match the training data feature order
        original_df = self.load_data()
        expected_columns = [col for col in original_df.columns if col != 'Churn']
        
        for col in expected_columns:
            if col not in data.columns:
                # This means a column was not sent from the frontend. This needs to be addressed
                # by the frontend providing all expected columns.
                # For now, let's assume all columns are passed and throw an error if not
                raise ValueError(f"Missing expected column for transformation: {col}")

        # Reindex the input data to ensure column order and presence
        data = data[expected_columns]

        # Encode categorical features using loaded encoders
        for column in self.categorical_columns:
            if column in data.columns:
                data[column] = self.label_encoders[column].transform(data[column])
        
        # Scale numerical features using loaded scaler
        data[self.numerical_columns] = self.scaler.transform(data[self.numerical_columns])
        
        return data

    def save_transformers(self):
        """Save the fitted scaler and label encoders"""
        joblib.dump(self.scaler, os.path.join(self.transformers_dir, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(self.transformers_dir, 'label_encoders.joblib'))
        print("Transformers (scaler and label encoders) saved.")

    def load_transformers(self):
        """Load the fitted scaler and label encoders"""
        self.scaler = joblib.load(os.path.join(self.transformers_dir, 'scaler.joblib'))
        self.label_encoders = joblib.load(os.path.join(self.transformers_dir, 'label_encoders.joblib'))
        print("Transformers (scaler and label encoders) loaded.") 