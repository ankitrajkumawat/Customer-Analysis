import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Add the parent directory to the path so we can import the preprocessing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        self.models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def train_svm(self):
        """Train and save SVM model"""
        print("Training SVM model...")
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data_for_svm()
        
        # Initialize and train SVM
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = os.path.join(self.models_dir, 'svm_model.joblib')
        joblib.dump(svm_model, model_path)
        print(f"SVM model saved to {model_path}")
        
        return svm_model, accuracy
    
    def train_lstm(self):
        """Train and save LSTM model"""
        print("\nTraining LSTM model...")
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data_for_lstm()
        
        # Get input shape
        n_features = X_train.shape[2]
        
        # Build LSTM model
        model = Sequential([
            LSTM(64, input_shape=(1, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM Model Accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, 'lstm_model.h5')
        model.save(model_path)
        print(f"LSTM model saved to {model_path}")
        
        return model, accuracy
    
    def train_all_models(self):
        """Train both models and return their accuracies"""
        svm_model, svm_accuracy = self.train_svm()
        lstm_model, lstm_accuracy = self.train_lstm()
        
        return {
            'svm': {'model': svm_model, 'accuracy': svm_accuracy},
            'lstm': {'model': lstm_model, 'accuracy': lstm_accuracy}
        }

if __name__ == "__main__":
    # Get the path to the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 
                            'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Train models
    trainer = ModelTrainer(data_path)
    results = trainer.train_all_models()
    
    # Save the fitted transformers
    trainer.preprocessor.save_transformers()

    print("\nTraining Summary:")
    print(f"SVM Model Accuracy: {results['svm']['accuracy']:.4f}")
    print(f"LSTM Model Accuracy: {results['lstm']['accuracy']:.4f}") 