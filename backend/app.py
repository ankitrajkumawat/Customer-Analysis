import os
import sys
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import traceback

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.preprocessing import DataPreprocessor
# from backend.eda import EDAProcessor # We no longer need to import EDAProcessor here as we are serving static images

app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')
CORS(app)

# Initialize models and preprocessor
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(current_dir), 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
models_dir = os.path.join(current_dir, 'models', 'saved_models')

preprocessor = DataPreprocessor(data_path)
preprocessor.load_transformers() # Load the fitted transformers
svm_model = joblib.load(os.path.join(models_dir, 'svm_model.joblib'))
lstm_model = tf.keras.models.load_model(os.path.join(models_dir, 'lstm_model.h5'))
# eda_processor = EDAProcessor(data_path) # No longer needed here as images are pre-generated

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        input_data = pd.DataFrame([data])
        
        # Convert numerical inputs to appropriate types
        input_data['tenure'] = pd.to_numeric(input_data['tenure'])
        input_data['MonthlyCharges'] = pd.to_numeric(input_data['MonthlyCharges'])
        
        print(f"Input DataFrame before preprocessing: {input_data}")
        
        # Ensure all columns expected by the preprocessor are present
        # Add 'TotalCharges' as it's often derived or requires specific handling
        # This will now correctly multiply numeric types
        if 'TotalCharges' not in input_data.columns:
            input_data['TotalCharges'] = input_data['tenure'] * input_data['MonthlyCharges']
        
        # Convert SeniorCitizen to numeric if it's in data and not already numeric
        if 'SeniorCitizen' in input_data.columns:
            input_data['SeniorCitizen'] = pd.to_numeric(input_data['SeniorCitizen'], errors='coerce').fillna(0).astype(int)

        processed_data = preprocessor.transform_new_data(input_data)
        print(f"Processed data: {processed_data}")
        
        # Get predictions from both models
        svm_pred = svm_model.predict_proba(processed_data)[0]
        lstm_pred = lstm_model.predict(processed_data.values.reshape(1, 1, -1))[0][0]
        
        # Combine predictions (ensemble)
        final_pred = (svm_pred[1] + lstm_pred) / 2
        
        # Get confidence scores
        svm_confidence = max(svm_pred) * 100
        lstm_confidence = max(lstm_pred, 1 - lstm_pred) * 100
        
        return jsonify({
            'prediction': 'Churn' if final_pred > 0.5 else 'No Churn',
            'probability': float(final_pred),
            'svm_confidence': float(svm_confidence),
            'lstm_confidence': float(lstm_confidence),
            'ensemble_confidence': float(max(final_pred, 1 - final_pred) * 100)
        })
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback_data = request.get_json()
        # Here you would typically save the feedback to a database
        # For now, we'll just return a success message
        return jsonify({'message': 'Feedback received successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model-info')
def model_info():
    # Recalculate accuracies for display or load pre-calculated ones
    # For simplicity, let's just load the accuracies from the train_models.py output if available
    # Otherwise, we'd need to run evaluations here or save accuracies during training.
    
    # Placeholder - in a real app, you'd save and load these after training
    svm_accuracy = 0.8112 # From train_models.py output
    lstm_accuracy = 0.8133 # From train_models.py output

    return jsonify({
        'svm_accuracy': float(svm_accuracy),
        'lstm_accuracy': float(lstm_accuracy)
    })

# EDA Endpoints (Removed as images are pre-generated)
# @app.route('/api/eda/churn_distribution')
# def get_churn_distribution():
#     return eda_processor.get_churn_distribution_plot()
# 
# @app.route('/api/eda/tenure_vs_churn')
# def get_tenure_vs_churn():
#     return eda_processor.get_tenure_vs_churn_plot()
# 
# @app.route('/api/eda/monthly_charges_impact')
# def get_monthly_charges_impact():
#     return eda_processor.get_monthly_charges_impact_plot()
# 
# @app.route('/api/eda/service_impact')
# def get_service_impact():
#     return eda_processor.get_service_impact_plot()
# 
# @app.route('/api/eda/contract_type_analysis')
# def get_contract_type_analysis():
#     return eda_processor.get_contract_type_analysis_plot()
# 
# @app.route('/api/eda/payment_method_analysis')
# def get_payment_method_analysis():
#     return eda_processor.get_payment_method_analysis_plot()
# 
# @app.route('/api/eda/internet_service_analysis')
# def get_internet_service_analysis():
#     return eda_processor.get_internet_service_analysis_plot()
# 
# @app.route('/api/eda/demographic_patterns')
# def get_demographic_patterns():
#     return eda_processor.get_demographic_patterns_plot()
# 
# @app.route('/api/eda/additional_services_impact')
# def get_additional_services_impact():
#     return eda_processor.get_additional_services_impact_plot()
# 
# @app.route('/api/eda/overall_churn_factors')
# def get_overall_churn_factors():
#     return eda_processor.get_overall_churn_factors_plot()

if __name__ == '__main__':
    app.run(debug=True, port=5000) 