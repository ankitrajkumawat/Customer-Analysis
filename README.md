# Customer Churn Prediction System

This project implements a customer churn prediction system using both SVM and LSTM models, along with an interactive web interface for predictions and insights.

## Project Structure
```
customer_churn/
├── backend/
│   ├── models/
│   │   ├── train_models.py
│   │   ├── svm_model.joblib
│   │   └── lstm_model.h5
│   ├── data/
│   │   └── preprocessing.py
│   └── app.py
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/
│       ├── index.html
│       ├── insights.html
│       ├── predict.html
│       └── feedback.html
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models:
```bash
python backend/models/train_models.py
```

4. Run the application:
```bash
python backend/app.py
```

## Features

- **Home Page**: Comprehensive information about customer churn and its importance
- **Insights Page**: Interactive visualizations and analysis of customer churn patterns
- **Prediction Page**: Real-time churn prediction using both SVM and LSTM models
- **Feedback Form**: User feedback collection system

## Technologies Used

- Backend: Python, Flask, Scikit-learn, TensorFlow
- Frontend: HTML, CSS, JavaScript
- Data Visualization: Plotly, Matplotlib, Seaborn
- Machine Learning: SVM, LSTM #   C u s t o m e r - A n a l y s i s  
 