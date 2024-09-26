from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

FILE_PATH = os.getenv('FILE_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
FEATURES = ['vibration_level', 'temperature_C', 'pressure_PSI', 'flow_rate_m3h']
TARGET = 'failure'
SEQUENCE_LENGTH = 10
STEP = 1

# FastAPI app instance
app = FastAPI()

class Filepath(BaseModel):
    """Schema for prediction request."""
    filepath: str

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df[FEATURES]
    if TARGET in df.columns:
        Y = df[TARGET]

    X_local, Y_local = [], []
    for start in range(0, len(df) - SEQUENCE_LENGTH, STEP):
        end = start + SEQUENCE_LENGTH
        X_local.append(X[start:end].values)
        if TARGET in df.columns:
            Y_local.append(Y.iloc[end - 1])
    
    if TARGET in df.columns:
        return np.array(X_local), np.array(Y_local)
    else:
        return np.array(X_local), None

def save_predictions(df, predictions):
    df['predict'] = None  # Initialize 'predict' column
    df['predict'].iloc[SEQUENCE_LENGTH:] = predictions.flatten()  # Assign predictions
    df.to_csv(f'{FILE_PATH}_predicted.csv', index=False)
    
def scale_features(df):
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    return scaler

def load_trained_model(model_path):
    return load_model(model_path)

def predict(model, X_local):
    predictions = model.predict(X_local)
    return np.where(predictions > 0.5, 1, 0)

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['no failure', 'failure'])
    return report

@app.on_event("startup")
def startup_event():
    """Load model and preprocess data on startup."""
    global model, df
    model = load_trained_model('my_model.h5')

@app.post("/predict/")
def predict_failure(request: Filepath):
    """Endpoint to predict failure based on input data."""
    df = load_data(FILE_PATH)
    scale_features(df)
    input_data = df.copy()
    X_local, _ = preprocess_data(input_data)
    y_test_pred = predict(model, X_local)
    save_predictions(df, y_test_pred)
    
    if TARGET in df.columns:
        report = evaluate_model(df[TARGET][SEQUENCE_LENGTH:].values, y_test_pred)
        return {"report": report}
    
    return {f"prediction saved :{FILE_PATH.replace('.csv','_predicted.csv')}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
