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

# FILE_PATH = os.getenv('FILE_PATH')
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
    """Preprocess data to create sequences for model input.
    
    Args:
        df: The input DataFrame containing features with ou without target.

    Returns:
        np.array: Arrays of input sequences and corresponding target values.
    """
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

def save_predictions(df, predictions, FILEPATH):
    df['predict'] = None  # Initialize 'predict' column
    df['predict'].iloc[SEQUENCE_LENGTH:] = predictions.flatten()  # Assign predictions
    df.to_csv(FILEPATH.replace('.csv','_predicted.csv'), index=False)
    
    
def scale_features(df):
    """Scale features using MinMaxScaler to normalize data.
    
    Args:
        df: The input DataFrame to scale features.

    Task:
        The scaler object fitted to the features.
    """
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

def load_trained_model(model_path):
    """Load a trained Keras model.
    
    Args:
        model_path (str): Path to the trained model file.

    Returns:
        The loaded Keras model.
    """
    return load_model(model_path)

def predict(model, X_local):
    """Make predictions using the trained model.
    
    Args:
        model (keras.Model): The trained model.
        X_local (np.array): Input data for prediction.

    Returns:
        np.array: Array of predicted classes.
    """
    predictions = model.predict(X_local)
    return np.where(predictions > 0.5, 1, 0)

def evaluate_model(y_true, y_pred):
    """Evaluate the model and print the classification report only when the target variable is available.
    
    Args:
        y_true (np.array): True class labels.
        y_pred (np.array): Predicted class labels.

    Returns:
        str: The classification report as a string.
    """
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
    FILEPATH = request.filepath
    df = load_data(FILEPATH)
    scale_features(df)
    X_local, _ = preprocess_data(df)
    y_test_pred = predict(model, X_local)
    save_predictions(df, y_test_pred, FILEPATH)
    
    if TARGET in df.columns:
        report = evaluate_model(df[TARGET][SEQUENCE_LENGTH:].values, y_test_pred)
        return {"report": report}
    
    else:
        return {f"prediction saved :{request.filepath.replace('.csv','_predicted.csv')}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
