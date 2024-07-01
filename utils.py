import joblib
import os

def save_model(model, model_path, model_name='model.joblib'):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(model, os.path.join(model_path, model_name))
    print("Model saved to disk")

def load_model(model_path, model_name='model.joblib'):
    return joblib.load(os.path.join(model_path, model_name))
