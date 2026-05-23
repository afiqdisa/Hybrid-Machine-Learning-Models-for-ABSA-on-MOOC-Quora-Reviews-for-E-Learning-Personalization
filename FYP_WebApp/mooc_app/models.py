import os
import joblib

def load_sentiment_models(folder_path='Saved_Models'):
    """Loads all .joblib model files from a specified directory."""
    print(f"--- Loading models from folder: '{folder_path}' ---")
    try:
        models = {}
        # Correct the path to navigate up one level from 'mooc_app'
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', folder_path)
        for filename in os.listdir(dir_path):
            if filename.endswith('.joblib'):
                key_name = filename.replace('.joblib', '')
                models[key_name] = joblib.load(os.path.join(dir_path, filename))
                print(f"  Loaded: {filename}")
        print("--- All models loaded successfully! ---")
        return models
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        return None

# Load the models once when this module is imported
SENTIMENT_MODELS = load_sentiment_models()