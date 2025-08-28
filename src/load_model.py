import joblib

def load_model(model_path: str):
    """
    Load the pre-trained model and encoders from the specified path.
    
    Args:
        model_path (str): Path to the saved model file.
        
    Returns:
        dict: A dictionary containing the loaded model and encoders.
    """
    loaded_model = joblib.load(model_path)
    return loaded_model