import os
import pickle

def load_random_forest_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'random_forest', 'random_forest_model.pkl')
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_xgb_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'XGBModel', 'xgb_model.pkl')
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
