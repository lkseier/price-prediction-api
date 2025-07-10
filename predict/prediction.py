from pathlib import Path
import pickle
import pandas as pd
import os
import xgboost as xgb

# Load the pre-trained model from the models folder
model_path = Path(__file__).parent.parent /'model'/'model_xgboost.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

def predict(input_df: pd.DataFrame) -> list:
    """
    Predict the price of a house based on preprocessed data.

    Returns:
        float: Predicted price of the house, or an error message if prediction fails.
    """
    preds = model.predict(input_df)
    preds = [int(round(p)) for p in preds]
    return preds