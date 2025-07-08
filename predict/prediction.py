import joblib
import numpy as np

# Load the pre-trained model from the models folder
model = joblib.load('models/house_price_model.pkl')

def predict(preprocessed_data):
    """
    Predict the price of a house based on preprocessed data.

    Args:
        preprocessed_data (list or np.array): Preprocessed features of the house.

    Returns:
        float: Predicted price of the house, or an error message if prediction fails.
    """
    try:
        # Ensure the input is in the correct format (e.g., numpy array)
        input_data = np.array(preprocessed_data).reshape(1, -1)
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        return predicted_price
    except Exception as e:
        return f"Error: {str(e)}"