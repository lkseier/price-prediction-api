import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
import joblib, os, json, pickle

#filename = r"C:/Users/Becode/immo-eliza-ML/immoEliza-ML/Charly's model/data/data_cleanned.csv"
filename = os.path.join("..", "immoEliza-ML", "ml_ready_real_estate_data_soft_filled.csv")
df = pd.read_csv(filename)

features = ['bedroomCount','habitableSurface', 'province_encoded', 'epcScore_encoded',
            'bathroomCount', 'hasLift_encoded']

X = df[features] # X: features to the model
y = df['price'] # y: target variable (price)

# convert categorical variables to numerical values

if "price" in df.columns:
        # Remove rows with missing prices (can't train without target)
        before_price = len(df)
        df = df.dropna(subset=["price"])
        after_price = len(df)
        print(f"Removed {before_price - after_price} rows with missing prices")
        
for column in df.columns:
    if df[column].dtype == 'object':
        df = pd.get_dummies(df, columns=[column], drop_first=True)

#Define features and target variable
features = df.columns[df.columns != 'price'].tolist() # all columns except 'price'
X = df[features]
y = df['price']

print(X)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Apply one-hot encoding to categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cols = pd.DataFrame(encoder.fit_transform(X[categorical_cols]),
                            columns=encoder.get_feature_names_out(categorical_cols))

# reset index to align with original DataFrame
encoded_cols.index = X.index

# Drop original categorical columns and concatenate encoded columns
X = X.drop(categorical_cols, axis=1)
X = pd.concat([X, encoded_cols], axis=1)
print("Shape of X after concat:", X.shape)

# Remove outliers from the target variable y using IQR
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 - 1.5 * IQR

#Filter the dataset to keep only the non-outlier rows
mask = (y > lower_bound) & (y < upper_bound)
X = X.values
y = y.values
#----- End of outlier removal ------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#prepare the parameter grid for GridSearchCV
param_grid = {'n_estimators' : [300],
              'max_features' : [0.7],
              'max_depth' : [15],
               'min_samples_leaf' : [12], 
                'min_samples_split' : [10]
}

grid =  GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1, verbose=2) # Initialize GridSearchCV with the model and parameter grid

grid.fit(X_train, y_train) # Fit the grid search to the training data
best_model = grid.best_estimator_ # Get the best model from the grid search

# Predicting the target variable using the trained model
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")
print("R2_score:", r2_score(y_test, best_model.predict(X_test)))
print("mean_squared_error:", mean_squared_error(y_test, best_model.predict(X_test)))
print("mean_absolute_error:", mean_absolute_error(y_test, best_model.predict(X_test)))

import os
import joblib
import json

def initialize_model(model, feature_names, MODEL_DIR="models", model_filename="best_model.pkl"):
    """
    Initializes the model saving process:
    - Creates a directory for the model if it doesn't exist
    - Saves the trained model as a .pkl file
    - Saves the feature names to a JSON file for later use
    """

    # Step 1: Create model directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)

    # Step 2: Save the model
    model_path = os.path.join(MODEL_DIR, model_filename)
    print(f"Saving model to {model_path}")
    joblib.dump(best_model, "saved_models/best_model.pkl")
    print(f"Model saved at {model_path}")

    # Step 3: Save feature names to JSON
    features_path = os.path.join(MODEL_DIR, "features.json")
    with open(features_path, "w") as f:
        json.dump(feature_names, f)
    print(f"Features saved at {features_path}")


