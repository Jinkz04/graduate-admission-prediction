import joblib
import pandas as pd

# Load the linear model
linear_model = joblib.load('models/linear_reg.pkl')

# Check the model's feature names
print("Model Feature Names:")
print(linear_model.feature_names_in_)
