#Simple Laptop Price Predictor based on CPU, RAM, and Storage

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dataset of final score of 20 laptops based on various features and their corresponding prices 

data = {
    "CPU": [0.3,0.6,0.6,0.8,0.8,1.0,0.8,0.6,0.3,0.6,
            0.8,0.6,0.3,0.8,1.0,0.6,0.8,0.3,0.6,1.0],
    "GPU": [0.2,0.2,0.5,0.7,0.7,1.0,0.2,0.5,0.2,0.7,
            1.0,0.2,0.5,0.7,1.0,0.5,0.2,0.2,0.7,1.0],
    "RAM": [0.25,0.25,0.5,0.5,0.5,1.0,0.5,0.5,0.125,0.5,
            0.75,0.25,0.25,0.75,0.75,0.5,0.5,0.25,0.5,1.0],
    "Storage": [0.25,0.5,0.5,0.5,0.75,1.0,0.5,0.75,0.25,0.5,
                0.75,0.5,0.5,0.75,1.0,0.5,0.75,0.25,0.75,1.0],
    "Display": [0.6,0.6,0.6,0.6,0.8,0.8,0.8,0.6,0.4,0.6,
                1.0,0.6,0.6,0.8,1.0,0.6,0.8,0.4,0.8,1.0],
    "Build": [0.4,0.4,1.0,1.0,1.0,1.0,1.0,0.7,0.4,0.7,
              1.0,0.4,0.7,1.0,1.0,0.7,1.0,0.4,0.7,1.0],
    "Battery": [0.45,0.48,0.5,0.6,0.7,0.8,0.55,0.6,0.35,0.5,
                0.75,0.5,0.5,0.7,0.8,0.6,0.6,0.4,0.65,0.85],
    "Extras": [0.2,0.4,0.6,0.6,0.8,1.0,0.6,0.6,0.2,0.4,
               0.8,0.2,0.4,0.6,1.0,0.4,0.6,0.2,0.6,1.0],
    "Price": [30000,45000,65000,95000,130000,205000,110000,80000,22000,75000,
              180000,42000,55000,140000,210000,78000,105000,28000,115000,220000]
}

df = pd.DataFrame(data)

# Define Features & Target

X = df.drop("Price", axis=1)
y = df["Price"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression Model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression
print("Linear Regression Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Train Random Forest Model

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# Predict New Laptop Price

new_laptop = pd.DataFrame([{
    "CPU": 0.3,
    "GPU": 0.7,
    "RAM": 0.5,
    "Storage": 0.75,
    "Display": 0.8,
    "Build": 1.0,
    "Battery": 0.7,
    "Extras": 0.6
}])

predicted_price = rf_model.predict(new_laptop)
print("\nPredicted Price for New Laptop:", int(predicted_price[0]))
