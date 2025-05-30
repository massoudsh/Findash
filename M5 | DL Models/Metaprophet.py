# Import necessary libraries
from prophet import Prophet  # Prophet for time series forecasting
import pandas as pd  # For data manipulation and preparation
import matplotlib.pyplot as plt  # For additional custom plotting

# Step 1: Prepare the data
# ------------------------
# Example dataset with two columns: 'ds' (date) and 'y' (value to forecast)
data = {
    'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
           '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'],
    'y': [10, 12, 15, 18, 19, 22, 21, 23, 25, 30]
}
df = pd.DataFrame(data)

# Convert the 'ds' column to datetime format
df['ds'] = pd.to_datetime(df['ds'])

# Step 2: Initialize and fit the Prophet model
# --------------------------------------------
# Initialize the Prophet model
model = Prophet()

# Add custom monthly seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Add holiday effects (e.g., New Year holidays)
holidays = pd.DataFrame({
    'holiday': 'new_year',
    'ds': pd.to_datetime(['2023-01-01', '2024-01-01']),
    'lower_window': 0,  # Number of days before the holiday to consider
    'upper_window': 1   # Number of days after the holiday to consider
})
model = Prophet(holidays=holidays)

# Fit the model to the data
model.fit(df)

# Step 3: Create a future dataframe for predictions
# -------------------------------------------------
# Generate future dates for the next 30 days
future = model.make_future_dataframe(periods=30)  # Predict 30 days into the future

# Step 4: Make predictions
# -------------------------
# Predict values for the future dates
forecast = model.predict(future)

# Step 5: Visualize the forecast
# ------------------------------
# Plot the forecast with historical data
model.plot(forecast)
plt.title("Forecast with Facebook Prophet")
plt.show()

# Step 6: Analyze forecast components
# -----------------------------------
# Plot trend, seasonality, and holiday effects
model.plot_components(forecast)
plt.show()

# Step 7: Print forecast summary
# ------------------------------
# Display the first few rows of the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# Notes:
# - 'yhat' is the predicted value.
# - 'yhat_lower' and 'yhat_upper' are the lower and upper bounds of the prediction interval.