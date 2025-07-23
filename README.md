


# Predictive Maintenance: RUL Prediction Project

This project simulates sensor data and builds a machine learning model to predict the Remaining Useful Life (RUL) of equipment.

## 📦 Features
- Generates synthetic time series data (temperature, vibration, power usage)
- Cleans data (removes missing values and outliers)
- Creates rolling average features
- Visualizes relationships (scatter plots, heatmap)
- Trains a Linear Regression model to predict RUL
- Evaluates the model (MAE, MSE, R² score)
- Predicts RUL for new sensor readings

## 🛠 How it works
- Python libraries: `numpy`, `pandas`, `scipy`, `seaborn`, `matplotlib`, `scikit-learn`
- Generates date range from 2024-01-01 to 2026-09-26
- Saves synthetic data to `data.csv`
- Loads and processes data, then splits into training and testing sets
- Trains and tests the model, and prints evaluation metrics

## 📊 Output
- Visualizations: scatter plots and correlation heatmap
- Printed metrics in console: Mean Absolute Error, Mean Squared Error, R² score
- Predicted RUL for example new data points

## 🚀 How to run
1. Install dependencies:
   ```
   pip install numpy pandas scipy matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```
   python lrdcpm.py
   ```

## 📁 Files
- `lrdcpm.py`: main Python script
- `data.csv`: generated dataset
- `README.md`: project overview

## ✅ Notes
- This is an educational example using synthetic data.
- In real projects, use real sensor data and consider more advanced models.