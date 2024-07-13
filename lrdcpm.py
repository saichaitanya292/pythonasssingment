import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2026-09-26')
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
timestamps = date_range.tolist()
timestamp_strings = [ts.strftime('%Y%m%d%H%M%S') for ts in timestamps]
timestamp_series = pd.Series(timestamp_strings)
print(timestamp_strings)
np.random.seed(42)
temperature = np.random.normal(loc=75, scale=5, size=1000)
vibration = np.random.normal(loc=0.02, scale=0.005, size=1000)
power_usage = np.random.normal(loc=150, scale=20, size=1000)
RUL = np.random.normal(loc=100, scale=10, size=1000)

# Create DataFrame
synthetic_data = pd.DataFrame({
    'timestamp': timestamp_series,
    'temperature': temperature,
    'vibration': vibration,
    'power_usage': power_usage,
    'RUL': RUL
})

# Save to CSV
synthetic_data.to_csv('data.csv', index=False)
data = pd.read_csv('data.csv')
# Check for missing values
c = data.isnull().sum()

print(c)

# Fill missing values or drop them
data = data.dropna()

# Handle outliers if necessary (example using Z-score)

data = data[(np.abs(stats.zscore(data.drop('timestamp', axis=1))) < 3).all(axis=1)]

# Create rolling average features
data['temperature_roll_mean'] = data['temperature'].rolling(window=5).mean()
data['vibration_roll_mean'] = data['vibration'].rolling(window=5).mean()
data['power_usage_roll_mean'] = data['power_usage'].rolling(window=5).mean()

# Drop rows with NaN values created by rolling mean
data = data.dropna()



# Scatter plot
sns.pairplot(data, x_vars=['temperature', 'vibration', 'power_usage'], y_vars='RUL', height=5, aspect=0.7)
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()



# Features and target variable
X = data[['temperature', 'vibration', 'power_usage', 'temperature_roll_mean', 'vibration_roll_mean', 'power_usage_roll_mean']]
y = data['RUL']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Predict RUL for new data
new_data = pd.DataFrame({
    'temperature': [75, 80],
    'vibration': [0.02, 0.03],
    'power_usage': [150, 200],
    'temperature_roll_mean': [74, 79],
    'vibration_roll_mean': [0.021, 0.025],
    'power_usage_roll_mean': [145, 195]
})
predicted_rul = model.predict(new_data)
print("rul",predicted_rul)


