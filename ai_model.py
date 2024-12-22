# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Enable inline plots for Jupyter
%matplotlib inline
# Load the synthetic dataset
data = pd.read_csv('/Synthetic_Coating_Dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Display the processed dataset
print(data.head())

# Visualize correlations between numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Define features and target
X = data.drop(columns=['Contact Angle (°)', 'Durability Score'])  # Features
y = data[['Contact Angle (°)', 'Durability Score']]  # Targets

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)

# Initialize the model
model = RandomForestRegressor(random_state=42, n_estimators=100)

# Train the model
model.fit(X_train, y_train)

print("Model training completed.")

# Make predictions
y_pred = model.predict(X_test)

# Display a few predictions vs actual values
comparison = pd.DataFrame({'Actual': y_test['Contact Angle (°)'].values, 
                           'Predicted': y_pred[:, 0]})
print(comparison.head())

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Scatter plot for predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test['Contact Angle (°)'], y_pred[:, 0], alpha=0.7, color='blue')
plt.plot([y_test['Contact Angle (°)'].min(), y_test['Contact Angle (°)'].max()], 
         [y_test['Contact Angle (°)'].min(), y_test['Contact Angle (°)'].max()], 
         color='red', linewidth=2)
plt.title('Actual vs Predicted: Contact Angle')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance')
plt.show()
