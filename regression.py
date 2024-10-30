# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# 1. Data Preprocessing

# Drop the User ID column since it's just an identifier and doesn't contribute to predicting app usage
data_cleaned = data.drop('User ID', axis=1)

# Convert categorical variables into numerical using OneHotEncoder
# Categorical features: Gender, Activity Level, Location
categorical_features = ['Gender', 'Activity Level', 'Location']
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Use sparse_output=False
encoded_categorical = encoder.fit_transform(data_cleaned[categorical_features])

# Create a dataframe with the encoded features and concatenate with the numerical ones
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))
data_numerical = data_cleaned.drop(categorical_features, axis=1).reset_index(drop=True)
data_encoded = pd.concat([data_numerical, encoded_df], axis=1)

# 2. Splitting the Dataset

# Define independent variables (X) and the target variable (y)
X = data_encoded.drop('App Sessions', axis=1)  # Features (predictors)
y = data_encoded['App Sessions']  # Target variable (what we're predicting)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling

# We scale the numerical features (Age, Distance Travelled, Calories Burned) to standardize the range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Build the Regression Model

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 5. Predictions

# Use the trained model to predict App Sessions on the test set
y_pred = model.predict(X_test_scaled)

# 6. Model Evaluation

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate R-squared (Coefficient of Determination)
r_squared = r2_score(y_test, y_pred)

# Output the RMSE and R-squared
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")

# 7. Visualize the Results

# Plot actual vs predicted App Sessions
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
plt.xlabel('Actual App Sessions')
plt.ylabel('Predicted App Sessions')
plt.title('Actual vs Predicted App Sessions')
plt.grid(True)
plt.show()
