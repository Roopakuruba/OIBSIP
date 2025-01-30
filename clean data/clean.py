import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "AB_NYC_2019.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(['id', 'name', 'host_name', 'last_review'], axis=1)

# Fill missing values
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['host_id'] = df['host_id'].fillna(df['host_id'].mode()[0])
df['neighbourhood_group'] = df['neighbourhood_group'].fillna(df['neighbourhood_group'].mode()[0])
df['neighbourhood'] = df['neighbourhood'].fillna(df['neighbourhood'].mode()[0])

df = df.dropna()

# Encode categorical variables
encoder = LabelEncoder()
df['neighbourhood_group'] = encoder.fit_transform(df['neighbourhood_group'])
df['neighbourhood'] = encoder.fit_transform(df['neighbourhood'])
df['room_type'] = encoder.fit_transform(df['room_type'])

# Define features and target
X = df.drop(columns=['price'])
y = df['price']

# Remove outliers (optional)
y = np.log1p(y)  # Log transformation to handle skewed prices

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
