import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('apps.csv')

# Data Preparation: Clean and correct data types
# Fill missing values for 'Rating' and 'Reviews'
data['Rating'] = data['Rating'].fillna(data['Rating'].mean())
data['Reviews'] = data['Reviews'].fillna(0).astype(int)

# Convert Size to numeric (handling 'M' and 'k')
def convert_size(size):
    if isinstance(size, str):
        size = size.replace('M', '').replace('k', '')
        if 'M' in size:
            return float(size) * 1e6
        elif 'k' in size:
            return float(size) * 1e3
    return float(size) if pd.notnull(size) else None

data['Size'] = data['Size'].apply(convert_size)

# Convert Rating to float
data['Rating'] = data['Rating'].astype(float)

# Category Exploration: Investigate app distribution across categories
category_counts = data['Category'].value_counts()
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar')
plt.title('App Distribution Across Categories')
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.xticks(rotation=45)
plt.show()

# Metrics Analysis: Average rating and size by category
avg_rating = data.groupby('Category')['Rating'].mean()
avg_size = data.groupby('Category')['Size'].mean()

print("Average Ratings by Category:\n", avg_rating)
print("\nAverage Size by Category:\n", avg_size)

# Sentiment Analysis: Assess user sentiments through reviews
def get_sentiment(review):
    return TextBlob(review).sentiment.polarity

# Ensure reviews are strings before applying sentiment analysis
data['Sentiment'] = data['Reviews'].astype(str).apply(get_sentiment)

# Interactive Visualization: Scatter plot of Size vs Rating colored by Category
plt.figure(figsize=(12, 8))
plt.scatter(data['Size'], data['Rating'], alpha=0.5)
plt.title('Size vs Rating of Apps')
plt.xlabel('Size (in MB)')
plt.ylabel('Rating')
plt.colorbar(label='Category Code')  # Placeholder for color coding if needed
plt.show()

# Machine Learning Model Development: Predicting app ratings based on features
X = data[['Size', 'Reviews']]
y = data['Rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions and evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
