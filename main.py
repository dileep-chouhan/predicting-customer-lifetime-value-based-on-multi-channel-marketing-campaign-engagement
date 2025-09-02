import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 500
data = {
    'EmailCampaignOpens': np.random.randint(0, 10, num_customers),
    'SMSCampaignClicks': np.random.randint(0, 5, num_customers),
    'WebsiteVisits': np.random.randint(1, 20, num_customers),
    'SocialMediaEngagement': np.random.randint(0, 30, num_customers),
    'CustomerLifetimeValue': np.random.randint(100, 10000, num_customers)
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering (Not needed for synthetic data in this example) ---
# In a real-world scenario, this section would involve handling missing values, 
# outlier detection, and potentially creating new features (e.g., interaction terms).
# --- 3. Model Building ---
# Split data into training and testing sets
X = df.drop('CustomerLifetimeValue', axis=1)
y = df['CustomerLifetimeValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model (a simple model for demonstration)
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# --- 4. Model Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 5. Visualization ---
# Feature Importance (example using coefficients from linear regression)
plt.figure(figsize=(10, 6))
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients.plot(kind='bar')
plt.title('Feature Importance')
plt.ylabel('Coefficient')
plt.xlabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")
#Actual vs Predicted CLTV (example scatter plot)
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CLTV")
plt.ylabel("Predicted CLTV")
plt.title("Actual vs Predicted Customer Lifetime Value")
plt.savefig('actual_vs_predicted.png')
print("Plot saved to actual_vs_predicted.png")