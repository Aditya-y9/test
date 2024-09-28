import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import joblib

# Load the training dataset
train_data = pd.read_csv('Train.csv')

# Preprocessing
train_data['OptionType'] = train_data['OptionType'].map({'Call': 1, 'Put': 0})

# Feature engineering
train_data['Spot_Strike_Diff'] = train_data['Spot'] - train_data['Strike']
train_data['Inv_TimeToExpiry'] = 1 / (train_data['TimeToExpiry'] + 1e-5)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(train_data[['Strike', 'Spot', 'RiskfreeRate', 'MarketFearIndex', 'BuySellRatio']])
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Strike', 'Spot', 'RiskfreeRate', 'MarketFearIndex', 'BuySellRatio']))

# Combine with original data
train_data = pd.concat([train_data.reset_index(drop=True), poly_features_df], axis=1)

# Define features and target variable
features = train_data.columns.drop(['Id', 'OptionPrice', 'OptionType', 'Spot', 'Strike', 'TimeToExpiry'])  # Adjust accordingly
target = 'OptionPrice'

X = train_data[features]
y = train_data[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=50, cv=5, scoring='neg_mean_squared_error')
random_search.fit(X_train, y_train)

# Best model from random search
best_model = random_search.best_estimator_

# Validate the model
y_val_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f'Mean Squared Error on Validation Set: {mse}')

# Save the model to a file
joblib.dump(best_model, 'improved_option_price_model.pkl')
print("Improved model saved to 'improved_option_price_model.pkl'")
