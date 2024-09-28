import pandas as pd
import joblib  # For loading the model

# Load the saved model
model = joblib.load('option_price_model.pkl')

# Function to predict option prices
def predict_option_prices(input_file):
    # Read the input data
    input_data = pd.read_csv(input_file)

    # Preprocessing
    input_data['OptionType'] = input_data['OptionType'].map({'Call': 1, 'Put': 0})
    
    # Feature engineering
    input_data['Spot_Strike_Diff'] = input_data['Spot'] - input_data['Strike']
    input_data['Inv_TimeToExpiry'] = 1 / (input_data['TimeToExpiry'] + 1e-5)  # Prevent division by zero

    # Define features for prediction
    features = ['OptionType', 'Strike', 'Spot', 'Inv_TimeToExpiry', 'RiskfreeRate', 'MarketFearIndex', 'BuySellRatio', 'Spot_Strike_Diff']
    
    # Prepare input data for the model
    X_new = input_data[features]
    
    # Make predictions
    input_data['OptionPrice'] = model.predict(X_new)
    
    # Ensure non-negative prices
    input_data['OptionPrice'] = input_data['OptionPrice'].clip(lower=0)
    
    return input_data[['Id', 'OptionPrice']]

# Example usage
input_file = 'input.csv'  # Replace with your input CSV file name
predicted_prices_df = predict_option_prices(input_file)

# Output the predicted prices
print(predicted_prices_df.to_csv(index=False, header=True))
