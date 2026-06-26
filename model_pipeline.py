import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib

# 1. LOAD THE DATASET
print("Loading data...")
df = pd.read_csv("dataset/SkyCity_Restaurants_Cleaned.csv")

# 2. FEATURE ENGINEERING (Build interactions)
print("Engineering features...")
df['UE_Cost_Interaction'] = df['CommissionRate'] * df['UE_share']
df['DD_Cost_Interaction'] = df['CommissionRate'] * df['DD_share']
df['SD_Cost_Interaction'] = df['DeliveryCostPerOrder'] * df['SD_share']

# Define explicitly what columns we are using
categorical_features = ['CuisineType', 'Segment', 'Subregion']
numerical_features = [
    'GrowthFactor', 'AOV', 'MonthlyOrders', 'InStoreShare', 
    'UE_share', 'DD_share', 'SD_share', 'CommissionRate', 
    'DeliveryRadiusKM', 'DeliveryCostPerOrder',
    'UE_Cost_Interaction', 'DD_Cost_Interaction', 'SD_Cost_Interaction'
]

X = df[categorical_features + numerical_features]
y = df['TotalNetProfit']

# 3. PREPROCESSING CONFIGURATION
# Encode text strings safely; let numeric terms pass right through
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

print("Transforming features...")
X_processed = preprocessor.fit_transform(X)

# Save the encoder setup so your Streamlit app can use it later
joblib.dump(preprocessor, 'preprocessor.pkl')

# 4. SPLIT DATA INTO TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 5. INITIALIZE AND TRAIN ALL MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42)
}

print("\n--- Training and Evaluating Models ---")
best_r2 = -1
best_model = None
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"{name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: ${mae:.2f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# Save the winning model configuration
joblib.dump(best_model, 'best_profit_model.pkl')
print(f"\n--> Successfully saved '{best_model_name}' as 'best_profit_model.pkl'")