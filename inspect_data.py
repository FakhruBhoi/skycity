import pandas as pd
import numpy as np

# Load the dataset (adjust the path to match your exact file name)
df = pd.read_csv("dataset/SkyCity Auckland Restaurants & Bars (1).csv")

# 1. Print the shape of the data (Rows, Columns)
print(f"Dataset Shape: {df.shape}")

# 2. Display the first 5 rows to see what it looks like
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Info & Missing Values ---")
print(df.info())

print("\n--- Summary of Missing Values per Column ---")
print(df.isnull().sum())

# Create the ultimate target variable
df['TotalNetProfit'] = (
    df['InStoreNetProfit'] + 
    df['UberEatsNetProfit'] + 
    df['DoorDashNetProfit'] + 
    df['SelfDeliveryNetProfit']
)

print("\n--- Total Net Profit Statistics ---")
print(df['TotalNetProfit'].describe())

print("\n--- Checking for Anomalies ---")
# Check if key rates are within realistic bounds (e.g., percentages between 0 and 1)
print(df[['COGSRate', 'OPEXRate', 'CommissionRate']].describe())

# Keep only rows where rates are logical
df = df[(df['COGSRate'] >= 0) & (df['COGSRate'] <= 1)]

# Save the cleaned data to a new file so you don't overwrite the original
df.to_csv("dataset/SkyCity_Restaurants_Cleaned.csv", index=False)
print("\nData inspection and cleaning complete! Saved to 'dataset/SkyCity_Restaurants_Cleaned.csv'")