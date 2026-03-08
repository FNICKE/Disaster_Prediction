import pandas as pd
import numpy as np

def analyze():
    with open('c:/VS projects/Major-Project/disaster_prediction_system/analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("Loading landslides...\n")
        df = pd.read_csv('c:/VS projects/Major-Project/disaster_prediction_system/datasets/landslides.csv')
        f.write(f"Shape: {df.shape}\n")
        
        f.write("\nColumns:\n")
        f.write(str(df.columns) + "\n")
        
        df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)
        df['distance']   = pd.to_numeric(df['distance'],   errors='coerce').fillna(0)
        df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
        df['injuries']   = pd.to_numeric(df['injuries'],   errors='coerce').fillna(0)
        df = df.dropna(subset=['latitude', 'longitude'])
        
        features = ['latitude', 'longitude', 'population', 'distance']
        X = df[features]
        y = ((df['fatalities'] > 0) | (df['injuries'] > 0)).astype(int)
        
        f.write("\nTarget Distribution:\n")
        f.write(str(y.value_counts()) + "\n")
        
        f.write("\nSummary statistics of features for Positive targets (Casualties = 1)\n")
        f.write(str(X[y == 1].describe()) + "\n")
        
        f.write("\nSummary statistics of features for Negative targets (Casualties = 0)\n")
        f.write(str(X[y == 0].describe()) + "\n")

if __name__ == "__main__":
    analyze()
