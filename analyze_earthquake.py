import pandas as pd
import numpy as np

def analyze():
    with open('c:/VS projects/Major-Project/disaster_prediction_system/analysis_earthquake.txt', 'w', encoding='utf-8') as f:
        f.write("Loading earthquake_data_tsunami.csv...\n")
        df = pd.read_csv('c:/VS projects/Major-Project/disaster_prediction_system/datasets/earthquake_data_tsunami.csv')
        f.write(f"Shape: {df.shape}\n")
        
        f.write("\nColumns:\n")
        f.write(str(df.columns) + "\n")
        
        f.write("\nSummary Statistics:\n")
        f.write(str(df.describe()) + "\n")
        
        f.write("\nTarget 'tsunami' distribution:\n")
        f.write(str(df['tsunami'].value_counts()) + "\n")
        
        f.write("\nMagnitude distribution (>6.0):\n")
        f.write(str((df['magnitude'] >= 6.0).value_counts()) + "\n")

if __name__ == "__main__":
    analyze()
