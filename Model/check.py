import pandas as pd
import numpy as np
import os

DATA_FOLDER_NAME = 'Data/OneDrive_3_11-22-2025/'
COL_BAGGAGE = 'total_checked_bag_count'


print(f"--- Loading data from folder {DATA_FOLDER_NAME} to analyze '{COL_BAGGAGE}' ---")

try:
    all_files = [os.path.join(DATA_FOLDER_NAME, f) for f in os.listdir(DATA_FOLDER_NAME) if f.endswith('.csv')]
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER_NAME}")

    df_list = []
    for file_path in all_files:
        df_single = pd.read_csv(file_path, usecols=[COL_BAGGAGE])
        df_list.append(df_single)
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Get statistics
    stats = df[COL_BAGGAGE].describe()
    
    print("\n--- Baggage Statistics ---")
    print(stats)
    
    mean_bags = stats['mean']
    mae = 1.20
    percent_error = (mae / mean_bags) * 100
    
    print("\n--- Model Quality ---")
    print(f"Average bags per flight (mean): {mean_bags:.2f}")
    print(f"Your Mean Absolute Error (MAE): {mae}")
    print(f"Your model's average error is {percent_error:.2f}% of the average.")

except Exception as e:
    print(f"An error occurred: {e}")