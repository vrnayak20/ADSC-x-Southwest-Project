import pandas as pd
import numpy as np

DATA_FILE_NAME = 'Data\sample_09202024_tmln_obf.csv'
COL_BAGGAGE = 'total_checked_bag_count'


print(f"--- Loading data to analyze '{COL_BAGGAGE}' ---")

try:
    df = pd.read_csv(DATA_FILE_NAME, usecols=[COL_BAGGAGE])
    
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