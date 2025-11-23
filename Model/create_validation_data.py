import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os, json

# Import the data loading and feature engineering function from the training script
from operational_model import load_and_clean_data, ALL_NUMERIC_FEATURES, ALL_CATEGORICAL_FEATURES, COL_BAGGAGE

# --- Configuration ---
MODEL_FILE = 'baggage_predictor_model.joblib'
DATA_FOLDER = 'Data/OneDrive_3_11-22-2025/'
OUTPUT_FILE = 'baggage_validation_data.csv'
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

def create_validation_file():
    """
    Loads the trained model, runs predictions on the test set of the original data,
    and saves the results to a CSV file for validation and analysis.
    """
    print("--- Starting Validation Data Creation ---")

    # 1. Load and process the data using the same function as the training script
    df = load_and_clean_data(DATA_FOLDER)
    if df is None:
        print("Halting execution due to data loading failure.")
        return

    # 2. Load the pre-trained model pipeline
    print(f"Loading model from {MODEL_FILE}...")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found: {MODEL_FILE}")
        print("Please ensure you have trained the model by running 'operational_model.py' first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # 3. Recreate the exact same train/test split as in the training script
    print(f"Splitting data... (Test size: {TEST_SET_SIZE}, Random State: {RANDOM_STATE})")
    X = df[ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES]
    y = df[COL_BAGGAGE]

    # We only need the test set for validation
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    # 4. Make predictions on the test set
    print("Generating predictions on the test set...")
    y_pred = model.predict(X_test)

    # 5. Create the validation DataFrame
    print("Assembling the validation data file...")
    
    # Start with the original test data, indexed correctly
    validation_df = df.loc[X_test.index].copy()
    
    # Add the actual and predicted baggage counts
    validation_df['PREDICTED_BAGGAGE'] = y_pred
    
    # Calculate the prediction error (residual)
    validation_df['prediction_error'] = validation_df[COL_BAGGAGE] - validation_df['PREDICTED_BAGGAGE']

    # --- Metrics JSON Output ---
    try:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        avg_bags = df[COL_BAGGAGE].mean()  # average bags per flight across full dataset
        metrics = {
            'r2': round(r2, 4),
            'mae': round(mae, 2),
            'avg_bags_per_flight': round(avg_bags, 2),
            'generated_at_utc': pd.Timestamp.utcnow().isoformat()
        }
        os.makedirs('Results', exist_ok=True)
        with open(os.path.join('Results', 'model_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        print(f"Saved metrics JSON: R2={metrics['r2']}, MAE={metrics['mae']}, AVG_BAGS={metrics['avg_bags_per_flight']}")
    except Exception as e:
        print(f"Could not save metrics JSON: {e}")

    # 6. Save the results to a CSV file
    try:
        validation_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n--- Success! ---")
        print(f"Validation data saved to '{OUTPUT_FILE}'")
        print(f"The file contains {len(validation_df)} rows.")
    except Exception as e:
        print(f"\n--- Error ---")
        print(f"Could not save the validation file: {e}")

if __name__ == "__main__":
    create_validation_file()
