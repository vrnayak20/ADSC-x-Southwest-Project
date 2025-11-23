import pandas as pd
from datetime import datetime
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

DATA_FILE_NAME = 'Data/OneDrive_3_11-22-2025/' 

# Date/Route Features
COL_DATE_DEP = 'SCHD_DEP_CENT_TS'
COL_DATE_ARR = 'SCHD_ARR_CENT_TS'
COL_ORIGIN = 'ORIG_STN_CDE'
COL_DEST = 'DEST_STN_CDE'

# --- Passenger-related features ---
COL_PASSENGERS = 'passenger_count'
COL_ORIG_PASSENGERS = 'originating_passenger_count'
COL_CONN_PASSENGERS = 'inbound_connecting_passenger_count'
COL_THRU_PASSENGERS = 'through_passenger_count'
COL_CHECKED_IN_PASSENGERS = 'checked_in_passenger_count'

# --- Aircraft Capacity (Used for Load Factor calculation) ---
COL_AIRCRAFT_CAPACITY = 'SFL_ACFT_CAPY_CT'

# --- Target Variable (Our Goal) ---
COL_BAGGAGE = 'total_checked_bag_count'

# --- List of all numeric features ---
BASE_NUMERIC_FEATURES = [
    'passenger_count',
    'originating_passenger_count',
    'inbound_connecting_passenger_count',
    'through_passenger_count',
    'checked_in_passenger_count',

    'SFL_NSTP_MILE_CT',     
    'SFL_BLK_MIN_ITRVL',    
    'SFL_ACFT_CAPY_CT',     
    'SFL_TURN_MINS_QTY',    
    'SCHD_AOG_AT_ORIG_CT',  
    'SCHD_AOG_AT_DEST_CT',  
    'DAY_OF_YEAR',          
    'tail_seq',

    'incoming_flight_dependency_count',
    'ORIG_SCHD_OPI_SCORE',
    'DEST_SCHD_OPI_SCORE',
    'EST_DEP_DELAY_MIN',
    'SCHD_ACFT_AT_GATE_CT',

    'standby_passenger_count',
    'wheelchair_passenger_count',

    'pax_cnnct_est_minutes_befr_dep_min'
]

NEW_RATIO_FEATURES = [
    'load_factor',
    'connecting_ratio',
    'originating_ratio',
    'checked_in_ratio'
]

NEW_CALENDAR_FEATURES = [
    'is_holiday', 
    'is_weekend', 
    'day_sin', 
    'day_cos'
]

ALL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + NEW_RATIO_FEATURES + NEW_CALENDAR_FEATURES

ALL_CATEGORICAL_FEATURES = [
    'ROUTE',
    'YEAR',
    'MONTH',
    'DAY_OF_WEEK',
    'DEPARTURE_HOUR',
    COL_ORIGIN,
    COL_DEST
]
# ==============================================================================


def load_and_clean_data(folder_path):
    """
    Loads data from all CSVs in a folder, cleans it, and creates advanced features.
    """
    print(f"--- Loading Data from {folder_path} ---")
    
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not all_files:
        print(f"FATAL ERROR: No CSV files found in {folder_path}")
        return None

    df_list = []
    all_cols_to_load = [
        COL_DATE_DEP, COL_DATE_ARR, COL_ORIGIN, COL_DEST, COL_BAGGAGE
    ] + BASE_NUMERIC_FEATURES
    all_cols_to_load = list(set(all_cols_to_load))

    for file_name in all_files:
        try:
            df_single = pd.read_csv(file_name, usecols=all_cols_to_load)
            df_list.append(df_single)
        except FileNotFoundError:
            print(f"WARNING: File not found: {file_name}")
            continue
        except ValueError as e:
            print(f"WARNING: A required column was not found in {file_name}. Skipping.")
            print(f"Details: {e}")
            continue
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            continue
    
    if not df_list:
        print("FATAL ERROR: No data could be loaded.")
        return None

    df = pd.concat(df_list, ignore_index=True)

    print(f"Loaded {len(df)} rows from {len(df_list)} files.")
    df.columns = df.columns.str.strip()

    # --- 1. Basic Cleaning ---
    print("Cleaning data structure...")
    df = df.dropna(subset=[COL_DATE_DEP, COL_DATE_ARR, COL_ORIGIN, COL_DEST, COL_PASSENGERS, COL_BAGGAGE])
    
    # Convert date columns
    df[COL_DATE_DEP] = pd.to_datetime(df[COL_DATE_DEP])
    df[COL_DATE_ARR] = pd.to_datetime(df[COL_DATE_ARR])
    
    # --- 2. Categorical Features ---
    df['YEAR'] = df[COL_DATE_DEP].dt.year.astype(str)
    df['MONTH'] = df[COL_DATE_DEP].dt.month.astype(str)
    df['DAY_OF_WEEK'] = df[COL_DATE_DEP].dt.dayofweek.astype(str)
    df['DEPARTURE_HOUR'] = df[COL_DATE_DEP].dt.hour.astype(str)
    df['ROUTE'] = df[COL_ORIGIN] + '-' + df[COL_DEST]
    
    # --- 3. Holiday & Weekend Features ---
    print("Creating holiday and calendar features...")
    
    def is_holiday_season(date):
        # Christmas / New Year (Dec 20 - Jan 5)
        if (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5):
            return 1
        # Thanksgiving (Late Nov)
        if (date.month == 11 and date.day >= 20 and date.day <= 30):
            return 1
        # July 4th (July 1 - July 7)
        if (date.month == 7 and date.day >= 1 and date.day <= 7):
            return 1
        return 0

    # Apply the holiday logic
    df['is_holiday'] = df[COL_DATE_DEP].apply(is_holiday_season)
    
    # Is Weekend? (Fri=4, Sat=5, Sun=6)
    df['is_weekend'] = df[COL_DATE_DEP].dt.dayofweek.isin([4, 5, 6]).astype(int)

    # --- 4. Cyclical Time Features ---
    # This helps the model understand that Dec 31 is close to Jan 1
    day_of_year = df[COL_DATE_DEP].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.0)

    # --- 5. Clean Numeric Columns ---
    print("Forcing numeric columns to clean numbers...")
    for col in BASE_NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[BASE_NUMERIC_FEATURES] = df[BASE_NUMERIC_FEATURES].fillna(0)
    df = df[(df[COL_PASSENGERS] > 0) & (df[COL_BAGGAGE] >= 0)]
    
    # --- 6. Advanced Ratios ---
    print("Creating advanced ratio features...")
    df['load_factor'] = np.where(df[COL_AIRCRAFT_CAPACITY] > 0, df[COL_PASSENGERS] / df[COL_AIRCRAFT_CAPACITY], 0)
    df['connecting_ratio'] = df[COL_CONN_PASSENGERS] / df[COL_PASSENGERS]
    df['originating_ratio'] = df[COL_ORIG_PASSENGERS] / df[COL_PASSENGERS]
    df['checked_in_ratio'] = df[COL_CHECKED_IN_PASSENGERS] / df[COL_PASSENGERS]
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df[NEW_RATIO_FEATURES] = df[NEW_RATIO_FEATURES].fillna(0)
    
    print(f"Cleaning and engineering complete. {len(df)} rows remain.")
    return df


def build_model_pipeline(numeric_features, categorical_features):
    """
    Creates the full ML preprocessing and model pipeline.
    """
    # --- Numeric Transformer ---
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # --- Categorical Transformer ---
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(max_categories=1000,
                                 handle_unknown='ignore', 
                                 sparse_output=True))
    ])

    # --- Preprocessor ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Final Model Pipeline (Single XGBoost) ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            objective='count:poisson', # Optimized for count data
            n_estimators=1000,      # Reduced for faster training
            learning_rate=0.2,    
            max_depth=20,          # Reduced for faster training
            reg_alpha=1,
            n_jobs=-1,            
            random_state=42
        ))
    ])
    
    return model_pipeline


def main():
    """
    Main function to run the entire pipeline.
    """
    df = load_and_clean_data(DATA_FILE_NAME)
    
    if df is None:
        return

    # ==========================================================================
    # --- Single Model: Operational Baggage Predictor ---
    # ==========================================================================
    print("\n--- Starting: Operational Baggage Model ---")
    
    # Use the feature lists defined at the top of the script
    X = df[ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES]
    y = df[COL_BAGGAGE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model_pipeline(ALL_NUMERIC_FEATURES, ALL_CATEGORICAL_FEATURES)
    
    print("Training baggage model...")
    model.fit(X_train, y_train)

    # --- Evaluate the baggage model ---
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Baggage Model Evaluation ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} bags")
    print("\n--- Script Finished ---")

    # --- Feature Importance ---
    try:
        print("\n--- Feature Importances ---")
        
        # 1. Access the steps from the pipeline
        regressor = model.named_steps['regressor']
        preprocessor = model.named_steps['preprocessor']
        
        # 2. Get feature names from the preprocessor
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        
        # Categorical features are OneHotEncoded, so we get the new expanded names
        ohe_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(ALL_CATEGORICAL_FEATURES)
        
        # Combine them
        all_feature_names = list(num_features) + list(ohe_features)
        
        # 3. Match importances to names
        importances = pd.Series(regressor.feature_importances_, index=all_feature_names)
        importances = importances.sort_values(ascending=False)
        
        print("Top 20 most important features:")
        print(importances.head(20))

    except Exception as e:
        print(f"Could not get feature importances: {e}")
    

    print("\n--- Saving Model ---")
    # Save the entire pipeline (preprocessor + model)
    joblib.dump(model, 'baggage_predictor_model.joblib')
    print("Model saved as 'baggage_predictor_model.joblib'")

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")