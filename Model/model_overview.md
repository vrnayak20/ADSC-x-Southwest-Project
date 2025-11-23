# Model Overview: Operational Baggage Predictor

## Summary

The final model is a supervised regression model built using XGBoost. It predicts the operational volume of checked bags for a specific flight leg 2–4 hours prior to departure using day-of operational signals.

## Model Architecture & Training

### Model Choice: XGBoost Regressor

An **XGBoost (Extreme Gradient Boosting)** model was selected for this regression task due to its high performance, scalability, and interpretability.

*   **High Performance**: XGBoost is renowned for its predictive accuracy, making it ideal for a high-stakes operational environment where precision matters.
*   **Scalability**: The model can handle the large volume of flight data (~500,000 records per year) efficiently.
*   **Interpretability**: Built-in feature importance allows us to understand *why* the model makes its predictions, which is critical for operational trust.
*   High performance for heterogeneous tabular features.
*   Handles class imbalance in operational edge cases (low volume flights).
*   Offers feature importance for transparency.

### Training Pipeline (operational_model.py)

1. Load raw flight data from Data/ (dataset provided securely; not in repo).
2. Clean and engineer features (see Data/Data_Sources.md).
3. Split 80/20 train/test with random_state for reproducibility.
4. Fit XGBoost regressor.
5. Evaluate and print metrics; save model as baggage_predictor_model.joblib.

### Features Used

Key engineered / selected features (non-exhaustive):
* passenger_count, originating_passenger_count, inbound_connecting_passenger_count
* load_factor, connecting_ratio
* OPI_SCORE (weather / operations proxy)
* route (ORIG_STN_CDE-DEST_STN_CDE categorical)
* departure_hour, day_of_week, month, day_sin, day_cos
* is_weekend, is_holiday
* SFL_ACFT_CAPY_CT, SFL_BLK_MIN_ITRVL, SFL_NSTP_MILE_CT

### Hyperparameters

The model was tuned using a randomized search with cross-validation. Key hyperparameters include:

*   `n_estimators`: 1000
*   `learning_rate`: 0.2
*   `max_depth`: 20
*   `reg_alpha`: 1
*   subsample & colsample_by_tree use defaults
*   random_state fixed for reproducibility

These parameters were chosen to balance model complexity and generalization, preventing overfitting while capturing nuanced patterns.

## Performance

Latest metrics (see Results/model_metrics.json):
* R²: 0.9100
* MAE: 3.65 bags
Interpretation: High explanatory power; low absolute error. Users should still monitor edge cases (very early flights, holidays) for drift.

## Environment Setup (Reproducibility)

To run training, validation, figure generation, and the dashboard:
1. Create and activate a virtual environment (Windows example):
   python -m venv venv
   .\\venv\\Scripts\\activate
2. Install dependencies:
   pip install -r requirements.txt
3. (Optional) Upgrade pip before install:
   python -m pip install --upgrade pip
4. Verify install:
   python -c "import xgboost, pandas, sklearn, streamlit; print('OK')"
5. Run workflow:
   python Model/operational_model.py
   python Model/create_validation_data.py
   python Model/report_figures.py
   streamlit run Dashboard/app.py

## Reproducibility

1. (Optional) Retrain model:
   python Model/operational_model.py
2. Refresh validation set:
   python Model/create_validation_data.py
3. Regenerate figures & metrics:
   python Model/report_figures.py
4. Launch dashboard:
   streamlit run Dashboard/app.py