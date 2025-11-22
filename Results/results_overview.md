# Results & Performance Analysis

## Key Performance Metrics

The model was evaluated on a 20% hold-out test set consisting of ~80,000 flights.

| Metric                      | Value         | Interpretation                                                                                             |
| --------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| **R-Squared (R²)**          | **0.98**      | The model explains 98% of the variance in baggage volume. This indicates an exceptionally high fit to the operational reality. |
| **Mean Absolute Error (MAE)** | **1.20 Bags** | On average, the prediction deviates from the actual bag count by just 1.20 bags.                           |
| **Avg Error Rate**          | **~6%**       | Based on an average flight volume of ~21 bags.                                                             |

## Visualizations

### 1. Operational Precision (Combo Chart)

*   **File**: `slide_combo_chart.png`
*   This chart (generated via `visualize_combo_chart.py`) overlays the predicted error against the total operational volume for the last 7 days.
    *   **Blue Area**: Total Baggage Volume (Workload).
    *   **Red Line**: Net Prediction Error.
*   **Insight**: The error line remains flat and centered near zero even during massive volume spikes (3,000+ bags/hour), proving the model is stable under stress.

### 2. Prediction Accuracy (Scatter Plot)

*   **File**: `accuracy_plot.png`
*   A scatter plot comparing Actual vs. Predicted bag counts for individual flights.
*   **Insight**: The data points form a tight diagonal line (`y=x`) with very little dispersion, visually confirming the 0.98 R².

## Feature Importance

The model identified **Connection Ratio** and **Passenger Count** as the top drivers of baggage volume. Interestingly, `Route`, `Weather Scores`, and `Time of Day` played a significant secondary role, validating the hypothesis that "Operational Chaos" and "Trip Purpose" impact baggage rates significantly.