# Results & Performance Analysis

## Key Performance Metrics

The model was evaluated on a 20% hold-out test set consisting of ~80,000 flights.

| Metric                      | Value         | Interpretation                                                                                             |
| --------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| **R-Squared (R²)**          | **0.91**      | The model explains 91% of the variance in baggage volume. This indicates a high fit to the operational reality. |
| **Mean Absolute Error (MAE)** | **3.72 Bags** | On average, the prediction deviates from the actual bag count by just 3.72 bags.                           |
| **Avg Error Rate**          | **~24%**       | Based on an average flight volume of ~16 bags.                                                             |

Source file: `model_metrics.json` (captures reproducible metric outputs from the latest run).

## Visualizations

The following figures are produced by the operational modeling and reporting scripts and saved in the `Results/` directory.

1. **fig1_scatter_real.png** – Actual vs. Predicted bag counts. Tight diagonal pattern (y = x) with limited dispersion visually supports the reported R² (~0.91).
2. **fig2_combo_real.png** – Combined chart of total baggage volume (area) vs. net prediction error (line) over recent high-volume operational windows; error line remains centered near zero even during spikes (3,000+ bags/hour).
3. **fig3_error_by_hour.png** – Error distribution across hours of the day; helps identify any systematic under/over prediction tied to temporal operational dynamics.
4. **fig4_residuals_vs_predicted.png** – Residuals vs. predicted values; absence of funnel shape indicates no strong heteroscedasticity and stable performance across volume ranges.

## Metric Artifact

`model_metrics.json` stores structured outputs (e.g., R², MAE, error rate) enabling downstream automation, dashboard refresh, and auditability.

## Feature Importance

The model identified **Connection Ratio** and **Passenger Count** as the top drivers of baggage volume. `Route` and `Time of Day` played a significant secondary role, supporting the hypothesis that operational variability and trip purpose influence baggage rates.