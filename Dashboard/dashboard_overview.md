# Dashboard Overview

The dashboard renders pre-generated model artifacts (metrics JSON + PNG figures) from the `Results/` directory. When you launch the Streamlit app (`app.py`), it automatically runs the refresh scripts so you do not need to manually execute anything in `Model/` unless you are retraining the model itself.

## Components

1. Artifact Refresh (automated by `app.py` on start)
   - Automatically calls `Model/create_validation_data.py` to rebuild `baggage_validation_data.csv` and `model_metrics.json` using the current `baggage_predictor_model.joblib`.
   - Automatically calls `Model/report_figures.py` to regenerate figures + `model_metrics.json` in `Results/`.

2. Dashboard Application (`Dashboard/app.py`)
   - Orchestrates the above refresh step, then loads `Results/model_metrics.json` and `Results/fig*.png`.
   - Provides date filtering and figure display. No retraining occurs here.

## Workflow

If the model or raw data changed and you want updated visuals:
1. (Optional) Retrain model manually:
   ```bash
   python Model/operational_model.py
   ```
2. Start dashboard (this runs validation + figure generation automatically):
   ```bash
   streamlit run Dashboard/app.py
   ```

## Artifacts Used by Dashboard
- Metrics: `Results/model_metrics.json`
- Figures:
  - `fig1_scatter_real.png` (Actual vs Predicted)
  - `fig2_combo_real.png` (Volume vs Error)
  - `fig3_error_by_hour.png` (Error by Hour)
  - `fig4_residuals_vs_predicted.png` (Residual diagnostic)

## Notes
- Manual execution of `create_validation_data.py` or `report_figures.py` is not required for normal dashboard use; `app.py` handles this.
- Explicit retraining (`operational_model.py`) is a separate manual step and only needed when model parameters or training data change.

## Contribution & Rationale
The dashboard was built as a thin Streamlit layer on top of the existing modeling workflow. It orchestrates the artifact refresh scripts (validation data + figures + metrics) and then renders the latest outputs from the Results directory without duplicating model logic. This design separates the various steps in the process of presenting the data, keeps the UI lightweight and quick to load, and ensures users can see up-to-date performance diagnostics. Overall, it turns raw prediction outputs into actionable, interpretable operational intelligence while remaining easy to maintain and extend.
