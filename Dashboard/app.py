import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import json

def run_pipeline():
    """
    Runs the model pipeline scripts to update the data and figures.
    Trains the model first only if the model artifact is missing.
    """
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(workspace_dir, 'Model')

    # Determine if model exists (saved at workspace root)
    model_file = os.path.join(workspace_dir, 'baggage_predictor_model.joblib')
    scripts = []
    if not os.path.exists(model_file):
        scripts.append('operational_model.py')  # train model only if missing

    # Follow-up scripts (always run)
    scripts.extend([
        'create_validation_data.py',
        'report_figures.py'
    ])

    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    status_placeholder.info("Initializing model pipeline...")

    for i, script_name in enumerate(scripts):
        script_path = os.path.join(model_dir, script_name)

        if os.path.exists(script_path):
            status_placeholder.info(f"Running {script_name}...")
            try:
                subprocess.run([sys.executable, script_path], check=True, cwd=workspace_dir)
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to run {script_name}. See terminal for details.")
                print(f"Error running {script_name}: {e}")
        else:
            print(f"Script not found: {script_name}, skipping.")

        progress_bar.progress((i + 1) / len(scripts))

    status_placeholder.success("Model pipeline updated successfully!")
    status_placeholder.empty()
    progress_bar.empty()

def main():
    """
    The main function for the Streamlit dashboard.
    """
    st.set_page_config(page_title="Southwest Baggage Prediction Dashboard", layout="wide")
    
    if 'pipeline_run' not in st.session_state:
        run_pipeline()
        st.session_state['pipeline_run'] = True

    st.title("Southwest Airlines Baggage Prediction Dashboard")

    # Load dynamic model metrics
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Results', 'model_metrics.json')
    r2_display = "N/A"
    mae_display = "N/A"
    mae_val = None
    avg_bags = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                _metrics = json.load(f)
            r2_val = _metrics.get('r2')
            mae_val = _metrics.get('mae')
            avg_bags = _metrics.get('avg_bags_per_flight')
            if isinstance(r2_val, (int, float)):
                r2_display = f"{r2_val:.4f}"
            if isinstance(mae_val, (int, float)):
                mae_display = f"{mae_val:.2f} Bags"
        except Exception as e:
            r2_display = "Error"
            mae_display = "Error"
            mae_val = None
            avg_bags = None
            print(f"Failed to load metrics: {e}")

    # --- Key Performance Metrics ---
    st.header("Model Performance Overview")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(label="R-Squared (R²)", value=r2_display)
    with col_m2:
        st.metric(label="Mean Absolute Error (MAE)", value=mae_display)
    with col_m3:
        st.metric(label="Avg Bags per Flight", value=(f"{avg_bags:.2f}" if isinstance(avg_bags, (int, float)) else "N/A"))
    with col_m4:
        if isinstance(mae_val, (int, float)) and isinstance(avg_bags, (int, float)) and avg_bags != 0:
            st.metric(label="Avg Error Rate", value=f"{(mae_val / avg_bags * 100):.2f}%")
        else:
            st.metric(label="Avg Error Rate", value="N/A")
    
    st.markdown("---")

    # --- Global Visualizations (Static Results) ---
    st.header("Global Model Results")
    workspace_dir = os.path.dirname(os.path.dirname(__file__))
    
    tab1, tab2, tab3, tab4 = st.tabs(["Operational Precision", "Accuracy & Error", "Diagnostics", "Feature Importance"])

    with tab1:
        st.subheader("Operational Precision: Volume vs. Variance")
        st.markdown("""
        **Insight:** The net error line (red) stays relatively close to zero versus large volume swings, with only modest deviation during peak load. 
        This indicates the model maintains stability under varying operational stress rather than collapsing at high throughput.
        """)
        combo_path = os.path.join(workspace_dir, 'Results', 'fig2_combo_real.png')
        if os.path.exists(combo_path):
            st.image(combo_path, use_container_width=True, caption="Blue Area: Total Volume | Red Line: Net Prediction Error")
        else:
            st.warning("Operational Precision chart (fig2) not found.")

    with tab2:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.subheader("Prediction Accuracy")
            st.markdown("""
            **Insight:** Points cluster tightly around the diagonal reference line (y = x), supporting strong explanatory power (≈0.98 R²). 
            Minor dispersion and a few edge deviations reflect realistic flight-level variability rather than systematic bias.
            """)
            scatter_path = os.path.join(workspace_dir, 'Results', 'fig1_scatter_real.png')
            if os.path.exists(scatter_path):
                st.image(scatter_path, use_container_width=True, caption="Actual vs. Predicted Baggage Counts")
            else:
                st.warning("Scatter plot image not found.")
        
        with col_v2:
            st.subheader("Error by Hour of Day")
            st.markdown("""
            **Insight:** Errors remain centered near zero across hours with a slightly wider spread during busiest operational windows. 
            This suggests the model generalizes well temporally, with only mild performance variability at peak intensity.
            """)
            error_by_hour_path = os.path.join(workspace_dir, 'Results', 'fig3_error_by_hour.png')
            if os.path.exists(error_by_hour_path):
                st.image(error_by_hour_path, use_container_width=True, caption="Error Distribution by Hour")
            else:
                st.warning("Error by hour image not found.")

    with tab3:
        st.subheader("Residual Analysis")
        st.markdown("""
        **Insight:** Residuals show no directional pattern (no systematic over- or under-prediction). 
        Variance grows slightly at higher predicted volumes, a mild heteroscedasticity typical of real operational data and not indicative of structural misspecification.
        """)
        residuals_path = os.path.join(workspace_dir, 'Results', 'fig4_residuals_vs_predicted.png')
        if os.path.exists(residuals_path):
            st.image(residuals_path, use_container_width=True)
        else:
            st.warning("Residuals plot not found.")

    with tab4:
        st.subheader("Feature Importance")
        st.markdown("""
        **Insight:** Connection Ratio and Passenger Count emerge as primary drivers, aligning with operational intuition about transfer complexity and demand. 
        Route, Weather, and Time of Day provide meaningful secondary signal, indicating the model captures both volume mechanics and contextual nuances without over-reliance on any single proxy.
        """)

    st.markdown("---")

    # --- Data Loading for Interactive Section ---
    sample_path = os.path.join(workspace_dir, 'Data', 'Sample Folder', 'southwest_data.csv')
    
    data = None
    if os.path.exists(sample_path):
        data = pd.read_csv(sample_path)
    else:
        st.error("Sample data not found. Please ensure 'Data/Sample Folder/southwest_data.csv' exists.")
        return

    if 'SCHD_DEP_CENT_TS' in data.columns:
        data['departure_time'] = pd.to_datetime(data['SCHD_DEP_CENT_TS'])
    else:
        st.error("Data does not contain 'SCHD_DEP_CENT_TS' column.")
        return

    # --- Sample Data Display ---
    st.header("Sample Flight Data")

    if data.empty:
        st.info("No data available.")
    else:
        cols_to_show = ['FLT_KEY', 'ORIG_STN_CDE', 'DEST_STN_CDE', 'SCHD_DEP_CENT_TS', 'total_checked_bag_count', 'PREDICTED_BAGGAGE']
        cols_to_show = [c for c in cols_to_show if c in data.columns]
        
        if cols_to_show:
            st.dataframe(data[cols_to_show], use_container_width=True)
        else:
            st.dataframe(data, use_container_width=True)
        
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sample Data as CSV",
            data=csv,
            file_name='sample_data.csv',
            mime='text/csv',
        )

if __name__ == '__main__':
    main()
