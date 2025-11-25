import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_FILE = 'baggage_validation_data.csv' # Ensure this file exists from your previous steps
sns.set_theme(style="whitegrid")

def generate_figures():
    print("Loading real data...")
    try:
        df = pd.read_csv(DATA_FILE)
        if 'SCHD_DEP_CENT_TS' in df.columns:
             df['SCHD_DEP_CENT_TS'] = pd.to_datetime(df['SCHD_DEP_CENT_TS'])
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_FILE}. Run create_validation_file.py first.")
        return

    # --- Calculate and Save Metrics ---
    # This ensures model_metrics.json is updated whenever figures are generated
    print("Calculating metrics...")
    y_true = df['total_checked_bag_count']
    y_pred = df['PREDICTED_BAGGAGE']
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    avg_bags = y_true.mean()
    
    metrics = {
        'r2': round(r2, 4),
        'mae': round(mae, 2),
        'avg_bags_per_flight': round(avg_bags, 2),
        'generated_at_utc': pd.Timestamp.utcnow().isoformat(),
        'source': 'report_figures.py'
    }
    
    os.makedirs('Results', exist_ok=True)
    with open(os.path.join('Results', 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f)
    print(f"Updated Results/model_metrics.json: R2={metrics['r2']}, MAE={metrics['mae']}")

    # Calculate Diff for the Combo Chart
    df['Diff'] = df['total_checked_bag_count'] - df['PREDICTED_BAGGAGE']

    # ==========================================================================
    # FIGURE 1: Prediction Accuracy (Scatter Plot)
    # ==========================================================================
    # To avoid the "scam" look, we add transparency (alpha) to show point density
    # and ensure the red "perfect" line is clearly visible as a reference, not data.
    
    plt.figure(figsize=(10, 6))
    
    # Filter extreme 1% outliers just for the clean visual (optional)
    q_low = df['total_checked_bag_count'].quantile(0.005)
    q_high = df['total_checked_bag_count'].quantile(0.995)
    df_scatter = df[(df['total_checked_bag_count'] > q_low) & (df['total_checked_bag_count'] < q_high)]

    sns.scatterplot(x='total_checked_bag_count', y='PREDICTED_BAGGAGE', 
                    data=df_scatter, alpha=0.3, color='#1f77b4', s=20, label='Flight Data')
    
    # The "Perfect" line
    min_val = 0
    max_val = df_scatter['total_checked_bag_count'].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')
    
    plt.title('Figure 1: Prediction Accuracy (Actual vs. Predicted)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Baggage Count per Flight')
    plt.ylabel('Predicted Baggage Count per Flight')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Results/fig1_scatter_real.png', dpi=300)
    print("Generated fig1_scatter_real.png")

    # ==========================================================================
    # FIGURE 2: Operational Precision (Combo Chart)
    # ==========================================================================
    # Replicating the logic from error_slide.py to show volume vs. variance over time.
    
    # 1. Filter for the last 7 days of data
    last_date = df['SCHD_DEP_CENT_TS'].max()
    start_date = last_date - pd.Timedelta(days=7)
    df_week = df[df['SCHD_DEP_CENT_TS'] >= start_date].copy()

    # 2. Aggregate data hourly
    df_hourly = df_week.set_index('SCHD_DEP_CENT_TS').resample('h')[
        ['total_checked_bag_count', 'PREDICTED_BAGGAGE']
    ].sum().reset_index()

    # 3. Calculate the prediction error (Difference)
    df_hourly['Diff'] = df_hourly['total_checked_bag_count'] - df_hourly['PREDICTED_BAGGAGE']

    # --- PLOTTING ---
    sns.set_theme(style="white")
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # --- LEFT AXIS (Volume) ---
    # Plot volume as a filled area chart
    ax1.fill_between(df_hourly['SCHD_DEP_CENT_TS'], df_hourly['total_checked_bag_count'], 
                     color='#1f77b4', alpha=0.3, label='Total Baggage Volume')
    ax1.plot(df_hourly['SCHD_DEP_CENT_TS'], df_hourly['total_checked_bag_count'], 
             color='#1f77b4', linewidth=1)
    
    ax1.set_xlabel('Date & Time', fontsize=12)
    ax1.set_ylabel('Total Volume (Bags)', fontsize=14, color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0, df_hourly['total_checked_bag_count'].max() * 1.2)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- RIGHT AXIS (Error) ---
    ax2 = ax1.twinx()
    sns.lineplot(x='SCHD_DEP_CENT_TS', y='Diff', data=df_hourly, 
                 color='#d62728', linewidth=2, ax=ax2, label='Model Error (Net)',
                 legend=False) # Prevent duplicate legend
    
    # Add a zero-error reference line
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.set_ylabel('Prediction Error (Bags)', fontsize=14, color='#d62728', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    # Center the error axis for a fair comparison
    max_error = max(abs(df_hourly['Diff'].min()), abs(df_hourly['Diff'].max()))
    ax2.set_ylim(-max_error * 3, max_error * 3)

    # --- FINAL POLISH ---
    plt.title('Operational Precision: Volume vs. Variance', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Manually combine legends from both axes into one
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Format the x-axis to show Date and Hour
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %Hh'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('Results/fig2_combo_real.png', dpi=300)
    print("Generated fig2_combo_real.png")

    # ==========================================================================
    # FIGURE 3: Prediction Error by Hour of Day (Box Plot)
    # ==========================================================================
    # This plot shows if the model's error is consistent throughout the day.
    # It helps to identify if performance degrades during specific operational hours.
    df['DEPARTURE_HOUR'] = df['SCHD_DEP_CENT_TS'].dt.hour
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='DEPARTURE_HOUR', y='Diff', data=df, palette='coolwarm')
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')
    plt.title('Figure 3: Prediction Error by Hour of Day', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Prediction Error (Actual - Predicted)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Results/fig3_error_by_hour.png', dpi=300)
    print("Generated fig3_error_by_hour.png")

    # ==========================================================================
    # FIGURE 4: Residuals vs. Predicted Values
    # ==========================================================================
    # This plot helps to identify if the error variance is consistent across all
    # levels of prediction (homoscedasticity). We want to see a random cloud of
    # points with no discernible pattern.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PREDICTED_BAGGAGE', y='Diff', data=df, alpha=0.3, color='purple')
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Figure 4: Residuals vs. Predicted Values', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Baggage Count')
    plt.ylabel('Prediction Error (Residuals)')
    plt.tight_layout()
    plt.savefig('Results/fig4_residuals_vs_predicted.png', dpi=300)
    print("Generated fig4_residuals_vs_predicted.png")


if __name__ == "__main__":
    generate_figures()