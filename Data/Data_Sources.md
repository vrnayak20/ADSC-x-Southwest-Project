# Data Sources & Structure

## 1. Data Origin

The primary dataset used for this project is **Flight-Level Operational Data** provided directly by Southwest Airlines. It represents a historical snapshot of flight operations, passenger loads, and baggage metrics.

## 2. Access Method

The data was accessed via a secure CSV file provided for the **ADSC x Southwest collaborative project**. The raw data is stored locally within the `Data/` directory (git-ignored for privacy) and loaded into the training pipeline using `pandas`.

## 3. Data Structure

The dataset is structured as tabular, cross-sectional time-series data, where each row represents a unique flight leg.

### Key Feature Groups

*   **Identifiers**: `Date`, `Origin Station (ORIG_STN_CDE)`, `Destination Station (DEST_STN_CDE)`.
*   **Passenger Metrics (Primary Drivers)**: `Total passengers (passenger_count)`, `Originating (originating_passenger_count)`, `Connecting (inbound_connecting_passenger_count)`, and `Checked-in counts`.
*   **Operational Constraints**: `Aircraft Capacity (SFL_ACFT_CAPY_CT)`, `Scheduled Block Time (SFL_BLK_MIN_ITRVL)`, and `Distance (SFL_NSTP_MILE_CT)`.
*   **"Chaos" Metrics**: `Weather scores (OPI_SCORE)`, `Tail changes (TAIL_CHG_CT)`, and `Connection dependency counts`.
*   **Target Variable**: `total_checked_bag_count` (Continuous numeric).

## 4. Preprocessing Steps

To prepare the raw data for the XGBoost model, the following preprocessing pipeline is applied in `operational_model.py`:

### Data Cleaning

*   Rows with missing target values or critical timestamps are dropped.
*   Numeric columns are coerced to numeric types; errors are handled by filling with `0`.

### Feature Engineering

*   **Time Features**: Extracted `DEPARTURE_HOUR`, `DAY_OF_WEEK`, `MONTH`, and created cyclical time features (`day_sin`, `day_cos`) to represent annual seasonality.
*   **Route Construction**: Concatenated Origin and Destination (e.g., "HOU-AUS") to create a specific route category.
*   **Operational Flags**: Created boolean flags for `is_holiday` (covering major US holidays like Christmas, Thanksgiving, and July 4th) and `is_weekend`.
*   **Ratios & Interactions**: Calculated `load_factor` (Pax/Capacity) and `connecting_ratio` (Transfers/Total Pax).