# data_generator.py

"""
Data Generation Module for Simulated Telemetry

Disclaimer for Layman:
This script is like a movie director creating a scene. It generates
artificial sensor data that looks like it could come from a real machine
(like a rocket engine). We first create 'normal' data representing smooth
operation. Then, we intentionally add 'problems' or 'anomalies' – like sudden
spikes, drops, or gradual changes – to mimic sensor failures or system issues.
We keep a secret log of exactly when and where we added these problems, so later,
we can check if our 'detective' (the anomaly detection system) can find them.
The goal is to create realistic test data for our detector.
"""

import numpy as np
import pandas as pd
import json
import random
from config import (
    SAMPLING_RATE, DURATION, N_SAMPLES, SENSOR_CONFIG, CORRELATION_MATRIX,
    ANOMALY_CONFIG, NORMAL_DATA_PATH, ANOMALOUS_DATA_PATH, ANOMALY_LOG_PATH
)

def generate_baseline_data():
    """
    Generates baseline time series data for all sensors, including trends and
    correlated noise, but WITHOUT injected anomalies.

    Returns:
        pd.DataFrame: DataFrame with normal sensor data.
        np.ndarray: The generated noise array (for potential reuse/analysis).
    """
    print("Generating baseline sensor data...")
    time_indices = np.arange(N_SAMPLES)
    data = {}
    means = []
    std_devs = []
    sensor_names = list(SENSOR_CONFIG.keys())

    # 1. Generate baseline trends
    for name, config in SENSOR_CONFIG.items():
        baseline = config['mean'] + config['trend'](time_indices)
        data[name] = baseline
        means.append(config['mean']) # Store mean for noise generation
        std_devs.append(config['std_dev'])

    # 2. Generate correlated noise
    # Ensure correlation matrix matches std devs
    std_dev_array = np.array(std_devs)
    # Construct covariance matrix from correlation matrix and standard deviations
    # Cov(X, Y) = Corr(X, Y) * StdDev(X) * StdDev(Y)
    cov_matrix = np.outer(std_dev_array, std_dev_array) * CORRELATION_MATRIX

    # Generate multivariate normal noise
    # Each row is a time step, each column is a sensor
    noise = np.random.multivariate_normal(np.zeros(len(sensor_names)), cov_matrix, size=N_SAMPLES)
    print(f"Generated noise shape: {noise.shape}")

    # 3. Add noise to baselines
    for i, name in enumerate(sensor_names):
        data[name] += noise[:, i]

    # Create DataFramegit

    import datetime
    # ... inside generate_baseline_data ...
    run_time_origin = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current time as string
    df = pd.DataFrame(data, index=pd.to_datetime(np.arange(N_SAMPLES) / SAMPLING_RATE, unit='s', origin=run_time_origin))

    print("Baseline data generation complete.")
    return df, noise


def inject_anomalies(df_normal):
    """
    Injects various types of anomalies into the normal baseline data based on ANOMALY_CONFIG.

    Args:
        df_normal (pd.DataFrame): The normal sensor data.

    Returns:
        pd.DataFrame: DataFrame with anomalies injected.
        list: A log detailing the injected anomalies (type, sensor, start, end).
    """
    print("Injecting anomalies into the data...")
    df_anomalous = df_normal.copy()
    anomaly_log = []
    sensor_names = df_normal.columns
    active_anomalies = {name: None for name in sensor_names} # Track ongoing anomalies

    min_duration, max_duration = ANOMALY_CONFIG['anomaly_duration_range']

    for i in range(N_SAMPLES):
        timestamp = df_anomalous.index[i]
        for sensor in sensor_names:
            std_dev = SENSOR_CONFIG[sensor]['std_dev']

            # Check if an ongoing anomaly finishes
            if active_anomalies[sensor] and active_anomalies[sensor]['end_index'] == i:
                 # Restore value if it was a dropout? Or just let new anomalies overwrite?
                 # For now, just mark as finished. The effect persists until overwritten.
                 active_anomalies[sensor] = None

            # If no anomaly is active for this sensor, check if a new one should start
            if not active_anomalies[sensor]:
                anomaly_type = None
                rand_val = random.random()

                if rand_val < ANOMALY_CONFIG['spike_rate']:
                    anomaly_type = 'spike'
                elif rand_val < ANOMALY_CONFIG['spike_rate'] + ANOMALY_CONFIG['level_shift_rate']:
                     anomaly_type = 'level_shift'
                elif rand_val < ANOMALY_CONFIG['spike_rate'] + ANOMALY_CONFIG['level_shift_rate'] + ANOMALY_CONFIG['drift_rate']:
                     anomaly_type = 'drift'
                elif rand_val < ANOMALY_CONFIG['spike_rate'] + ANOMALY_CONFIG['level_shift_rate'] + ANOMALY_CONFIG['drift_rate'] + ANOMALY_CONFIG['dropout_rate']:
                     anomaly_type = 'dropout'

                if anomaly_type:
                    start_index = i
                    if anomaly_type == 'spike':
                        end_index = i # Spikes are instantaneous
                        magnitude = ANOMALY_CONFIG['spike_magnitude_factor'] * std_dev * random.choice([-1, 1])
                        df_anomalous.loc[timestamp, sensor] += magnitude
                        anomaly_log.append({
                            'sensor': sensor, 'type': anomaly_type,
                            'start_index': start_index, 'end_index': end_index,
                            'start_time': str(timestamp), 'end_time': str(timestamp),
                            'magnitude': magnitude
                        })
                    else:
                        duration = random.randint(min_duration, max_duration)
                        end_index = min(i + duration, N_SAMPLES - 1)

                        active_anomalies[sensor] = {'type': anomaly_type, 'start_index': start_index, 'end_index': end_index}
                        log_entry = {
                            'sensor': sensor, 'type': anomaly_type,
                            'start_index': start_index, 'end_index': end_index,
                            'start_time': str(timestamp), 'end_time': str(df_anomalous.index[end_index]),
                        }

                        if anomaly_type == 'level_shift':
                            magnitude = ANOMALY_CONFIG['level_shift_magnitude_factor'] * std_dev * random.choice([-1, 1])
                            active_anomalies[sensor]['magnitude'] = magnitude
                            log_entry['magnitude'] = magnitude
                        elif anomaly_type == 'drift':
                            slope = ANOMALY_CONFIG['drift_slope_factor'] * std_dev * random.choice([-1, 1]) / SAMPLING_RATE # Per step slope
                            active_anomalies[sensor]['slope'] = slope
                            log_entry['slope'] = slope * SAMPLING_RATE # Report per second
                        elif anomaly_type == 'dropout':
                            dropout_value = 0.0 # Or NaN, or sensor floor value
                            active_anomalies[sensor]['value'] = dropout_value
                            log_entry['value'] = dropout_value

                        anomaly_log.append(log_entry)

            # Apply effect of the active anomaly for this step
            if active_anomalies[sensor]:
                anomaly_info = active_anomalies[sensor]
                anomaly_type = anomaly_info['type']

                if anomaly_type == 'level_shift':
                    df_anomalous.loc[timestamp, sensor] += anomaly_info['magnitude']
                elif anomaly_type == 'drift':
                    time_in_drift = i - anomaly_info['start_index']
                    df_anomalous.loc[timestamp, sensor] += anomaly_info['slope'] * time_in_drift
                elif anomaly_type == 'dropout':
                    df_anomalous.loc[timestamp, sensor] = anomaly_info['value']

    print(f"Injected {len(anomaly_log)} anomalies.")
    # Simple interpretation for layman
    num_spikes = sum(1 for a in anomaly_log if a['type'] == 'spike')
    num_shifts = sum(1 for a in anomaly_log if a['type'] == 'level_shift')
    num_drifts = sum(1 for a in anomaly_log if a['type'] == 'drift')
    num_dropouts = sum(1 for a in anomaly_log if a['type'] == 'dropout')
    print(f"Breakdown: Spikes={num_spikes}, Level Shifts={num_shifts}, Drifts={num_drifts}, Dropouts={num_dropouts}")
    print("Anomaly injection complete.")
    return df_anomalous, anomaly_log


if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # --- Generate Normal Data ---
    df_normal, _ = generate_baseline_data()
    # Save normal data (useful for training the detector)
    df_normal.to_csv(NORMAL_DATA_PATH)
    print(f"Normal operation data saved to {NORMAL_DATA_PATH}")

    # --- Generate Anomalous Data ---
    df_anomalous, anomaly_log = inject_anomalies(df_normal)
    # Save anomalous data (useful for testing the detector)
    df_anomalous.to_csv(ANOMALOUS_DATA_PATH)
    print(f"Anomalous data saved to {ANOMALOUS_DATA_PATH}")

    # Save anomaly log
    with open(ANOMALY_LOG_PATH, 'w') as f:
        json.dump(anomaly_log, f, indent=4)
    print(f"Anomaly log saved to {ANOMALY_LOG_PATH}")

    # --- Interpretation for Layman ---
    print("\n--- Data Generation Summary ---")
    print("We have created two sets of simulated sensor readings:")
    print(f"1. Normal Data ({NORMAL_DATA_PATH}): Represents the system running smoothly.")
    print(f"   This data will be used to teach our 'detective' what normal looks like.")
    print(f"2. Anomalous Data ({ANOMALOUS_DATA_PATH}): Looks mostly normal, but includes hidden problems (anomalies).")
    print(f"   This data will be used to test if our 'detective' can spot the problems.")
    print(f"3. Anomaly Log ({ANOMALY_LOG_PATH}): A 'cheat sheet' listing all the problems we added.")
    print(f"   We use this later to check the detective's accuracy.")
    print("The data includes sensors like Temperature, Pressure, etc., with realistic noise and correlations.")

    # Optional: Plot generated data for visual inspection
    import matplotlib.pyplot as plt
    print("\nPlotting generated anomalous data (first 1000 points)...")
    plot_df = df_anomalous.head(1000) # Plot only a subset for clarity
    axs = plot_df.plot(subplots=True, figsize=(12, 8), title="Sample of Generated Anomalous Data")
    # Mark anomalies in the plot (simple version: mark start points)
    points_to_plot = set()
    for anomaly in anomaly_log:
        if anomaly['start_index'] < 1000:
            points_to_plot.add((anomaly['start_index'], anomaly['sensor']))
        # Optionally mark end points too for non-spikes if end_index < 1000

    for ax, sensor in zip(axs, plot_df.columns):
         for idx, s in points_to_plot:
              if s == sensor:
                  ax.axvline(plot_df.index[idx], color='r', linestyle='--', alpha=0.7, label='Anomaly Start' if idx == min(p[0] for p in points_to_plot if p[1]==sensor) else "") # Avoid duplicate labels
         if any(s == sensor for _, s in points_to_plot):
             ax.legend()
         ax.set_ylabel(sensor)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()