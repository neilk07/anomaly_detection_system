# config.py

"""
Configuration File for the Real-Time Anomaly Detection Project

Disclaimer for Layman:
This file acts like a settings panel for our project. Instead of changing
settings inside the main code, we define key parameters here. This makes
it easier to experiment and adjust the simulation or detection behaviour
without digging deep into the logic. Think of it as setting the rules
for the game we're about to play with the data.
"""

# --- Data Generation Parameters ---
SAMPLING_RATE = 10  # Hz (samples per second)
DURATION = 600      # seconds (Total simulation time)
N_SAMPLES = DURATION * SAMPLING_RATE

# Define sensors and their normal operating characteristics
SENSOR_CONFIG = {
    'Temperature': {'mean': 70.0, 'std_dev': 0.5, 'trend': lambda t: 10 * np.sin(2 * np.pi * t / (N_SAMPLES/2))}, # Slow sine wave trend
    'Pressure': {'mean': 500.0, 'std_dev': 5.0, 'trend': lambda t: 50 * ((t > N_SAMPLES // 3) & (t < 2 * N_SAMPLES // 3))}, # Step function trend # Step function trend
    'Vibration': {'mean': 0.1, 'std_dev': 0.05, 'trend': lambda t: 0}, # Relatively stable with noise
    'FuelFlow': {'mean': 100.0, 'std_dev': 1.0, 'trend': lambda t: -20 * (t > N_SAMPLES // 2)} # Drop after halfway
}
# Correlation between sensor noise (adjust as needed for realism)
# Rows/Cols order: Temp, Press, Vib, FuelFlow
CORRELATION_MATRIX = [
    [1.0, 0.6, 0.2, 0.4],  # Temp correlates positively with Pressure and FuelFlow
    [0.6, 1.0, 0.1, 0.5],  # Pressure correlates positively with Temp and FuelFlow
    [0.2, 0.1, 1.0, 0.3],  # Vibration weakly correlates with others
    [0.4, 0.5, 0.3, 1.0]   # FuelFlow correlates positively with Temp and Pressure
]

# --- Anomaly Injection Parameters ---
# Specify types, frequency, and characteristics of anomalies to inject
ANOMALY_CONFIG = {
    'spike_rate': 0.001,  # Probability of a spike per sensor per time step
    'level_shift_rate': 0.0005, # Probability of starting a level shift
    'drift_rate': 0.0003, # Probability of starting a drift
    'dropout_rate': 0.0002, # Probability of starting a sensor dropout
    'anomaly_duration_range': (5 * SAMPLING_RATE, 20 * SAMPLING_RATE), # 5 to 20 seconds
    'spike_magnitude_factor': 5, # Spike is factor * std_dev
    'level_shift_magnitude_factor': 3, # Level shift is factor * std_dev
    'drift_slope_factor': 0.1, # Drift slope relative to std_dev per second
}

# --- Anomaly Detection Parameters ---
IFOREST_N_ESTIMATORS = 100 # Number of trees in the Isolation Forest
IFOREST_CONTAMINATION = 'auto' # Expected proportion of anomalies (can be float like 0.01 or 'auto')
# Note: 'auto' works well generally, but tuning might be needed.
# For real-time simulation, we often use a rolling window or process point-by-point
# Let's use point-by-point for simplicity in this undergrad scope.

# --- Evaluation ---
# How to map anomaly log (start/end) to point-wise ground truth
# Allow a buffer around anomalies for detection to be considered correct
EVALUATION_BUFFER = 2 * SAMPLING_RATE # +/- 2 seconds around the actual anomaly duration

# --- File Paths ---
NORMAL_DATA_PATH = "simulated_normal_data.csv"
ANOMALOUS_DATA_PATH = "simulated_anomalous_data.csv"
ANOMALY_LOG_PATH = "anomaly_log.json"
MODEL_PATH = "anomaly_detector_model.joblib"

# --- Streamlit App Config ---
APP_TITLE = "Real-Time Telemetry Anomaly Detection Demo"
SIMULATION_SPEED = 0.05 # Seconds delay between steps in Streamlit simulation loop
PLOT_WINDOW_SIZE = 200 # How many data points to show in the live plot

# --- Imports needed for lambda functions in SENSOR_CONFIG ---
# This is slightly unusual but keeps config self-contained.
# Ensure NumPy is imported where config is used.
import numpy as np