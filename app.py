# app.py

"""
Streamlit Application for Real-Time Anomaly Detection Simulation

Disclaimer for Layman:
This script creates an interactive web dashboard (using Streamlit) to visualize
our anomaly detection system in action. It simulates receiving sensor data
point-by-point, just like a real system would. For each new data point:
1. It shows the sensor readings on updating charts.
2. It feeds the data to our trained 'detective' (the Isolation Forest model).
3. It highlights points that the detective flags as anomalies on the charts.
4. It displays alerts when anomalies are detected.
5. It shows the overall accuracy scores (Precision, Recall, F1) calculated live.

This helps us see how the system performs over time and understand its behavior
in a more intuitive way than just looking at static data files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from collections import deque

# Import project modules
from config import (
    APP_TITLE, ANOMALOUS_DATA_PATH, ANOMALY_LOG_PATH, MODEL_PATH,
    SIMULATION_SPEED, PLOT_WINDOW_SIZE, SENSOR_CONFIG, EVALUATION_BUFFER
)
from anomaly_detection import AnomalyDetector # Assuming anomaly_detector.py is in the same dir
from evaluation import load_anomaly_log, create_ground_truth_labels, evaluate_predictions # Assuming evaluation.py is in the same dir

# --- Helper Functions ---

@st.cache_resource # Cache the loaded detector to avoid reloading on every interaction
def load_detector():
    """Loads the trained anomaly detection model."""
    try:
        detector = AnomalyDetector()
        detector.load_model(MODEL_PATH)
        return detector
    except Exception as e:
        st.error(f"Error loading the anomaly detector model: {e}")
        st.error(f"Ensure '{MODEL_PATH}' exists. Run anomaly_detector.py to train and save the model.")
        return None

@st.cache_data # Cache the loaded data to avoid reloading
def load_data():
    """Loads the anomalous test data and anomaly log."""
    try:
        df = pd.read_csv(ANOMALOUS_DATA_PATH, index_col='Timestamp', parse_dates=True)
        # Ensure correct column order expected by the model/scaler
        df = df[list(SENSOR_CONFIG.keys())]
    except FileNotFoundError:
        st.error(f"Error: Anomalous data file not found at {ANOMALOUS_DATA_PATH}")
        st.error("Please run data_generator.py first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

    anomaly_log = load_anomaly_log(ANOMALY_LOG_PATH)
    if anomaly_log is None:
        st.warning("Could not load anomaly log. Ground truth evaluation will not be available.")
        ground_truth = None
    else:
        # Generate ground truth labels for the entire dataset once
        ground_truth = create_ground_truth_labels(df.index, anomaly_log, buffer=EVALUATION_BUFFER)

    return df, anomaly_log, ground_truth

# --- Streamlit App Layout ---

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.markdown("""
**Disclaimer:** This application simulates real-time anomaly detection on pre-generated data.
It demonstrates how such a system might work by processing data step-by-step, applying
a trained machine learning model (Isolation Forest), and visualizing the results.
""")

# --- Load Model and Data ---
detector = load_detector()
df, anomaly_log, ground_truth = load_data()

if detector is None or df is None:
    st.stop() # Stop execution if model or data failed to load

# --- Initialize Session State ---
# Streamlit reruns the script on interaction. Use session state to preserve variables.
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
    st.session_state.running = False
    # Use deques for efficient rolling window plotting data
    st.session_state.plot_data = {col: deque(maxlen=PLOT_WINDOW_SIZE) for col in df.columns}
    st.session_state.plot_timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
    st.session_state.predictions_history = deque(maxlen=PLOT_WINDOW_SIZE)
    st.session_state.ground_truth_history = deque(maxlen=PLOT_WINDOW_SIZE) if ground_truth is not None else None
    st.session_state.all_predictions = [] # Store all predictions for final evaluation
    st.session_state.detected_anomaly_sensors = set() # Track sensors with recent anomalies


# --- Simulation Controls ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start/Resume", key="start"):
        st.session_state.running = True
with col2:
    if st.button("Pause", key="pause"):
        st.session_state.running = False
with col3:
    if st.button("Reset", key="reset"):
        st.session_state.current_step = 0
        st.session_state.running = False
        st.session_state.plot_data = {col: deque(maxlen=PLOT_WINDOW_SIZE) for col in df.columns}
        st.session_state.plot_timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
        st.session_state.predictions_history = deque(maxlen=PLOT_WINDOW_SIZE)
        st.session_state.ground_truth_history = deque(maxlen=PLOT_WINDOW_SIZE) if ground_truth is not None else None
        st.session_state.all_predictions = []
        st.session_state.detected_anomaly_sensors = set()


# --- Real-Time Simulation Loop ---
if st.session_state.running and st.session_state.current_step < len(df):
    # Get current data point (as a DataFrame to keep structure)
    current_data_point = df.iloc[[st.session_state.current_step]]
    current_timestamp = current_data_point.index[0]

    # Make prediction
    prediction, score = detector.predict(current_data_point)
    prediction = prediction[0] # Get the single prediction value
    score = score[0] # Get the single score value
    st.session_state.all_predictions.append(prediction)

    # Update plot data
    st.session_state.plot_timestamps.append(current_timestamp)
    for col in df.columns:
        st.session_state.plot_data[col].append(current_data_point[col].iloc[0])
    st.session_state.predictions_history.append(prediction)
    if st.session_state.ground_truth_history is not None:
         st.session_state.ground_truth_history.append(ground_truth[st.session_state.current_step])

    # --- Display Current Status ---
    st.subheader(f"Simulation Step: {st.session_state.current_step + 1} / {len(df)}")
    st.metric("Current Timestamp", str(current_timestamp.round('ms'))) # Nicer formatting

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if prediction == -1:
            st.warning(f"🚨 Anomaly Detected! Score: {score:.3f}")
            # Find which sensors might be contributing (heuristic: large deviation from mean in scaled space)
            # This is a simple heuristic, more sophisticated attribution methods exist.
            scaled_point = detector.scaler.transform(current_data_point)[0]
            abs_scaled_deviations = np.abs(scaled_point)
            deviating_sensors = [df.columns[i] for i, dev in enumerate(abs_scaled_deviations) if dev > 2.0] # Example threshold: 2 std devs
            if deviating_sensors:
                 st.write(f"   > Potential contributing sensors: {', '.join(deviating_sensors)}")
                 st.session_state.detected_anomaly_sensors.update(deviating_sensors) # Track sensors involved
            else:
                 st.write("   > Anomaly detected based on combined sensor values.")

        else:
            st.success(f"Status: Normal. Score: {score:.3f}")
            # Clear sensors involved if normal? Or let them persist for a while? Let's clear.
            # st.session_state.detected_anomaly_sensors.clear()

    with status_col2:
        if ground_truth is not None:
            true_label = ground_truth[st.session_state.current_step]
            if true_label == -1:
                st.info("Ground Truth: This point IS part of a known anomaly.")
                if prediction == -1:
                    st.success("Detection Status: Correct Detection (True Positive)")
                else:
                    st.error("Detection Status: Missed Anomaly (False Negative)")
            else:
                st.info("Ground Truth: This point is Normal.")
                if prediction == -1:
                    st.warning("Detection Status: False Alarm (False Positive)")
                else:
                    st.success("Detection Status: Correct Normal (True Negative)")
        else:
            st.info("Ground Truth: Not available.")

    # --- Live Plotting ---
    st.subheader("Live Sensor Readings")
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(st.session_state.plot_data, index=list(st.session_state.plot_timestamps))
    plot_df['Anomaly'] = list(st.session_state.predictions_history)

    # Use Streamlit's built-in line chart, consider alternatives for more customization
    # Create placeholder charts
    chart_placeholders = {col: st.empty() for col in df.columns}

    for col in df.columns:
        # Create df for this sensor's plot
        sensor_plot_df = plot_df[[col, 'Anomaly']].copy()
        sensor_plot_df['color'] = np.where(sensor_plot_df['Anomaly'] == -1, 'Anomaly', 'Normal')
        sensor_plot_df['size'] = np.where(sensor_plot_df['Anomaly'] == -1, 50, 10) # Make anomalies bigger points (if using scatter)

        # Using st.line_chart (simpler, less control over color/markers for anomalies)
        # Need to potentially plot normal and anomaly points separately if using line_chart
        # Let's filter data and plot separately
        normal_points = sensor_plot_df[sensor_plot_df['Anomaly'] == 1][[col]]
        anomaly_points = sensor_plot_df[sensor_plot_df['Anomaly'] == -1][[col]]

        # Simple placeholder update (st.line_chart doesn't easily support conditional coloring)
        # Consider using Altair or Plotly for better visualization here if needed.
        # For now, just plot the line. Highlighting requires more complex plotting.
        chart_placeholders[col].line_chart(sensor_plot_df[[col]], use_container_width=True)

        # A better way with Plotly (if installed: pip install plotly)
        # try:
        #     import plotly.graph_objects as go
        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(x=sensor_plot_df.index, y=sensor_plot_df[col], mode='lines', name='Normal', line=dict(color='blue')))
        #     anomaly_points = sensor_plot_df[sensor_plot_df['Anomaly'] == -1]
        #     fig.add_trace(go.Scatter(x=anomaly_points.index, y=anomaly_points[col], mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
        #     fig.update_layout(title=col, xaxis_title="Timestamp", yaxis_title="Value", height=300, margin=dict(l=20, r=20, t=40, b=20))
        #     chart_placeholders[col].plotly_chart(fig, use_container_width=True)
        # except ImportError:
        #      chart_placeholders[col].line_chart(sensor_plot_df[[col]], use_container_width=True) # Fallback

    # --- Advance Simulation ---
    st.session_state.current_step += 1
    time.sleep(SIMULATION_SPEED) # Control simulation speed
    st.rerun() # Trigger rerun to update the UI

elif st.session_state.current_step >= len(df):
    st.success("Simulation Finished!")
    st.session_state.running = False

    # --- Final Evaluation ---
    if ground_truth is not None and len(st.session_state.all_predictions) == len(ground_truth):
        st.subheader("Overall Evaluation Results")
        # Use evaluate_predictions, but capture print output for Streamlit
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            evaluate_predictions(ground_truth, np.array(st.session_state.all_predictions))
        evaluation_output = f.getvalue()
        st.text(evaluation_output) # Display captured output

        # Layman Summary of Final Results
        st.markdown("---")
        st.markdown("#### Final Summary for Layman:")
        st.markdown("After analyzing the entire simulated dataset:")
        st.markdown("* The **Precision** score tells us how trustworthy the anomaly alerts were (percentage of alerts that were real problems).")
        st.markdown("* The **Recall** score tells us how many of the real problems the system managed to find.")
        st.markdown("* The **F1-Score** gives a single measure of overall accuracy, balancing Precision and Recall.")
        st.markdown("Use these scores to judge the detector's effectiveness for this specific scenario.")

    else:
        st.warning("Could not perform final evaluation. Ground truth unavailable or prediction mismatch.")

# Optional: Display session state for debugging
# st.sidebar.write("Session State:", st.session_state)