# app.py (Corrected Version - Fix for inconsistent lengths)

"""
Streamlit Application for Real-Time Anomaly Detection Simulation (Enhanced)

Disclaimer for Layman:
This script creates an interactive web dashboard to visualize our anomaly
detection system. We've improved it based on feedback! Instead of just a
potentially chaotic live view, we now have:
1.  **Live Simulation Tab:** Watch sensor data and detections happen in real-time.
2.  **Detected Anomalies Tab:** A running list of all anomalies flagged by the
    system, so you don't miss any. You can check this during or after the simulation.
3.  **Overall Report Tab:** After the simulation finishes, see the final accuracy
    scores (Precision, Recall, F1) comparing detections to the ground truth.
4.  **Download Button:** Get a CSV file listing all the detected anomalies for
    your records.
5.  **Speed Control:** Adjust how fast the simulation runs.

This makes the tool easier to use and provides clear, persistent results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from collections import deque
import io # For creating in-memory text file for download
import traceback # For detailed error reporting

# Import project modules
from config import (
    APP_TITLE, ANOMALOUS_DATA_PATH, ANOMALY_LOG_PATH, MODEL_PATH,
    SIMULATION_SPEED as DEFAULT_SIMULATION_SPEED, # Use default from config
    PLOT_WINDOW_SIZE, SENSOR_CONFIG, EVALUATION_BUFFER
)
# Check if these modules exist and handle potential import errors gracefully
try:
    from anomaly_detector import AnomalyDetector
except ImportError:
    st.error("Error: `anomaly_detector.py` not found. Please ensure it's in the same directory.")
    st.stop()
try:
    from evaluation import load_anomaly_log, create_ground_truth_labels, evaluate_predictions
except ImportError:
    st.error("Error: `evaluation.py` not found. Please ensure it's in the same directory.")
    st.stop()


# --- Helper Functions ---

@st.cache_resource # Cache the loaded detector
def load_detector():
    """Loads the trained anomaly detection model."""
    try:
        detector = AnomalyDetector()
        detector.load_model(MODEL_PATH)
        return detector
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found.")
        st.error("Please run `python anomaly_detector.py` to train and save the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading the anomaly detector model: {e}")
        return None

@st.cache_data # Cache the loaded data
def load_data():
    """Loads the anomalous test data and anomaly log."""
    try:
        df = pd.read_csv(ANOMALOUS_DATA_PATH, index_col='Timestamp', parse_dates=True)
        # Ensure correct column order expected by the model/scaler
        if not all(col in df.columns for col in SENSOR_CONFIG.keys()):
             st.error("Error: Loaded data is missing expected sensor columns. Did the config change after data generation?")
             return None, None, None
        df = df[list(SENSOR_CONFIG.keys())] # Reorder/select columns just in case
    except FileNotFoundError:
        st.error(f"Error: Anomalous data file not found at {ANOMALOUS_DATA_PATH}")
        st.error("Please run `python data_generator.py` first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

    anomaly_log = load_anomaly_log(ANOMALY_LOG_PATH)
    ground_truth = None
    if anomaly_log is None:
        st.warning("Could not load anomaly log. Ground truth evaluation will not be available.")
    else:
        try:
            ground_truth = create_ground_truth_labels(df.index, anomaly_log, buffer=EVALUATION_BUFFER)
        except Exception as e:
             st.error(f"Error creating ground truth labels: {e}")
             st.warning("Ground truth evaluation might be unavailable.")

    return df, anomaly_log, ground_truth

def convert_df_to_csv(df_to_convert):
   """Converts a DataFrame to CSV bytes for download."""
   output = io.StringIO()
   # Ensure Timestamp is a column if it's the index, otherwise it might not write properly
   if isinstance(df_to_convert.index, pd.DatetimeIndex):
        df_to_convert.index.name = 'Timestamp' # Ensure index has a name
        df_to_convert.to_csv(output, index=True)
   else:
        df_to_convert.to_csv(output, index=False) # Don't write default numerical index if Timestamp is a column
   return output.getvalue().encode('utf-8')


# --- Streamlit App Layout ---

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.markdown("""
**Disclaimer:** This application simulates real-time anomaly detection on pre-generated data.
Use the tabs below to navigate between the live simulation, the log of detected anomalies,
and the final evaluation report.
""")

# --- Load Model and Data ---
# Placeholders for status updates during loading
load_status = st.empty()
load_status.info("Loading anomaly detection model...")
detector = load_detector()
load_status.info("Loading telemetry data...")
df, anomaly_log, ground_truth = load_data()
load_status.empty() # Clear status message

if detector is None or df is None:
    st.error("Application cannot start due to loading errors. Please check messages above.")
    st.stop() # Stop execution if model or data failed to load

# --- Initialize Session State ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
    st.session_state.running = False
    # Deques for efficient rolling window plotting data
    st.session_state.plot_data = {col: deque(maxlen=PLOT_WINDOW_SIZE) for col in df.columns}
    st.session_state.plot_timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
    st.session_state.predictions_history = deque(maxlen=PLOT_WINDOW_SIZE)
    # Store detected anomalies persistently
    st.session_state.detected_anomalies_list = []
    # Store all predictions for final eval
    st.session_state.all_predictions = []
    # Store evaluation results
    st.session_state.evaluation_results = None
    st.session_state.simulation_speed = DEFAULT_SIMULATION_SPEED

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")
start_button = st.sidebar.button("Start/Resume", key="start", use_container_width=True)
pause_button = st.sidebar.button("Pause", key="pause", use_container_width=True)
reset_button = st.sidebar.button("Reset", key="reset", use_container_width=True)

st.sidebar.header("Settings")
st.session_state.simulation_speed = st.sidebar.slider(
    "Simulation Speed (delay per step, seconds)",
    min_value=0.0, max_value=1.0,
    value=st.session_state.simulation_speed,
    step=0.01,
    format="%.2f"
)

if start_button:
    st.session_state.running = True
if pause_button:
    st.session_state.running = False
if reset_button:
    # Clear all relevant session state variables
    st.session_state.current_step = 0
    st.session_state.running = False
    st.session_state.plot_data = {col: deque(maxlen=PLOT_WINDOW_SIZE) for col in df.columns}
    st.session_state.plot_timestamps = deque(maxlen=PLOT_WINDOW_SIZE)
    st.session_state.predictions_history = deque(maxlen=PLOT_WINDOW_SIZE)
    st.session_state.detected_anomalies_list = []
    st.session_state.all_predictions = []
    st.session_state.evaluation_results = None
    # No need to reset simulation_speed unless desired
    st.rerun() # Force rerun after reset to clear UI elements

# --- Main Application Tabs ---
tab_live, tab_detected, tab_report = st.tabs(["📊 Live Simulation", "🚨 Detected Anomalies Log", "📜 Overall Report"])

with tab_live:
    st.header("Live Simulation View")
    # Use containers to manage content updates more cleanly
    live_status_placeholder = st.container()
    charts_placeholder = st.container()

with tab_detected:
    st.header("Log of Detected Anomalies")
    st.markdown("This table updates whenever the model flags a data point as an anomaly (-1).")
    detected_log_placeholder = st.container() # Placeholder for the dataframe and button

with tab_report:
    st.header("Overall Evaluation Report")
    st.markdown("This report is generated after the simulation completes or is manually stopped.")
    report_placeholder = st.container()


# --- Real-Time Simulation Logic ---
if st.session_state.running and st.session_state.current_step < len(df):
    # Get current data point (as a DataFrame to keep structure)
    current_data_point_df = df.iloc[[st.session_state.current_step]]
    current_data_point_series = current_data_point_df.iloc[0] # As Series for easier access
    current_timestamp = current_data_point_series.name # Index is timestamp

    # Make prediction
    prediction, score = detector.predict(current_data_point_df)
    prediction = prediction[0] # Get the single prediction value
    score = score[0] # Get the single score value
    st.session_state.all_predictions.append(prediction)

    # --- Update Live Tab Status ---
    with live_status_placeholder:
        st.subheader(f"Simulation Step: {st.session_state.current_step + 1} / {len(df)}")
        st.metric("Current Timestamp", str(current_timestamp.round('ms')))

        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if prediction == -1:
                st.warning(f"🚨 Anomaly Detected! Score: {score:.3f}")
                # Simple heuristic for contributing sensors
                scaled_point = detector.scaler.transform(current_data_point_df)[0]
                abs_scaled_deviations = np.abs(scaled_point)
                threshold = 2.0 # Example threshold
                deviating_sensors = [df.columns[i] for i, dev in enumerate(abs_scaled_deviations) if dev > threshold]
                contrib_str = ', '.join(deviating_sensors) if deviating_sensors else 'Combined'
                st.write(f"   > Potential contributing sensors: {contrib_str}")

                # Add to persistent log
                anomaly_record = {
                    'Timestamp': current_timestamp, # Keep as datetime object initially if needed later
                    'Step': st.session_state.current_step + 1,
                    'Anomaly Score': f"{score:.3f}",
                    'Contributing Sensors (Heuristic)': contrib_str,
                }
                # Add actual sensor values at the time of anomaly
                for sensor, value in current_data_point_series.items():
                    anomaly_record[f'Value_{sensor}'] = f"{value:.2f}"

                st.session_state.detected_anomalies_list.append(anomaly_record)
            else:
                st.success(f"Status: Normal. Score: {score:.3f}")

        with status_col2:
            # Show ground truth comparison if available
            if ground_truth is not None:
                # Check if current_step is within bounds of ground_truth
                if st.session_state.current_step < len(ground_truth):
                    true_label = ground_truth[st.session_state.current_step]
                    if true_label == -1:
                        st.info("Ground Truth: Known Anomaly")
                        if prediction == -1: st.success("Detection: Correct (TP)")
                        else: st.error("Detection: Missed (FN)")
                    else:
                        st.info("Ground Truth: Normal")
                        if prediction == -1: st.warning("Detection: False Alarm (FP)")
                        else: st.success("Detection: Correct (TN)")
                else:
                    st.warning("Ground truth index out of bounds.") # Should not happen if logic is correct
            else:
                st.info("Ground Truth: N/A")

    # Update plot data (deques)
    st.session_state.plot_timestamps.append(current_timestamp)
    for col in df.columns:
        st.session_state.plot_data[col].append(current_data_point_series[col])
    st.session_state.predictions_history.append(prediction)

    # --- Update Live Tab Plots ---
    with charts_placeholder:
        plot_df_live = pd.DataFrame(st.session_state.plot_data, index=list(st.session_state.plot_timestamps))
        # No 'Anomaly' column needed if using simpler line charts

        st.subheader("Live Sensor Readings")
        plot_cols = st.columns(min(len(df.columns), 2)) # Max 2 plots per row
        col_idx = 0
        for sensor in df.columns:
            with plot_cols[col_idx % len(plot_cols)]:
                st.markdown(f"**{sensor}**")
                # Plot only this sensor's data using the deque data directly
                sensor_data_df = pd.DataFrame({sensor: list(st.session_state.plot_data[sensor])}, index=list(st.session_state.plot_timestamps))
                if not sensor_data_df.empty:
                    st.line_chart(sensor_data_df, height=200, use_container_width=True)
            col_idx += 1

    # --- Update Detected Anomalies Tab ---
    with detected_log_placeholder: # Update content within the container
        if st.session_state.detected_anomalies_list:
            # Create DataFrame ensuring Timestamp becomes the index
            detected_df_display = pd.DataFrame(st.session_state.detected_anomalies_list)
            if 'Timestamp' in detected_df_display.columns:
                 detected_df_display = detected_df_display.set_index('Timestamp')
            st.dataframe(detected_df_display.tail(20)) # Show last 20 detected
            # Prepare data for download button
            csv_data = convert_df_to_csv(detected_df_display) # Use the full dataframe
            st.download_button(
                label="Download Detected Anomalies Log as CSV",
                data=csv_data,
                file_name='detected_anomalies_log.csv',
                mime='text/csv',
                key='download_detected' # Added key for stability
            )
        else:
            st.info("No anomalies detected yet in this session.")

    # --- Advance Simulation Step ---
    st.session_state.current_step += 1
    # Apply simulation speed delay
    if st.session_state.simulation_speed > 0:
        time.sleep(st.session_state.simulation_speed)
    # Trigger rerun for next step
    st.rerun()

# --- Post-Simulation Logic ---
# This block executes when simulation is paused OR finishes normally
elif not st.session_state.running and st.session_state.current_step > 0:
    # Show appropriate message in live tab status area
    with live_status_placeholder:
        if st.session_state.current_step == len(df):
            st.success("✅ Simulation Finished!")
        else:
            st.info(f"⏸️ Simulation paused at step {st.session_state.current_step}. Press Start/Resume to continue.")

    # Update the detected anomalies tab one last time to show the full log
    with detected_log_placeholder:
        if st.session_state.detected_anomalies_list:
            # Create DataFrame ensuring Timestamp becomes the index
            detected_df_display = pd.DataFrame(st.session_state.detected_anomalies_list)
            if 'Timestamp' in detected_df_display.columns:
                 detected_df_display = detected_df_display.set_index('Timestamp')
            st.dataframe(detected_df_display) # Show all detected when paused/finished
            # Prepare data for download button (needed again here for when paused)
            csv_data = convert_df_to_csv(detected_df_display)
            st.download_button(
                label="Download Detected Anomalies Log as CSV",
                data=csv_data,
                file_name='detected_anomalies_log.csv',
                mime='text/csv',
                key='download_detected_paused' # Different key maybe needed if button is shown in two places
            )
        else:
            st.info("No anomalies detected during the simulation run.")

    # Generate and display the final report in the report tab
    with report_placeholder:
        st.subheader("Overall Evaluation Results")
        # Check if ground truth is available and predictions were made
        if ground_truth is not None and len(st.session_state.all_predictions) > 0:
             # Calculate results only if they haven't been calculated yet for this run
            if st.session_state.evaluation_results is None:
                # !--- CORRECTED SECTION ---!
                # Get the list of predictions actually made
                y_pred_list = st.session_state.all_predictions
                num_predictions = len(y_pred_list) # Get the actual count

                # Convert predictions to numpy array
                y_pred = np.array(y_pred_list)

                # Slice ground_truth based on the *number of predictions made*
                # This ensures y_true and y_pred have the same length
                y_true = ground_truth[:num_predictions]
                # !--- END CORRECTED SECTION ---!

                st.write(f"Evaluating based on {num_predictions} simulation steps.")
                # Use try-except block for robustness during calculation
                try:
                    eval_output_buffer = io.StringIO()
                    from contextlib import redirect_stdout
                    with redirect_stdout(eval_output_buffer):
                        # Ensure evaluate_predictions can handle the data shapes
                        if len(y_true) == len(y_pred): # Final sanity check
                             evaluate_predictions(y_true, y_pred)
                        else:
                             # This should not happen with the fix, but good to have a fallback
                             st.error(f"Internal Error: Mismatch persists - y_true len {len(y_true)}, y_pred len {len(y_pred)}")
                             eval_output_buffer.write("Evaluation failed due to length mismatch.")

                    # Store the captured output in session state
                    st.session_state.evaluation_results = eval_output_buffer.getvalue()
                except Exception as e:
                    st.error(f"ERROR during evaluation calculation: {e}")
                    st.error(f"Traceback:\n{traceback.format_exc()}")
                    st.session_state.evaluation_results = "Evaluation failed. See error details above."

            # Display the stored evaluation results (either the captured output or error message)
            if st.session_state.evaluation_results:
                 st.text(st.session_state.evaluation_results)
            else:
                 st.info("Evaluation results are being calculated or encountered an issue.")


            st.markdown("---")
            st.markdown("#### Final Summary for Layman:")
            st.markdown("This report compares the detector's performance against the known ground truth (if available) up to the point the simulation ran.")
            st.markdown("* **Precision:** How often was an alert a real problem?")
            st.markdown("* **Recall:** How many of the real problems did the system find?")
            st.markdown("* **F1-Score:** Combined accuracy score.")
            st.markdown("* **Confusion Matrix:** Breaks down correct/incorrect detections and false alarms.")

        elif ground_truth is None:
            st.warning("Ground truth data was not available, so a full evaluation cannot be performed.")
            st.info(f"The model flagged {len(st.session_state.detected_anomalies_list)} data points as anomalous during the run.")
        else: # Case where simulation hasn't run yet (current_step is 0 but not running)
            st.info("Run the simulation to generate the final report.")

# --- Initial State Logic ---
# This block executes only when the app first loads and simulation hasn't started
elif st.session_state.current_step == 0:
     with live_status_placeholder:
          st.info("Simulation ready. Press Start/Resume in the sidebar.")
     with detected_log_placeholder:
          st.info("Detected anomalies will appear here during the simulation.")
     with report_placeholder:
          st.info("Run the simulation to generate the final report.")

# Optional: Display session state for debugging (useful during development)
# st.sidebar.header("Debug Info")
# st.sidebar.write(st.session_state)