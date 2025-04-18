# evaluation.py

"""
Evaluation Module for Anomaly Detection Performance

Disclaimer for Layman:
This script acts like a judge comparing the 'detective's' findings (our model's
predictions) against the 'cheat sheet' (the ground truth anomaly log we created).
The goal is to measure how well the detective did its job. We use standard scores:

- Precision: When the detective shouts 'Anomaly!', how often is it correct?
             (High precision = fewer false alarms)
- Recall: Of all the real anomalies that happened, how many did the detective find?
          (High recall = fewer missed incidents)
- F1-Score: A single score balancing Precision and Recall. Good for overall assessment.

We need to be careful comparing time-based events. If the detective flags an anomaly
just slightly before or after it truly started, we might still consider it a success.
This script handles that logic.
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

from config import ANOMALY_LOG_PATH, EVALUATION_BUFFER, ANOMALOUS_DATA_PATH

def load_anomaly_log(log_path=ANOMALY_LOG_PATH):
    """ Loads the anomaly log from a JSON file. """
    try:
        with open(log_path, 'r') as f:
            anomaly_log = json.load(f)
        return anomaly_log
    except FileNotFoundError:
        print(f"Error: Anomaly log file not found at {log_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {log_path}")
        return None

def create_ground_truth_labels(timestamps, anomaly_log, buffer=EVALUATION_BUFFER):
    """
    Creates a point-wise ground truth label array from the event-based anomaly log.

    Args:
        timestamps (pd.DatetimeIndex): Timestamps of the data series.
        anomaly_log (list): The loaded anomaly log.
        buffer (int): Number of samples before/after anomaly to still count detection.

    Returns:
        np.ndarray: Array of ground truth labels (1 for normal, -1 for anomaly).
    """
    n_samples = len(timestamps)
    # Start with all points labeled as normal (1)
    ground_truth = np.ones(n_samples, dtype=int)

    if not anomaly_log:
        print("Warning: Anomaly log is empty or could not be loaded. Ground truth assumes no anomalies.")
        return ground_truth

    # Map timestamps to indices for faster lookup
    ts_to_idx = pd.Series(index=timestamps, data=np.arange(n_samples))

    for anomaly in anomaly_log:
        try:
            # Get start and end index from log (use actual index, not just timestamp string)
            start_idx = anomaly['start_index']
            end_idx = anomaly['end_index']

            # Apply buffer: Extend the anomaly window slightly
            buffer_start_idx = max(0, start_idx - buffer)
            buffer_end_idx = min(n_samples - 1, end_idx + buffer)

            # Mark the buffered interval as anomalous (-1)
            # Slicing includes start, excludes end, so use +1 for end index
            ground_truth[buffer_start_idx : buffer_end_idx + 1] = -1

        except KeyError:
            print(f"Warning: Skipping anomaly due to missing index keys: {anomaly}")
            continue
        except Exception as e:
            print(f"Warning: Error processing anomaly {anomaly}: {e}")
            continue

    num_true_anomalies = (ground_truth == -1).sum()
    print(f"Created ground truth labels: {num_true_anomalies} points marked as anomalous (including buffer).")
    return ground_truth

def evaluate_predictions(y_true, y_pred):
    """
    Calculates and prints evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth labels (1 normal, -1 anomaly).
        y_pred (np.ndarray): Predicted labels (1 normal, -1 anomaly).
    """
    print("\n--- Anomaly Detection Evaluation ---")

    # Ensure labels are in a consistent format if needed (e.g., 0/1 or -1/1)
    # Scikit-learn metrics often expect 0/1, let's convert: 1 -> 0 (normal), -1 -> 1 (anomaly)
    y_true_binary = np.where(y_true == -1, 1, 0)
    y_pred_binary = np.where(y_pred == -1, 1, 0)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, pos_label=1, average='binary', zero_division=0
    )
    # pos_label=1 means we are calculating metrics for the 'anomaly' class

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]) # Labels: 0=Normal, 1=Anomaly

    print("\n--- Interpretation for Layman ---")
    print("We compared the detector's flags against the actual problems (ground truth).")
    print(f"Precision: {precision:.3f}")
    print("  > Of all the times the detector flagged an anomaly, {:.1f}% were actual problems.".format(precision * 100))
    print("  > A high precision means the detector rarely cries wolf (few false alarms).")
    print(f"Recall: {recall:.3f}")
    print("  > Of all the actual problems that occurred, the detector found {:.1f}%.".format(recall * 100))
    print("  > A high recall means the detector rarely misses real issues.")
    print(f"F1-Score: {f1:.3f}")
    print("  > This is a combined score balancing Precision and Recall (higher is better).")

    print("\nConfusion Matrix:")
    print("            Predicted Normal | Predicted Anomaly")
    print("-------------------------------------------------")
    print(f"Actual Normal | {cm[0, 0]:<15} | {cm[0, 1]:<17} (False Alarms)")
    print(f"Actual Anomaly| {cm[1, 0]:<15} (Missed) | {cm[1, 1]:<17} (Correct Detections)")
    print("-------------------------------------------------")
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (Correctly ignored normal): {tn}")
    print(f"False Positives (Falsely flagged normal): {fp}")
    print(f"False Negatives (Missed anomalies): {fn}")
    print(f"True Positives (Correctly detected anomalies): {tp}")

    print("\nDetailed Classification Report:")
    # Target names: 0 -> Normal, 1 -> Anomaly
    print(classification_report(y_true_binary, y_pred_binary, target_names=['Normal (1)', 'Anomaly (-1)'], zero_division=0))


if __name__ == "__main__":
    print("Running Evaluation Module Example...")

    # 1. Load the anomalous data (which should include predictions if anomaly_detector.py was run)
    # For this example, let's assume 'anomaly_detector.py' was just run and saved predictions.
    # In a real app, predictions would come directly from the detector.
    try:
        # We need the data with predictions made in the previous step
        # Let's reload the output from the anomaly_detector example
        df_test_with_preds = pd.read_csv(ANOMALOUS_DATA_PATH, index_col='Timestamp', parse_dates=True)
        # We need to re-run prediction here if the file doesn't have the columns
        # For simplicity, assume anomaly_detector.py was run and we have predictions

        # *** Re-run prediction to ensure columns exist for standalone execution ***
        from anomaly_detection import AnomalyDetector # Import here for standalone run
        from config import SENSOR_CONFIG # Import here for standalone run
        detector = AnomalyDetector()
        try:
            detector.load_model() # Load the saved model
            predictions, _ = detector.predict(df_test_with_preds[list(SENSOR_CONFIG.keys())])
            df_test_with_preds['Anomaly_Predicted'] = predictions
        except Exception as e:
             print(f"Could not load/run model to get predictions: {e}")
             print("Make sure anomaly_detector.py was run successfully first.")
             exit()
        # *** End re-run prediction section ***

        if 'Anomaly_Predicted' not in df_test_with_preds.columns:
             print("Error: 'Anomaly_Predicted' column not found in data.")
             print("Please ensure anomaly_detector.py was run and predictions were made.")
             exit()

        y_pred = df_test_with_preds['Anomaly_Predicted'].values
        timestamps = df_test_with_preds.index
        print(f"Loaded test data with predictions. Total points: {len(y_pred)}")

    except FileNotFoundError:
        print(f"Error: Anomalous data file {ANOMALOUS_DATA_PATH} not found.")
        print("Please run data_generator.py and anomaly_detector.py first.")
        exit()

    # 2. Load the anomaly log (ground truth)
    anomaly_log = load_anomaly_log()
    if anomaly_log is None:
        exit()

    # 3. Create ground truth labels
    y_true = create_ground_truth_labels(timestamps, anomaly_log)

    # 4. Evaluate
    evaluate_predictions(y_true, y_pred)

    print("\n--- Evaluation Summary ---")
    print("The evaluation scores tell us how reliable our anomaly detector is on this specific test data.")
    print("In a real project, we would use these scores to:")
    print("  - Compare different detection algorithms.")
    print("  - Tune the detector's settings (like the 'contamination' parameter) for better performance.")
    print("  - Understand the trade-offs (e.g., detecting more anomalies might cause more false alarms).")