# anomaly_detector.py (WITH DEBUGGING CODE ADDED)

"""
Anomaly Detection Module using Isolation Forest

Disclaimer for Layman:
This script defines our 'detective' – the anomaly detection model. We're using
a technique called 'Isolation Forest'. Imagine trying to describe an unusual person
in a crowd. You can often 'isolate' them with just a few questions (e.g., "Are they
wearing a bright pink hat?"). Normal people require more questions to single out.
Isolation Forest works similarly with data points. It builds random 'decision trees'
to separate data points. Anomalies, being different, usually get isolated in fewer
steps (closer to the 'root' of the tree).

This script has two main jobs:
1. `train`: Teach the model what 'normal' data looks like using the normal dataset.
2. `predict`: Look at new data points and decide if they look 'normal' or 'anomalous'
   based on how easily they are isolated.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib # For saving/loading the model and scaler
import traceback # For detailed error reporting

from config import (
    NORMAL_DATA_PATH, MODEL_PATH, IFOREST_N_ESTIMATORS, IFOREST_CONTAMINATION, SENSOR_CONFIG, ANOMALOUS_DATA_PATH
)

class AnomalyDetector:
    """ Wraps the Isolation Forest model and the data scaler. """
    def __init__(self, n_estimators=IFOREST_N_ESTIMATORS, contamination=IFOREST_CONTAMINATION, random_state=42):
        """
        Initializes the scaler and the Isolation Forest model.

        Args:
            n_estimators (int): Number of trees in the forest.
            contamination (float or 'auto'): Expected proportion of outliers.
            random_state (int): Seed for reproducibility.
        """
        # Scaler: Makes all sensor data use a similar scale (e.g., mean 0, std dev 1)
        # This is important because Isolation Forest can be sensitive to feature ranges.
        self.scaler = StandardScaler()

        # Isolation Forest Model: The core 'detective'.
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            # behaviour='new' # Use 'new' behavior for contamination from scikit-learn 0.22+
            # n_jobs=-1 # Use all available CPU cores for training/prediction (optional)
        )
        self.is_fitted = False # Flag to check if train() has been called

    def train(self, df_normal):
        """
        Trains the scaler and the Isolation Forest model on normal data.

        Args:
            df_normal (pd.DataFrame): DataFrame containing ONLY normal operating data.
                                      Index should be Timestamp, columns are sensors.
        """
        print("Training anomaly detector...")
        # --- Layman Explanation ---
        print("   Step 1: Learning the 'normal' scale of each sensor.")
        # Fit the scaler on normal data and transform it
        # Fit learns the mean and std dev, transform applies the scaling
        scaled_data = self.scaler.fit_transform(df_normal)
        print("   Sensor data scaled to have zero mean and unit variance.")

        print("   Step 2: Training the Isolation Forest model.")
        print("   The model learns to distinguish typical patterns in the scaled normal data.")
        # Fit the Isolation Forest model on the scaled normal data
        self.model.fit(scaled_data)

        self.is_fitted = True
        print("Anomaly detector training complete.")

        # --- Interpretation for Layman ---
        print("\n--- Training Summary ---")
        print("The 'detective' (Isolation Forest model) has now studied the normal data.")
        print("It learned the typical ranges and relationships between sensors during normal operation.")
        print("It also learned how to measure the 'weirdness' (anomaly score) of any future data point.")

    def predict(self, df_new_data):
        """
        Predicts anomalies in new data points.

        Args:
            df_new_data (pd.DataFrame): DataFrame of new data to check for anomalies.
                                        Must have the same columns as the training data.

        Returns:
            np.ndarray: An array of predictions: 1 for normal, -1 for anomaly.
            np.ndarray: An array of anomaly scores (lower scores are more anomalous).
        """
        if not self.is_fitted:
            raise Exception("Model is not trained yet. Call train() first.")

        # Scale the new data using the *already fitted* scaler
        scaled_data = self.scaler.transform(df_new_data)

        # Predict: Returns 1 for inliers (normal), -1 for outliers (anomalies)
        predictions = self.model.predict(scaled_data)

        # Decision function: Returns the raw anomaly score. Lower = more anomalous.
        # Scikit-learn scores are shifted so negative is outlier, positive is inlier.
        scores = self.model.decision_function(scaled_data)

        return predictions, scores

    def save_model(self, file_path=MODEL_PATH):
        """ Saves the trained scaler and model to disk. """
        if not self.is_fitted:
            raise Exception("Model is not trained yet. Cannot save.")
        print(f"Saving model and scaler to {file_path}...")
        # Save both the scaler and the model in a dictionary
        joblib.dump({'scaler': self.scaler, 'model': self.model}, file_path)
        print("Model saved.")

    def load_model(self, file_path=MODEL_PATH):
        """ Loads a trained scaler and model from disk. """
        print(f"Loading model and scaler from {file_path}...")
        try:
            saved_state = joblib.load(file_path)
            self.scaler = saved_state['scaler']
            self.model = saved_state['model']
            self.is_fitted = True
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {file_path}. Train the model first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


if __name__ == "__main__":
    # --- Example Usage ---
    print("Running Anomaly Detector Module Example...")

    # 1. Load the NORMAL data generated previously
    try:
        # --- START DEBUGGING BLOCK ---
        print(f"--- DEBUG: Attempting to load {NORMAL_DATA_PATH} ---")
        try:
            # Use utf-8-sig encoding to automatically handle potential BOM
            with open(NORMAL_DATA_PATH, 'r', encoding='utf-8-sig') as f:
                header = f.readline().strip() # Read first line, remove leading/trailing whitespace
                print(f"DEBUG: Raw header read from file: '{header}'")
                # Split header by comma, strip whitespace from each part
                cols = [c.strip() for c in header.split(',')]
                print(f"DEBUG: Columns detected after split: {cols}")
                # Check if 'Timestamp' exists exactly in the list of columns
                if 'Timestamp' in cols:
                    print("DEBUG: 'Timestamp' IS in the detected columns list.")
                else:
                    print("DEBUG: 'Timestamp' IS NOT in the detected columns list.")
                    # If not found, check the first column carefully for hidden chars
                    if cols: # Check if cols list is not empty
                         print(f"DEBUG: Comparing 'Timestamp' == '{cols[0]}'? { 'Timestamp' == cols[0] }")
                         print(f"DEBUG: Representation of first detected column: {repr(cols[0])}")
                    else:
                         print("DEBUG: No columns detected after split.")
        except FileNotFoundError:
             print(f"DEBUG: Error - File not found during manual header check: {NORMAL_DATA_PATH}")
             # Re-raise the error so the main exception block catches it
             raise
        except Exception as debug_e:
            print(f"DEBUG: Error during manual header check: {debug_e}")
        print("--- DEBUG: End of manual header check ---")
        # --- END DEBUGGING BLOCK ---

        # Now, the original line that might be failing:
        df_normal_train = pd.read_csv(NORMAL_DATA_PATH, index_col='Timestamp', parse_dates=True)
        print(f"Loaded normal training data: {df_normal_train.shape}")

    except FileNotFoundError:
        print(f"\nERROR: Normal data file {NORMAL_DATA_PATH} not found.")
        print("Please run data_generator.py first to create the data.")
        exit()
    # Add specific catch for ValueError to see context if it still happens
    except ValueError as e:
        print(f"\nERROR: Encountered ValueError during pd.read_csv!")
        print(f"Error message: {e}")
        print("This usually means the column specified in 'index_col' ('Timestamp') was not found exactly as named in the CSV header.")
        print("Please re-verify the DEBUG output above regarding the header and detected columns.")
        print("\nFull Traceback for ValueError:")
        print(traceback.format_exc())
        exit() # Stop execution after this specific error
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during data loading:")
        print(f"Error message: {e}")
        print("\nFull Traceback:")
        print(traceback.format_exc())
        exit() # Stop execution

    # --- The rest of the script ---

    # Ensure columns match SENSOR_CONFIG (in case data generation changes)
    try:
         df_normal_train = df_normal_train[list(SENSOR_CONFIG.keys())]
    except KeyError as e:
         print(f"\nERROR: Mismatch between columns in loaded data and SENSOR_CONFIG.")
         print(f"Missing/unexpected column: {e}")
         print(f"Columns loaded: {df_normal_train.columns.tolist()}")
         print(f"Columns expected in config: {list(SENSOR_CONFIG.keys())}")
         exit()

    # 2. Initialize and Train the detector
    detector = AnomalyDetector()
    detector.train(df_normal_train)

    # 3. Save the trained model
    detector.save_model()

    # 4. Load the ANOMALOUS data for testing prediction
    try:
        # --- Add similar debug for anomalous data loading ---
        print(f"\n--- DEBUG: Attempting to load {ANOMALOUS_DATA_PATH} ---")
        try:
            with open(ANOMALOUS_DATA_PATH, 'r', encoding='utf-8-sig') as f:
                header = f.readline().strip()
                print(f"DEBUG: Raw header read from file: '{header}'")
                cols = [c.strip() for c in header.split(',')]
                print(f"DEBUG: Columns detected after split: {cols}")
                if 'Timestamp' not in cols:
                    print("DEBUG: 'Timestamp' IS NOT in the detected columns list.")
                    if cols:
                         print(f"DEBUG: Representation of first detected column: {repr(cols[0])}")
                    else:
                         print("DEBUG: No columns detected after split.")
        except Exception as debug_e:
            print(f"DEBUG: Error during manual header check: {debug_e}")
        print("--- DEBUG: End of manual header check ---")

        df_anomalous_test = pd.read_csv(ANOMALOUS_DATA_PATH, index_col='Timestamp', parse_dates=True)
        print(f"Loaded anomalous test data: {df_anomalous_test.shape}")

    except FileNotFoundError:
        print(f"\nERROR: Anomalous data file {ANOMALOUS_DATA_PATH} not found.")
        print("Please run data_generator.py first to create the data.")
        exit()
    except ValueError as e:
         print(f"\nERROR: Encountered ValueError during pd.read_csv for anomalous data!")
         print(f"Error message: {e}")
         print("Please check the DEBUG output above for this file's header.")
         print("\nFull Traceback for ValueError:")
         print(traceback.format_exc())
         exit()
    except Exception as e:
         print(f"\nERROR: An unexpected error occurred loading anomalous data:")
         print(f"Error message: {e}")
         print("\nFull Traceback:")
         print(traceback.format_exc())
         exit()

    try:
        df_anomalous_test = df_anomalous_test[list(SENSOR_CONFIG.keys())]
    except KeyError as e:
         print(f"\nERROR: Mismatch between columns in loaded anomalous data and SENSOR_CONFIG.")
         print(f"Missing/unexpected column: {e}")
         print(f"Columns loaded: {df_anomalous_test.columns.tolist()}")
         print(f"Columns expected in config: {list(SENSOR_CONFIG.keys())}")
         exit()

    # 5. Make predictions on the anomalous data
    print("\nMaking predictions on the test data (containing anomalies)...")
    predictions, scores = detector.predict(df_anomalous_test)

    # Add predictions and scores back to the DataFrame for inspection
    df_anomalous_test['Anomaly_Predicted'] = predictions # 1=normal, -1=anomaly
    df_anomalous_test['Anomaly_Score'] = scores

    num_detected_anomalies = (predictions == -1).sum()
    print(f"Detector flagged {num_detected_anomalies} points as anomalies out of {len(df_anomalous_test)} total points.")

    # --- Interpretation for Layman ---
    print("\n--- Prediction Summary ---")
    print("The trained 'detective' analyzed the test data which contained hidden problems.")
    print("For each moment in time, it assigned a score indicating how 'normal' or 'weird' it looked.")
    print("Based on a threshold learned during training, it flagged the weirdest points as anomalies (-1).")
    print(f"In this test run, it identified {num_detected_anomalies} potential issues.")
    print("The next step (Evaluation) will check how accurate these flags are against the 'cheat sheet' (anomaly log).")

    # Display first few rows with predictions
    print("\nSample of test data with predictions:")
    print(df_anomalous_test.head())

    # Display rows flagged as anomalies
    print("\nSample of points flagged as anomalies:")
    # Check if any anomalies were predicted before trying to print
    anomalies_found = df_anomalous_test[df_anomalous_test['Anomaly_Predicted'] == -1]
    if not anomalies_found.empty:
        print(anomalies_found.head())
    else:
        print("No anomalies were detected in the test data sample.")