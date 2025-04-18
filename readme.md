# Real-Time Anomaly Detection System for Simulated Multi-Sensor Telemetry

This project demonstrates an end-to-end system for detecting anomalies in simulated time-series telemetry data, relevant to fields like aerospace or industrial monitoring. It includes data generation, model training (using Isolation Forest), evaluation, and a real-time simulation dashboard built with Streamlit.

## Project Goal

To create a feasible undergraduate-level project showcasing skills in:
*   Time-series data simulation
*   Multi-sensor data handling
*   Anomaly detection algorithms (Isolation Forest)
*   Model evaluation
*   Real-time data processing simulation
*   Data visualization and dashboarding (Streamlit)

## Layman's Explanation

Imagine watching the sensors on a complex machine, like a rocket engine. Usually, the readings (temperature, pressure, etc.) stay within normal ranges. This system tries to automatically spot when something looks 'weird' or 'off' – an anomaly.

1.  **Data Simulation (`data_generator.py`):** We first create fake sensor data that looks realistic, including normal operation and specific problems (like sudden spikes or gradual drifts). We keep a 'cheat sheet' of the problems we added.
2.  **Training the 'Detective' (`anomaly_detector.py`):** We use a machine learning technique called Isolation Forest. We show it lots of 'normal' data so it learns what smooth operation looks like.
3.  **Checking for Problems (`anomaly_detector.py`):** The trained 'detective' can then look at new sensor readings and assign a 'weirdness score'. If the score is too high, it flags an anomaly.
4.  **Judging the Detective (`evaluation.py`):** We compare the detective's flags against our 'cheat sheet' to see how accurate it was (did it find the real problems? did it raise false alarms?).
5.  **Live Dashboard (`app.py`):** We built an interactive web page where you can watch the simulated sensor data flow in, see the detective flag anomalies in real-time, and view the accuracy scores.

## Project Structure

*   `config.py`: Configuration settings (sampling rate, sensor details, anomaly parameters, etc.).
*   `data_generator.py`: Generates simulated normal and anomalous data (`.csv`) and an anomaly log (`.json`).
*   `anomaly_detector.py`: Defines, trains, and saves/loads the `AnomalyDetector` class using `StandardScaler` and `IsolationForest`.
*   `evaluation.py`: Functions to compare predictions against the ground truth log and calculate metrics (Precision, Recall, F1-Score).
*   `app.py`: The Streamlit application for visualizing the real-time simulation and results.
*   `requirements.txt`: Lists necessary Python libraries.
*   `README.md`: This file.
*   `simulated_normal_data.csv`: Output data (normal).
*   `simulated_anomalous_data.csv`: Output data (with anomalies).
*   `anomaly_log.json`: Ground truth log of injected anomalies.
*   `anomaly_detector_model.joblib`: Saved trained model and scaler.

## How to Run

1.  **Setup Environment:**
    *   Ensure you have Python 3.8+ installed.
    *   Create a virtual environment (recommended):
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Generate Data:**
    *   Run the data generator script. This will create the `.csv` data files and the `.json` anomaly log.
    *   ```bash
        python data_generator.py
        ```
    *   (Optional) Inspect the generated `matplotlib` plot to see the data.

3.  **Train the Model:**
    *   Run the anomaly detector script in training mode. This uses the `simulated_normal_data.csv` to train the model and saves it to `anomaly_detector_model.joblib`.
    *   ```bash
        python anomaly_detector.py
        ```

4.  **Run the Evaluation (Optional Standalone Check):**
    *   Run the evaluation script. This loads the anomalous data, the saved model, makes predictions, compares to the log, and prints metrics.
    *   ```bash
        python evaluation.py
        ```

5.  **Run the Streamlit Dashboard:**
    *   Launch the Streamlit application.
    *   ```bash
        streamlit run app.py
        ```
    *   Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`).
    *   Use the "Start/Resume", "Pause", and "Reset" buttons to control the simulation.

## Deployment (Streamlit Cloud)

1.  **Create GitHub Repository:** Push your project code (all `.py` files, `requirements.txt`, `README.md`) to a GitHub repository. *Do not* commit the large `.csv` files or the `.joblib` model file directly if they are very large (use Git LFS or generate/train them dynamically if needed, though for this scope they should be manageable). If they are small enough, committing them makes deployment easier.
2.  **Sign up for Streamlit Cloud:** Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign up (free tier available).
3.  **Deploy App:**
    *   Connect your GitHub account.
    *   Click "New app".
    *   Select your repository and the branch containing the code.
    *   Ensure the "Main file path" is set to `app.py`.
    *   Click "Deploy!".

Streamlit Cloud will install the dependencies from `requirements.txt` and run your `app.py`. Note that the data generation and model training steps (`data_generator.py`, `anomaly_detector.py`) won't automatically run on Streamlit Cloud unless you modify `app.py` to trigger them if the files don't exist (this can be slow on startup). For this project, it's easiest to run steps 2 & 3 locally and commit the resulting `*.csv`, `.json`, and `.joblib` files to GitHub before deploying.

## Potential Improvements / Further Work

*   **More Sophisticated Data Simulation:** Incorporate physics-based models or more complex stochastic processes (e.g., ARIMA baselines).
*   **Advanced Anomaly Types:** Simulate more complex contextual or collective anomalies.
*   **Alternative Algorithms:** Implement and compare other methods (e.g., Autoencoders, LSTMs, LOF, One-Class SVM).
*   **Feature Engineering:** Add rolling statistics (mean, std dev), derivatives, or frequency-domain features (FFT for vibration) before feeding data to the model.
*   **Hyperparameter Tuning:** Use techniques like Grid Search or Randomized Search to find optimal parameters for the Isolation Forest (e.g., `n_estimators`, `contamination`).
*   **Event-Based Evaluation:** Implement evaluation metrics that consider entire anomaly events rather than just point-wise accuracy (e.g., Time-series Aware Precision/Recall).
*   **Improved Visualization:** Use Plotly or Altair within Streamlit for more interactive charts with better anomaly highlighting.
*   **Anomaly Attribution:** Explore techniques to identify which sensor(s) contribute most to a detected anomaly.