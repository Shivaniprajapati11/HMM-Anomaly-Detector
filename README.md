# HMM-Based Anomaly Detector

## 1. Project Overview
The HMM-Based Anomaly Detector is a Python-based machine learning project that detects unusual patterns in machine sensor data using a Hidden Markov Model (HMM).

It monitors machine temperature and vibration readings and identifies abnormal operating conditions that may indicate possible machine failure.

## 2. Objectives
- To train a Gaussian Hidden Markov Model using normal sensor data.
- To calculate the log-likelihood of sensor readings.
- To determine an anomaly detection threshold.
- To detect unusual patterns using a sliding window approach.
- To visualize normal and abnormal sensor readings.

## 3. Technologies Used
- Python
- hmmlearn
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 4. Dataset Description

Two simulated sensor datasets are generated:

**Normal Data (normal_data.csv)**
- Contains 1000 sensor readings.
- Includes normal temperature and vibration values.

**Test Data (test_data.csv)**
- Contains 1000 sensor readings.
- Includes normal and artificially injected abnormal readings.

## 5. Model Implementation

A Gaussian Hidden Markov Model (GaussianHMM) is used to learn patterns from normal machine sensor data.

The model uses:
- Number of hidden states: 3
- Covariance type: Full
- Training iterations: 100
- Random state: 42

The sensor data is standardized using StandardScaler before model training.

## 6. Anomaly Detection Method

The system calculates the log-likelihood of training data using 50-reading windows.

The 5th percentile of training likelihood scores is selected as the anomaly threshold.

A sliding window approach is applied to test data:
- Window size: 50 readings.
- Step size: 10 readings.
- Windows with likelihood below the threshold are marked as anomalous.

## 7. Model Selection

Three hidden states are selected as the initial model configuration.

This provides a reasonable starting point for representing different operating patterns of the machine.

A model with fewer states may not represent enough variation, while a model with more states may introduce unnecessary complexity.

The choice of three states should be further validated by comparing models with different numbers of hidden states.

## 8. Visualization

The project generates a graph named `anomaly_detection.png`.

The graph displays:
- Machine temperature readings.
- Detected anomaly regions highlighted in red.
- Time-based sensor readings.

## 9. Real-World Applications

- Predictive maintenance in manufacturing industries.
- Industrial machine monitoring.
- Early detection of machine failures.
- Monitoring abnormal temperature and vibration patterns.

## 10. Conclusion

The project demonstrates how Hidden Markov Models can be used for anomaly detection in sequential sensor data.

The system learns normal machine behaviour and identifies unusual patterns using likelihood scores and a sliding window technique.

This approach can help industries identify potential machine problems early and reduce unexpected downtime.

## 11. How to Run the Project

1. Install the required Python libraries.
2. Open the project folder in VS Code.
3. Run the following command:

`python anomaly_detector.py`

4. The program generates sensor datasets, trains the HMM model, detects anomalies and displays the visualization.