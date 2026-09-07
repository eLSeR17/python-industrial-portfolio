# Notebook: Predictive Maintenance Analysis Demo

## Overview

This notebook demonstrates the key analysis workflows of the Predictive
Maintenance Engine. It is intended as documentation of what the system
can produce — actual execution requires a running Jupyter environment
with the project dependencies installed.

## Sections

### 1. Sensor Data Exploration
- Load time-series sensor data from TimescaleDB
- Plot vibration waveforms (X, Y, Z channels)
- Visualize temperature and pressure trends
- Overlay degradation onset markers

### 2. Feature Engineering Pipeline
- Demonstrate rolling statistics over 10/50/200-sample windows
- FFT frequency spectrum visualization
- Spectral entropy evolution over asset lifetime
- Correlation matrix of all engineered features

### 3. Anomaly Detection Analysis
- Isolation Forest decision boundary visualization
- Z-score distribution vs. threshold
- Confusion matrix for detected anomalies
- ROC curve for anomaly scoring threshold tuning

### 4. Failure Prediction Model
- Random Forest feature importance ranking
- Prediction vs. actual RUL scatter plot
- Calibration curve for failure probability
- Precision-recall curve for failure classification

### 5. Maintenance Schedule Optimization
- Gantt chart of maintenance windows
- Cost vs. reliability trade-off analysis
- What-if scenarios (defer maintenance by N days)
- ROI calculation for predictive vs. reactive maintenance

## How to Run

```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
jupyter notebook notebooks/

# Run cells in order (Shift+Enter)
```

## Requirements
- Running instance of the Predictive Maintenance API
- PostgreSQL/TimescaleDB with historical data
- matplotlib, seaborn, plotly for visualizations
