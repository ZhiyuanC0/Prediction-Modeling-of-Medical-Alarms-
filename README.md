# Medical Alarm Prediction Using ML and MAP Decision Rules
This project implements a machine learning-based system to predict medical alarm conditions from patient data using Maximum Likelihood (ML) and Maximum A Posteriori (MAP) decision rules. The goal is to accurately identify alarm states while minimizing false alarms and missed detections, which is critical in healthcare monitoring.

# Features
Analyzes a large dataset of over 30,000 samples, containing 7 physiological features from 9 patients

Applies statistical feature selection methods combining error rate minimization and correlation analysis to choose optimal, independent feature pairs

Compares ML and MAP approaches for alarm prediction performance, evaluating false alarm rates, miss detection rates, and overall error

Uses Python scientific libraries including NumPy, SciPy, and Matplotlib for data processing, analysis, and visualization

# Usage
The repository includes scripts for preprocessing data, calculating error rates, performing feature selection, and generating performance metrics. Visualizations illustrate the correlation between features and prediction results.

# Future Work
Potential improvements include integrating models that balance false alarms and miss detections more effectively and exploring advanced machine learning techniques for enhanced prediction accuracy.
