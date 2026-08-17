# AQI-Prediction
## To predict the AQI using Machine Learning Algorithms.
Predicting air quality from real gas-sensor and weather data using classical Machine Learning algorithms, built on the UCI "Air Quality" dataset.

Overview

This project takes ~9,357 hours (roughly one year, March 2004 – April 2005) of readings from a multi-sensor air quality monitoring device and uses them to classify air quality into three levels — Good, Moderate, and Poor. The work is split into two notebooks that mirror a typical applied-ML pipeline: data cleaning and feature engineering first, then model training, tuning, and evaluation.

	
Task	Multi-class classification (3 classes: Good / Moderate / Poor)
Data	UCI Air Quality Data Set (hourly gas sensor + weather readings)
Rows	9,357 hourly records → 9,334 after feature engineering
Best model (as reported)	XGBoost — 92.0% accuracy, 0.922 macro F1 (see Limitations before trusting this number)
Notebooks	AQI_prediction_phase1.ipynb (data prep), AQI_prediction_phase2.ipynb (modeling)

Dataset

The project uses the UCI Air Quality Data Set — hourly averaged readings from an array of 5 metal-oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device, deployed at road level in a polluted area of an Italian city between March 2004 and April 2005. Ground-truth pollutant concentrations were collected alongside the sensor readings from a co-located reference analyzer.

Raw columns include:

Ground-truth pollutants: CO(GT), NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)
Sensor responses: PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
Weather: Temperature (T), Relative Humidity (RH), Absolute Humidity (AH)

Missing readings in the source data are encoded as -200.

Methodology
Phase 1 — Data Preparation (AQI_prediction_phase1.ipynb)
Loading & cleaning: Combine Date/Time into a DateTime index, drop empty rows, and treat -200 as missing.
Missing values: NMHC(GT) is dropped (>90% missing); remaining gaps (~4–18%) are filled with time-based interpolation.
Duplicates: Checked for duplicate timestamps (none found).
Outliers: Visualized with boxplots and capped using the 1.5×IQR rule (Winsorization).
EDA: Time-series trend plots, distribution histograms, Q-Q plots, monthly/hourly (diurnal) seasonality analysis, violin plots, and a correlation heatmap. Key findings: strong winter-vs-summer seasonality, a clear morning/evening rush-hour double-peak in pollutants, right-skewed (non-normal) distributions, and high cross-correlation between several pollutants and their corresponding sensors.
Feature engineering: For every column, adds a 1-hour lag (_lag_1hr) and a 24-hour rolling mean (_roll_24hr), plus hour, day_of_week, and month features.
Scaling & export: Features are standardized with StandardScaler and saved to model_features_X.csv; the (unscaled) pollutant targets are saved to model_targets_y.csv.

Phase 2 — Modeling (AQI_prediction_phase2.ipynb)
Label creation: Each of the four pollutant targets is binned into Good/Moderate/Poor using its own 25th/75th percentile. A majority vote across the four pollutants produces a single AQ_Class label; ties (~19.7% of rows) are resolved with a custom rule that prefers Poor > Good > Moderate.
Baseline models: Logistic Regression, Decision Tree, Random Forest, KNN, SVM (RBF), and XGBoost are trained on an 80/20 train/test split and compared on accuracy and macro F1.
Decision Tree tuning: Both pre-pruning (GridSearchCV over max_depth, min_samples_split, min_samples_leaf) and post-pruning (cost-complexity pruning / ccp_alpha) are explored.
Random Forest tuning: GridSearchCV over n_estimators, max_depth, max_features, min_samples_leaf.
Final comparison: All tuned models are compared on the held-out test set and validated with 5-fold cross-validation to check stability across folds.

Results

As reported in the notebook (test set, 80/20 split):
Model	Accuracy	Macro F1	5-Fold Mean F1
XGBoost	0.920	0.922	0.920
Random Forest (Tuned)	0.913	0.916	0.910
SVM (RBF)	0.893	0.896	0.897
Decision Tree (Post-Pruned)	0.878	0.882	0.872
Decision Tree (Pre-Pruned)	0.874	0.878	0.859
Logistic Regression	0.864	0.868	0.871
Decision Tree (Baseline)	0.862	0.866	0.856
KNN	0.844	0.848	0.850
