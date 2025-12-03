<<<<<<< HEAD
Acea Water Prediction — Code Purpose and Project Context

📌 What Is This Code For?

This code is a unified machine learning pipeline designed to process hydrological datasets. It automates data cleaning, preprocessing, feature engineering, and model training to forecast water levels and flow rates. The pipeline ensures consistency across multiple datasets (aquifers, rivers, lakes, springs) and evaluates predictive performance using metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

🏢 What Project Is It Part Of?

The code was developed for the Acea Water Prediction competition hosted on Kaggle. This project provided real-world hydrological data and asked participants to build models that could predict water availability across multiple water sources.

🏭 Which Company Is Behind It?

The datasets were released by Acea Group, a major Italian multi-utility company. Acea operates in water, energy, and environmental services, and manages water supply for millions of people in central Italy.

🎯 What Did the Company Want?

Acea’s objective in launching the Kaggle challenge was to:

Encourage data scientists to analyze hydrological time series data.

Develop predictive models for groundwater depth, river hydrometry, lake levels, and spring flow rates.

Generate actionable insights to optimize water resource management.

Support sustainability and infrastructure planning by forecasting water availability more accurately.

✅ Summary

In short, this code is a solution pipeline for the Kaggle Acea Water Prediction competition. It was built to meet Acea Group’s request: create robust, reproducible models that can forecast water availability using diverse hydrological datasets.

Acea Water Prediction — Unified Pipeline

⚙️ How the Code Works

File Discovery

The pipeline looks for the expected CSV datasets (aquifers, rivers, lakes, springs) in the directories /content/acea-water-prediction or /content.

Helper functions like list_found_files() scan those paths and confirm which files are present.

Dataset Configuration

Each dataset has a configuration (DATASETS_CFG) that specifies:

Column renaming rules (rename_map)

Target variable (e.g., groundwater depth, flow rate, lake level)

Required columns

Columns to use for modeling

Preprocessing Utilities

Functions like robust_rename, parse_and_sort_date, coalesce_station_columns standardize the data.

Missing values are handled with forward_fill_exogenous.

Outliers are clipped using clip_outliers.

Time features (day of year sine/cosine, month) are added with add_time_features.

Lag and rolling window features are generated with add_lag_rolling.

Special Case: Lupa Dataset

For Water_Spring_Lupa.csv, non‑negative flow values are replaced with the average of the same day/month across years.

Negative values are preserved.

Train/Test Split

The data is split chronologically into training and testing sets using split_train_test.

Numeric features are selected with select_numeric_features.

Model Selection

Different models are chosen depending on dataset type:

Aquifers & Springs: Gradient Boosting Regressor

Rivers: Random Forest Regressor

Lakes: ElasticNet (with scaling pipeline)

Training & Evaluation

The chosen model is trained on the training set.

Predictions are made on the test set.

Performance is reported using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

Main Runner

The main() function loops through all datasets, applies preprocessing, trains models, and prints evaluation metrics.

📚 Libraries You Need to Install

Your code uses the following Python libraries:

Core scientific stack:

numpy → numerical computations

pandas → data manipulation and CSV handling

Machine learning (scikit-learn):

scikit-learn → provides models (GradientBoosting, RandomForest, ElasticNet), metrics (MAE, RMSE), preprocessing (StandardScaler), and pipelines

System utilities:

os → file and directory handling

re → regular expressions (used in file discovery)

math → mathematical functions (e.g., square root)

warnings → suppresses unnecessary warnings

✅ Installation Commands

Run these in your terminal or VS Code environment:

pip install numpy pandas scikit-learn

(The other modules like os, re, math, warnings are part of Python’s standard library, so you don’t need to install them separately.)
=======
# Acea-Water-Project
>>>>>>> 9db916af0f371da1c81b1f4b996381c65c90533a
