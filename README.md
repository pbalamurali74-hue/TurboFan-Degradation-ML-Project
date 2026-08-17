# TurboFan-Degradation-ML-Project
Turbo Fan Degradation Prediction Using NASA C-MAPSS Dataset

1. Project Overview

This project focuses on modeling and predicting the degradation of
aircraft turbofan engines using the NASA C-MAPSS (Commercial Modular
Aero-Propulsion System Simulation) dataset.

The dataset contains multivariate time-series measurements collected
from simulated turbofan engines. Each engine is monitored over multiple
operating cycles. The engines start in a normal operating condition and
develop degradation until failure in the training data. The test data
ends before failure, and the objective is to estimate the Remaining
Useful Life (RUL) of each test engine.

The NASA C-MAPSS data was developed for run-to-failure simulation and
was used in the PHM08 data competition. The source paper describes
damage propagation through progressively worsening degradation until a
failure criterion is reached.

2. Objectives

Load and understand the NASA C-MAPSS dataset.

Perform exploratory data analysis on turbofan engine sensor
measurements.

Analyze degradation patterns over operating cycles.

Preprocess and normalize sensor data where required.

Generate Remaining Useful Life (RUL) labels for training data.

Build machine learning models for turbofan degradation/RUL
prediction.

Evaluate model performance using suitable regression metrics.

Visualize predicted and actual RUL values.

Provide a basis for predictive maintenance of turbofan engines.

3. Dataset

This project uses the NASA C-MAPSS FD001 dataset initially.

FD001 contains:

Property                                Value

Training trajectories                     100
Test trajectories                         100
Operating conditions            1 (Sea Level)
Fault modes               1 (HPC Degradation)
Columns                                    26

Each row represents one snapshot of an engine during a particular
operating cycle.

The 26 columns are:

Unit number

Time/cycle

Operational setting 1

Operational setting 2

Operational setting 3 6--26. Sensor measurements 1--21

The training data contains complete run-to-failure trajectories. The
test data contains partial trajectories that stop before failure. True
RUL values are provided separately for evaluating predictions.

4. Available C-MAPSS Datasets

The project files also include FD002, FD003 and FD004.

Dataset     Train   Test   Conditions Fault Modes

FD001         100    100            1 1
FD002         260    259            6 1
FD003         100    100            1 2
FD004         248    249            6 2

FD002 and FD004 introduce six operating conditions, while FD003 and
FD004 include two fault modes.

5. Project Workflow

NASA C-MAPSS Dataset
        |
        v
Data Loading
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Sensor Selection
        |
        v
RUL Label Generation
        |
        v
Feature Engineering
        |
        v
Train / Validation Split
        |
        v
Machine Learning Model
        |
        v
RUL Prediction
        |
        v
Model Evaluation
        |
        v
Degradation Visualization

6. Technologies Used

Python

Google Colab / Jupyter Notebook

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Machine Learning / Regression algorithms

Deep learning libraries such as TensorFlow or PyTorch can be added later
if neural-network-based RUL prediction is required.

7. Suggested Project Structure

Turbo-Fan-Degradation/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── train_FD002.txt
│   ├── test_FD002.txt
│   ├── RUL_FD002.txt
│   ├── train_FD003.txt
│   ├── test_FD003.txt
│   ├── RUL_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
│
├── notebooks/
│   ├── experiment_1_load_dataset.ipynb
│   ├── experiment_2_statistics.ipynb
│   └── ...
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   └── predictions/
│
├── README.md
└── requirements.txt

8. Installation

If you are using Google Colab, most required libraries are already
available.

For a local Python environment:

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

9. Loading the Dataset

The C-MAPSS files are space-separated text files without a header.

Example:

import pandas as pd

df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)

print(df.head())
print(df.shape)

Column names can then be assigned:

columns = [
    "unit_number",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

df.columns = columns

10. Exploratory Data Analysis

The initial analysis should include:

Number of engines.

Number of cycles for each engine.

Dataset dimensions.

Missing-value analysis.

Descriptive statistics.

Sensor distributions.

Sensor behavior over operating cycles.

Correlation between sensors and degradation/RUL.

Identification of sensors that provide useful degradation
information.

Example:

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

11. RUL Concept

Remaining Useful Life (RUL) represents the number of operational cycles
an engine is expected to continue operating before failure.

For a training engine that fails at cycle N, a simple RUL definition
is:

RUL = N - current_cycle

For example, if an engine fails at cycle 200:

Cycle 1   -> RUL 199
Cycle 2   -> RUL 198
...
Cycle 199 -> RUL 1
Cycle 200 -> RUL 0

The exact RUL-labeling strategy should be kept consistent with the
chosen modeling approach.

12. Machine Learning Approach

Possible regression models include:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Support Vector Regression

For a first implementation, Random Forest Regression or Gradient
Boosting Regression can be used as a baseline.

A more advanced implementation can use:

LSTM

GRU

1D CNN

CNN-LSTM

Transformer-based time-series models

13. Evaluation Metrics

RUL prediction can be evaluated using metrics such as:

Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted
RUL.

Root Mean Squared Error (RMSE)

Penalizes larger prediction errors more strongly.

R² Score

Measures how well the model explains the variance in the target values.

Example:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

14. Expected Outcome

The completed project should be able to:

Read and analyze C-MAPSS turbofan engine data.

Identify degradation trends from sensor measurements.

Create an appropriate RUL prediction target.

Train a machine learning model.

Predict the remaining useful life of test engines.

Compare predicted RUL with the available true RUL values.

Visualize engine degradation and prediction performance.

15. Important Notes

FD001 is recommended as the starting dataset because it has one
operating condition and one fault mode.

FD002 and FD004 are more challenging because they contain six
operating conditions.

FD003 and FD004 contain two fault modes.

The data contains operational settings and sensor noise.

Training trajectories reach failure, whereas test trajectories
terminate before failure.

Data preprocessing and feature selection should be performed
carefully because the dataset is multivariate time-series data.

16. Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation," Proceedings of
the First International Conference on Prognostics and Health Management
(PHM08), Denver, Colorado, October 2008.

17. Project Status

Current Stage: Dataset loading, exploration and statistical
analysis.

Next Stage: Data preprocessing, RUL label generation and degradation
analysis.

Author

Purushotham Balamurali

Project: Turbo Fan Degradation Prediction using NASA C-MAPSS Dataset
