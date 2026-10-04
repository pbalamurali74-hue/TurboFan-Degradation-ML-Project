# ✈️ Turbofan Degradation Prediction — NASA C-MAPSS

A machine-learning project focused on analyzing **aircraft turbofan degradation** and predicting Remaining Useful Life using NASA's C-MAPSS run-to-failure simulation data.

## Objectives
- Explore multivariate engine sensor data.
- Identify degradation patterns over operating cycles.
- Generate RUL labels.
- Engineer useful time-series features.
- Train regression models.
- Evaluate RUL predictions with MAE, RMSE and R².
- Visualize degradation and prediction performance.

## Dataset
The repository works with NASA C-MAPSS datasets **FD001–FD004**, covering different operating conditions and fault modes.

## Workflow
```
C-MAPSS
  ↓
Data Loading
  ↓
EDA
  ↓
Preprocessing
  ↓
RUL Label Generation
  ↓
Feature Engineering
  ↓
Regression / Deep Learning
  ↓
Evaluation
```

## Tech Stack
Python • Pandas • NumPy • Matplotlib • Seaborn • Scikit-Learn • Jupyter

## Project status
This is the earlier/development version of the turbofan work. The more advanced, leakage-controlled implementation is maintained in **[Turbofan-Engine-RUL-Prediction](https://github.com/pbalamurali74-hue/Turbofan-Engine-RUL-Prediction)**.

## Author
**Purushotham Balamurali**