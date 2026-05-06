# House Prices — Advanced Regression Techniques 

## Result
- **Kaggle Score:** 0.121 RMSE
- **Rank:** 417 / 4,868 teams (Top 9%) 
- **Kaggle Notebook:** [View Here](https://www.kaggle.com/code/ngowthamkumar/notebooka3645426ac)

## Problem
Regression — predict sale prices of residential homes
in Ames, Iowa using 79 features describing almost
every aspect of the property.

## Dataset
- train.csv — 1460 houses with sale prices
- test.csv  — 1459 houses to predict

## Approach

### EDA Findings
- SalePrice right-skewed (mean $180k, max $755k)
- Log transformation needed for target variable
- 19 columns with missing values
- Many NA values mean "feature doesn't exist"

### Missing Value Strategy
| Type | Example | Strategy |
|------|---------|----------|
| Feature doesn't exist | PoolQC, GarageType | Fill "None"/0 |
| Genuinely missing | LotFrontage | Neighborhood median |
| Small count missing | Electrical (1) | Mode imputation |

### Feature Engineering
| Feature | Description |
|---------|-------------|
| TotalSF | Basement + 1st + 2nd floor area |
| HouseAge | Year sold - Year built |
| TotalBath | All bathrooms combined |
| Qual_TotalSF | OverallQual × TotalSF interaction |
| Neighborhood_Mean | Average price per neighborhood |

### Encoding
- Ordinal encoding for 20 quality columns (Poor→Excellent)
- One-hot encoding for 21 nominal columns
- Target encoding for Neighborhood

### Models
- XGBoost Regressor
- LightGBM Regressor
- Weighted ensemble (0.5 XGB + 0.5 LGBM)

## Results
| Version | Kaggle RMSE | Key Change |
|---------|-------------|------------|
| V1 | 0.135 | Basic features |
| V2 | 0.123 | Feature engineering + kept outliers |
| V3 | 0.121 | Interaction features + neighborhood encoding |

## Key Learnings
- Log transformation crucial for skewed targets
- Removing outliers hurt test performance
- Combining train+test before cleaning ensures consistency
- Feature engineering gave bigger gains than model tuning
- Data leakage in target encoding — must use KFold

## Tech Stack
Python | Pandas | XGBoost | LightGBM | Scikit-learn | Matplotlib | Seaborn