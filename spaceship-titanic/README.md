# Spaceship Titanic — Predict Alternate Dimension 

## Result
- **Kaggle Score:** 80.1% accuracy
- **Rank:** 1,084 / 2,307 teams (Top 47%)
- **Kaggle Notebook:** [View Here](https://www.kaggle.com/code/ngowthamkumar/notebookaf4387efd1)

## Problem
Binary classification — predict which passengers
were transported to an alternate dimension during
the Spaceship Titanic's collision with a spacetime anomaly.

## Dataset
- train.csv — 8,693 passengers with labels
- test.csv  — 4,277 passengers to predict
- Balanced dataset: 50.4% transported, 49.6% not

## Approach

### EDA Findings
| Feature | Transported Rate | Signal |
|---------|-----------------|--------|
| CryoSleep=True | 81.8% | Very Strong |
| ZeroSpend | 78.6% | Very Strong |
| Deck B/C | 70%+ | Strong |
| HomePlanet Europa | 65.9% | Moderate |
| VIP=True | 38.2% | Moderate |

### Domain Knowledge Applied
- CryoSleep passengers confined to cabins
  → spending must be 0 if CryoSleep=True
  → used to smartly impute missing spending values
- Cabin format "deck/num/side"
  → split into 3 separate features
- PassengerId format "gggg_pp"
  → extracted group information

### Missing Value Strategy
- Spending columns: filled using CryoSleep status
- CryoSleep: inferred from spending patterns
- Other categoricals: mode imputation

### Feature Engineering
| Feature | Description |
|---------|-------------|
| Deck, CabinNum, Side | Split from Cabin column |
| GroupSize, IsAlone | Extracted from PassengerId |
| TotalSpend | Sum of all 5 amenity columns |
| Log_Spending | Log transform of each spend column |
| NumZeroSpend | Count of zero-spend categories |
| Europa_Cryo | HomePlanet × CryoSleep interaction |
| CryoAlone | CryoSleep × IsAlone interaction |
| AgeBin | Age grouped into 7 bins |

### Data Leakage Detected and Fixed
- GroupTransportRate initially caused CV 95% vs Kaggle 79%
- Identified as data leakage
- Fixed using KFold target encoding
- CV score became honest: 79% ≈ Kaggle 80%

### Models
- XGBoost Classifier
- LightGBM Classifier
- Random Forest Classifier
- Weighted probability ensemble

## Results
| Version | Kaggle Score | Key Change |
|---------|-------------|------------|
| V1 | 79.7% | With data leakage |
| V2 | 80.1% | Fixed leakage, honest score |

## Key Learnings
- Domain knowledge crucial for smart imputation
- Data leakage: CV 95% vs Kaggle 79% revealed the problem
- KFold target encoding prevents leakage
- Balanced dataset needs different baseline thinking

## Tech Stack
Python | Pandas | XGBoost | LightGBM | Scikit-learn | Matplotlib | Seaborn