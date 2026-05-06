# Titanic - Machine Learning from Disaster 

## Result
- **Kaggle Score:** 77.0% accuracy
- **Rank:** 8,752 / 12,812 teams
- **Kaggle Notebook:** [View Here](https://www.kaggle.com/code/ngowthamkumar/notebook5ad3281e9e)

## Problem
Binary classification - predict which passengers
survived the Titanic disaster using passenger records
(age, gender, ticket class, cabin etc.)

## Dataset
- train.csv - 891 passengers with survival labels
- test.csv  - 418 passengers to predict

## Approach

### EDA Findings
- Women survived at 74% vs men at 19%
- 1st class passengers survived more than 3rd class
- Children had higher survival rates
- 38% overall survival rate

### Missing Values
| Column | Missing | Strategy |
|--------|---------|----------|
| Age | 177 (20%) | Median imputation |
| Cabin | 687 (77%) | Dropped (too many missing) |
| Embarked | 2 | Mode imputation |

### Feature Engineering
| Feature | Source | Reason |
|---------|--------|--------|
| Title | Name column | Social status signal |
| FamilySize | SibSp + Parch + 1 | Group survival pattern |
| IsAlone | FamilySize == 1 | Solo travelers at risk |

### Model
- Random Forest Classifier
- GridSearchCV hyperparameter tuning
- 5-fold cross validation

## Results
| Model | CV Accuracy |
|-------|-------------|
| Random Forest (baseline) | 81.5% |
| Random Forest (tuned) | 83.8% |
| Kaggle leaderboard | 77.0% |

## Key Learnings
- Train accuracy (98%) vs CV accuracy (83%) → overfitting
- Feature engineering > model complexity
- GridSearchCV found: max_depth=10, n_estimators=100

## Tech Stack
Python | Pandas | Scikit-learn | Matplotlib | Seaborn