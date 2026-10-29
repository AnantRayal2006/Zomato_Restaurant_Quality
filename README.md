# Predicting Restaurant Quality from Zomato Data

A machine-learning project that predicts whether a restaurant will be rated **Low (<3.5)**, **Good (3.5-3.9)** or **High (>=4.0)** from observable features, so diners get an early quality signal for new/unrated places and owners see which factors relate to better ratings.

## Problem and objectives
| # | Objective | Result |
|---|-----------|--------|
| O1 | Clean data (0 missing, 0 duplicates) | Met |
| O2 | >= 10 visuals and >= 5 statistical tests | 15 figures, 12 tests |
| O3 | Scenario A (with votes): Macro-F1 and Accuracy >= 0.70 | Macro-F1 0.742, Accuracy 0.771 |
| O4 | Scenario B (no votes): Macro-F1 >= 0.58 | Macro-F1 0.652, Accuracy 0.698 |
| O5 | Regression (A): RMSE <= 0.40, R2 >= 0.50 | RMSE 0.321, R2 0.663 |
| O6 | Key drivers via permutation importance | Done |
| O7 | Predict quality for all unrated restaurants | 2,148 scored |

## Dataset
Zomato Restaurants dataset (9,551 restaurants, 15 countries, 21 columns), Kaggle: "Zomato Restaurants Data". Stored in `data/`.

## Workflow
1. Cleaning: removed constant and leakage columns, imputed 18 invalid costs, handled 12 currencies (relative-cost feature), separated 2,148 "Not rated" restaurants.
2. Feature engineering: cuisine flags, chain size, locality density, relative cost, city group.
3. EDA and statistics: Shapiro-Wilk, Mann-Whitney U, Kruskal-Wallis, Spearman, Chi-square with effect sizes.
4. Modelling: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Neural Network (MLP); 5-fold CV; RandomizedSearchCV tuning; evaluation on a held-out 20% test set.
5. Regression of the exact rating, permutation importance, and predictions for unrated restaurants.

## Repository structure
```
Zomato_Restaurant_Quality_Prediction.ipynb   main notebook (run top to bottom)
data/        raw dataset
figures/     15 saved plots
outputs/     cleaned data, test tables, CV and regression results, predictions
models/      trained models (joblib)
```

## Run it
```
pip install -r requirements.txt
jupyter notebook Zomato_Restaurant_Quality_Prediction.ipynb
```

## Limitations
Single-platform snapshot (about 91% India), approximate exchange rates, self-selected reviews, and association is not causation.

## Author
**<Your Name>** | Reg. No. <...> | <University / Course>
GitHub: <link> | LinkedIn: <link>
