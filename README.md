# Car Insurance Claims Prediction

## Overview

This project focuses on predicting the likelihood of a car insurance claim being filed. Using a dataset of 10,302 car insurance policies, the primary goal was to build a classification model to accurately identify policies that are likely to result in a claim. The dataset contains 27 features, including policyholder demographics (age, income, education), vehicle information (age, type, value), and driving history (MVR points, past claims).

The target variable, `CLAIM_FLAG`, is binary, with approximately **26.5%** of the policies in the training set resulting in a claim. The final model is an XGBoost Classifier, which was selected and tuned to achieve the best predictive performance.

## Key Insights from Exploratory Data Analysis

The initial data exploration revealed several factors that correlate with the probability of a claim:

- **Driving History:** The number of claims in the last 5 years (`5_year_num_of_claims`) and Motor Vehicle Record (MVR) points (`license_points`) showed the strongest positive correlations with filing a new claim. Specifically, `5_year_num_of_claims` had a correlation coefficient of **0.22**.
- **Socioeconomic Factors:** Lower income levels and lower home values are associated with a higher likelihood of a claim. `value_of_home` had the strongest negative correlation at **-0.19**.
- **Demographics:** Single parents (`single_parent`) and individuals with more children (`num_of_children`) tend to file more claims.
- **Vehicle Characteristics:** The age of the vehicle (`vehicle_age`) showed a negative correlation, suggesting that owners of newer cars are slightly less likely to file claims.

## Major Steps

1.  **Data Cleaning:**
    - Renamed 26 columns for better readability.
    - Removed `$` and `,` from 5 currency-related columns and converted them to numeric types.
    - Stripped the `z_` prefix from 6 categorical features.
    - Dropped irrelevant columns: `ID` and `date_of_birth`.

2.  **Preprocessing & Feature Engineering:**
    - **Missing Values:** Handled 2,418 missing data points across 6 features. Numerical features were imputed using a `KNNImputer` (k=2), and categorical features were imputed using the most frequent value.
    - **Feature Transformation:** Applied a square root transformation to 6 right-skewed numerical features to normalize their distributions.
    - **Feature Scaling:** Standardized all 13 numerical features using `StandardScaler`.
    - **Encoding:**
        - One-hot encoded 2 nominal categorical features (`occupation`, `vehicle_type`), resulting in 12 new columns.
        - Ordinally encoded 1 feature (`highest_education`) based on its inherent rank.
        - Binary encoded 6 features (e.g., `single_parent`, `gender`).

3.  **Model Selection & Tuning:**
    - Evaluated 10 different classification models using 10-fold cross-validation. **CatBoost** and **Gradient Boost** emerged as the top performers.
    - Selected **XGBoost** for hyperparameter tuning due to its comparable performance and faster tuning time.
    - Conducted a `RandomizedSearchCV` with 2,000 iterations, followed by a more focused `GridSearchCV` to fine-tune the model's hyperparameters.

## Results

The tuned XGBoost Classifier demonstrated solid predictive performance on the unseen test data.

- **Cross-Validation Score (Training):** The final model achieved a weighted F1-score of **0.784** during 10-fold cross-validation. This represents a **0.8%** improvement over the baseline XGBoost model score of 0.777.
- **Test Set Performance:** On the hold-out test set (20% of the data), the model achieved a weighted F1-score of **0.775**.

The confusion matrix on the test set provides a clear picture of the model's classification accuracy:

![Confusion Matrix on Test Data](confusion_matrix.png)

This project successfully developed a reliable model for predicting car insurance claims, with driving history and socioeconomic factors being the most significant predictors.
