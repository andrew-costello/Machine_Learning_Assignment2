import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold

# Load steel.csv
data = pd.read_csv('steel.csv')
X = data.drop(columns='tensile_strength')
y = data['tensile_strength']

# Set up 10-fold cross-validation split
kf = KFold(n_splits=10, shuffle=True, random_state=298)

# Initialize default Random Forest Regressor
rf_default = RandomForestRegressor(random_state=298)

# Define evaluation metrics for cross-validation
scoring = {
    'MAE': 'neg_mean_absolute_error',        # mean absolute error
    'MAPE': 'neg_mean_absolute_percentage_error'  # mean absolute percentage error
}

# Perform cross-validation
cv_results = cross_validate(rf_default, X, y, cv=kf, scoring=scoring, return_train_score=True)

# Calculate mean train and test errors (convert negative to positive)
train_mae_mean = -np.mean(cv_results['train_MAE'])
test_mae_mean = -np.mean(cv_results['test_MAE'])
train_mape_mean = -np.mean(cv_results['train_MAPE'])
test_mape_mean = -np.mean(cv_results['test_MAPE'])

print("Random Forest Default Model")
print(f"Average Training MAE: {train_mae_mean:.4f}")
print(f"Average Test MAE: {test_mae_mean:.4f}")
print(f"Average Training MAPE: {train_mape_mean:.4f}")
print(f"Average Test MAPE: {test_mape_mean:.4f}")