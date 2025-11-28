import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate, KFold

# Load steel.csv
data = pd.read_csv('steel.csv')
X = data.drop(columns='tensile_strength')
y = data['tensile_strength']

# Set up 10-fold cross-validation split
kf = KFold(n_splits=10, shuffle=True, random_state=298)

# Initialize default KNN regressor
knn_default = KNeighborsRegressor()

# Define evaluation metrics for cross-validation
scoring = {
    'MAE': 'neg_mean_absolute_error',        # mean absolute error
    'MAPE': 'neg_mean_absolute_percentage_error'  # mean absolute percentage error
}

# Perform cross-validation
cv_results_knn = cross_validate(knn_default, X, y, cv=kf, scoring=scoring, return_train_score=True)

# Calculate mean train and test errors (convert negative to positive)
train_mae_mean = -np.mean(cv_results_knn['train_MAE'])
test_mae_mean = -np.mean(cv_results_knn['test_MAE'])
train_mape_mean = -np.mean(cv_results_knn['train_MAPE'])
test_mape_mean = -np.mean(cv_results_knn['test_MAPE'])

print("KNN Default Model")
print(f"Average Training MAE: {train_mae_mean:.4f}")
print(f"Average Test MAE: {test_mae_mean:.4f}")
print(f"Average Training MAPE: {train_mape_mean:.4f}")
print(f"Average Test MAPE: {test_mape_mean:.4f}")

# Hyperparameter tuning for KNN
neighbors_list = [3, 5, 7, 9]
weights_list = ['uniform', 'distance']

knn_tuning_results = []

for neighbor in neighbors_list:
    for weight in weights_list:
        model = KNeighborsRegressor(
            n_neighbors=neighbor, 
            weights=weight
        )
        scores = cross_validate(
            model, X, y, 
            cv=kf, 
            scoring=scoring, 
            return_train_score=False
        )
        test_mae = -np.mean(scores['test_MAE'])
        test_mape = -np.mean(scores['test_MAPE'])
        knn_tuning_results.append({
            'n_neighbors': neighbor,
            'weights': weight,
            'Test_MAE': test_mae,
            'Test_MAPE': test_mape
        })

# Create results table
knn_tuning_df = pd.DataFrame(knn_tuning_results)
print("\nKNN Regressor Hyperparameter Tuning Results:")
print(knn_tuning_df.round(4))

import matplotlib.pyplot as plt

# Plot Test MAE vs n_neighbors for different weights
plt.figure(figsize=(10, 6))

for weight in weights_list:
    subset = knn_tuning_df[knn_tuning_df['weights'] == weight]
    plt.plot(subset['n_neighbors'], subset['Test_MAE'], 
             marker='o', label=f'weights={weight}', linewidth=2.5)

plt.xlabel('n_neighbors', fontsize=12)
plt.ylabel('Test MAE', fontsize=12)
plt.title('KNN Regressor: Test MAE vs n_neighbors (10-fold CV)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find and print best hyperparameters
best_idx = knn_tuning_df['Test_MAE'].idxmin()
best_params = knn_tuning_df.loc[best_idx]
print(f"\nBest KNN hyperparameters:")
print(f"n_neighbors: {best_params['n_neighbors']}")
print(f"weights: {best_params['weights']}")
print(f"Best Test MAE: {best_params['Test_MAE']:.4f}")
print(f"Best Test MAPE: {best_params['Test_MAPE']:.4f}")