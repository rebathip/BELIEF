import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.linear_model import BayesianRidge
from scipy.stats import entropy
from scipy.stats import norm
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import BayesianRidge
from sklearn.utils import check_array
from scipy.stats import entropy
from numbers import Number
from sklearn.utils.extmath import safe_sparse_dot
from scipy.stats import norm, gaussian_kde
import os
import pandas as pd
import itertools
import random
from sklearn.metrics import mean_squared_error
import pickle

from sklearn.linear_model import ARDRegression
from scipy.stats import norm

import warnings
import os
import numpy as np
import random
import pickle
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.linear_model import ARDRegression  # ARD for automatic relevance determination

warnings.filterwarnings("ignore", category=FutureWarning)

from slice.slice_explainer import SEMBayesianRidge

def _deprecate_n_iter(n_iter, max_iter):
    """Handle deprecation of n_iter parameter and fallback to max_iter."""
    if n_iter != "deprecated":
        return n_iter
    return max_iter if max_iter is not None else 300

def calculate_entropy(data):
    if np.var(data) == 0:
        return 0

    scipy_kernel = gaussian_kde(data)

    #  We calculate the bandwidth for later use
    optimal_bandwidth = scipy_kernel.factor * np.std(data)

    # Calculate KDE for the entire dataset
    kde = gaussian_kde(data, bw_method=optimal_bandwidth)

    # Create a range of values to represent the KDE
    x = np.linspace(np.min(data), np.max(data), 1000)

    # Evaluate the density at each point in the range
    density = kde(x)

    # Normalize the density function
    normalized_density = density / np.sum(density * (x[1] - x[0]))

    # Calculate the probabilities of positive and negative values
    positive_probability = np.sum(normalized_density[x >= 0] * (x[1] - x[0]))
    negative_probability = np.sum(normalized_density[x < 0] * (x[1] - x[0]))

    if positive_probability == 0 or negative_probability == 0:
        sign_entropy = 0
    else:
        sign_entropy = -positive_probability * np.log2(positive_probability) \
                       - negative_probability * np.log2(negative_probability)

    return sign_entropy


def split_data_into_parts(X, y, K):
    """
    Split the feature matrix X and target values y into K approximately equal-sized parts.

    Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target values.
        K (int): The number of parts to split the data into.

    Returns:
        List of tuples: Each tuple contains the feature matrix and target values for one part.
    """
    num_samples = X.shape[0]
    part_size = num_samples // K
    parts = []

    # Shuffle the data randomly
    indices = np.random.permutation(num_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split the shuffled data into K parts
    for i in range(K):
        start_idx = i * part_size
        end_idx = start_idx + part_size if i < K - 1 else num_samples
        X_part = X_shuffled[start_idx:end_idx]
        y_part = y_shuffled[start_idx:end_idx]
        parts.append((X_part, y_part))

    return parts


def get_unstable_features(X, y, model, bs_indices, num_bootstraps=None):
    coeffs_bs = []
    if num_bootstraps != None:
      print("bootstraps generated")
      bs_indices = []
      for i in np.arange(0, num_bootstraps, step=1):
        max_bs_range = X.shape[0]
        indx_bs = random.choices(range(max_bs_range), k=max_bs_range)
        bs_indices.append(indx_bs)

    for i in range(len(bs_indices)):
      indices = bs_indices[i]
      X_sample, y_sample = X[indices], y[indices]
      model.fit(X_sample, y_sample)
      coeffs_bs.append(model.coef_)
    coeffs_bs = np.array(coeffs_bs)

    sign_entropies = []
    for column in range(coeffs_bs.shape[1]):
        data = coeffs_bs[:, column]
        sign_entropy = calculate_entropy(data)
        sign_entropies.append(sign_entropy)

    num_predictors = len(sign_entropies)
    sign_entropies = np.array(sign_entropies)
    av_sign_entropy = np.mean(sign_entropies)
    ratio_zero_entropy = np.count_nonzero(sign_entropies == 0) / (sign_entropies.shape[0])

    non_zero_indices = np.where(sign_entropies != 0)[0]
    zero_ent_indices = np.where(sign_entropies == 0)[0]

    return non_zero_indices, zero_ent_indices, sign_entropies, coeffs_bs


def compute_coss_rmse(model, X_train, y_train, X_test, y_test,eval_indices_bs):
    """
    Evaluate a given feature selection method on training and test data.

    Parameters:
    - X_train, y_train, X_test, y_test: train and test datasets.

    Returns:
    - rmse: Root Mean Squared Error using the selected features.
    - sign_entropies: Sign entropies of the selected features.
    """

    # Fit and predict using the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    _, _, sign_entropies, coeffs_bs = get_unstable_features(X_train, y_train, model, bs_indices=eval_indices_bs)

    return rmse, sign_entropies, coeffs_bs


def remove_highly_correlated_features_pd(X, threshold):
    # Convert X to a pandas DataFrame
    # Calculate the correlation matrix
    df = pd.DataFrame(X)
    corr_matrix = df.corr().abs()
    # Create a mask to remove the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set the upper triangle values to NaN
    corr_matrix.mask(mask, inplace=True)

    # Find the highly correlated features
    cols_to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]

    # Drop the highly correlated features from the DataFrame
    reduced_df = df.drop(columns=cols_to_drop)

    # Convert the reduced DataFrame back to numpy array
    X_reduced = reduced_df.to_numpy()

    return X_reduced, cols_to_drop


def get_energy_XY():
  # Load the dataset into a pandas DataFrame
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
  df = pd.read_csv(url)
  X = df.drop(columns=['Appliances', 'date', 'lights'])  # Remove 'Appliances', 'date', and 'lights' columns as features
  y = df['Appliances'].values  # Target variable is 'Appliances'

  threshold = 0.80
  X_reduced, removed_feature_names = remove_highly_correlated_features_pd(X,threshold)
  X = X_reduced
  print(removed_feature_names)

  print("Shape of X:", X.shape)
  print("Shape of y:", y.shape)
  print(X)
  # Scale features
  X_min = X.min(axis=0)
  X_max = X.max(axis=0)
  X_scaled = (X - X_min) / (X_max - X_min)
  X = X_scaled

  return X, y


def get_housing_XY():
  df = pd.read_csv('housing_train.csv', sep=',')
  df.head()
  # List of columns representing categorical variables
  categorical_columns = ['Id','MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
                        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
                        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                        'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
                        'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType',
                        'SaleCondition']
  # Remove categorical columns from the DataFrame
  numeric_df = df.drop(columns=categorical_columns, errors='ignore')
  numeric_df = numeric_df.interpolate()
  X = numeric_df.iloc[:, :-1]  # All columns except the last one
  y = numeric_df.iloc[:, -1].values   # Last column

  threshold = 0.80
  X_reduced, removed_feature_names = remove_highly_correlated_features_pd(X,threshold)
  X = X_reduced
  print(removed_feature_names)

  print("Shape of X:", X.shape)
  print("Shape of y:", y.shape)
  print(X)
  # Scale features
  X_min = X.min(axis=0)
  X_max = X.max(axis=0)
  X_scaled = (X - X_min) / (X_max - X_min)
  X = X_scaled
  return X, y



def get_XY(dataset_name):
  if dataset_name == 'housing':
    X, y = get_housing_XY()
  elif dataset_name == 'energy':
    X, y = get_energy_XY()
  else:
    print("Wrong Dataset Name")
    X = y = None

  return X, y


num_folds = 5
num_runs = 5
num_bootstraps = 1000
save_dir = 'regularization_results/'

dataset_names = ['energy', "housing"]
lambda_inits = [1, 0.5, 0.1, 0.01]  # [1, 5, 10, 20]  #Different lambda_init values

for dataset_name in dataset_names:
    X, y = get_XY(dataset_name)  # get your datas

    # Initialize OLS and ARD models once per dataset (outside lambda_init loop)
    ols_model = LinearRegression()  # Ordinary Least Squares
    ard_model = ARDRegression()  # ARD does not depend on lambda_init

    selected_features_dict = {}
    rmse_dict = {}
    entropy_dict = {}
    coeffs_dict = {}

    # Loop through lambda_inits for SEM, Bayesian Ridge, Ridge, and Lasso
    for lambda_init in lambda_inits:
        pkl_filename = f'{save_dir}{dataset_name}_baySEM.pkl'

        if os.path.exists(pkl_filename):
            print(f"{pkl_filename} Exists... skipping...")
            continue

        # Initialize models that depend on lambda_init
        sem_model = SEMBayesianRidge(lambda_init=lambda_init, use_sign_entropy_elimination=True, slack=1e-10, skip_iters=0)
        bayesian_ridge_model = BayesianRidge(lambda_init=lambda_init)  # Traditional Bayesian Ridge
        lasso_model = Lasso(alpha=lambda_init)  # Lasso with alpha = lambda_init
        ridge_model = Ridge(alpha=lambda_init)  # Ridge with alpha = lambda_init

        for i in range(num_runs):
            print(f"#############{dataset_name}####{lambda_init}############# {i}")
            data_parts = split_data_into_parts(X, y, num_folds)

            for j, test_data in enumerate(data_parts):
                # Use the remaining parts as the training set
                train_data = [part for k, part in enumerate(data_parts) if k != j]

                # Concatenate the training parts to form the full training set
                X_train = np.vstack([part[0] for part in train_data])  # Stack the X values
                y_train = np.hstack([part[1] for part in train_data])  # Stack the y values

                # Extract the test set
                X_test, y_test = test_data

                # Bootstrap sampling for 1000 iterations
                indices_bs = [random.choices(range(X_train.shape[0]), k=X_train.shape[0]) for _ in range(num_bootstraps)]

                # Compute RMSE and sign entropy for SEM Bayesian Ridge
                rmse_sem, sign_entropies_sem, coeffs_bs_sem = compute_coss_rmse(sem_model, X_train, y_train, X_test, y_test, indices_bs)
                rmse_dict.setdefault('sem_' + str(int(lambda_init * 10)), []).append(rmse_sem)
                entropy_dict.setdefault('sem_' + str(int(lambda_init * 10)), []).append(sign_entropies_sem)
                coeffs_dict.setdefault('sem_' + str(int(lambda_init * 10)), []).append(coeffs_bs_sem)

                # Compute RMSE and sign entropy for Bayesian Ridge (without SEM)
                rmse_br, sign_entropies_br, coeffs_bs_br = compute_coss_rmse(bayesian_ridge_model, X_train, y_train, X_test, y_test, indices_bs)
                rmse_dict.setdefault('br_' + str(int(lambda_init * 10)), []).append(rmse_br)
                entropy_dict.setdefault('br_' + str(int(lambda_init * 10)), []).append(sign_entropies_br)
                coeffs_dict.setdefault('br_' + str(int(lambda_init * 10)), []).append(coeffs_bs_br)

                # Compute RMSE and sign entropy for Lasso
                rmse_lasso, sign_entropies_lasso, coeffs_bs_lasso = compute_coss_rmse(lasso_model, X_train, y_train, X_test, y_test, indices_bs)
                rmse_dict.setdefault('lasso_' + str(int(lambda_init * 10)), []).append(rmse_lasso)
                entropy_dict.setdefault('lasso_' + str(int(lambda_init * 10)), []).append(sign_entropies_lasso)
                coeffs_dict.setdefault('lasso_' + str(int(lambda_init * 10)), []).append(coeffs_bs_lasso)

                # Compute RMSE and sign entropy for Ridge
                rmse_ridge, sign_entropies_ridge, coeffs_bs_ridge = compute_coss_rmse(ridge_model, X_train, y_train, X_test, y_test, indices_bs)
                rmse_dict.setdefault('ridge_' + str(int(lambda_init * 10)), []).append(rmse_ridge)
                entropy_dict.setdefault('ridge_' + str(int(lambda_init * 10)), []).append(sign_entropies_ridge)
                coeffs_dict.setdefault('ridge_' + str(int(lambda_init * 10)), []).append(coeffs_bs_ridge)

                # Print results for SEM, Bayesian Ridge, Lasso, Ridge
                print(f"SEM CoSS: {np.mean(sign_entropies_sem):.4f}  BR CoSS: {np.mean(sign_entropies_br):.4f}  "
                      f"Lasso CoSS: {np.mean(sign_entropies_lasso):.4f}  Ridge CoSS: {np.mean(sign_entropies_ridge):.4f}")
                print(f"SEM RMSE: {rmse_sem:.4f}  BR RMSE: {rmse_br:.4f}  "
                      f"Lasso RMSE: {rmse_lasso:.4f}  Ridge RMSE: {rmse_ridge:.4f}")

    res_dict = {'rmse': rmse_dict, 'entropy': entropy_dict, 'coeffs': coeffs_dict}

    print(pkl_filename)
    with open(pkl_filename, 'wb') as f1:
        pickle.dump(res_dict, f1)

    del selected_features_dict, rmse_dict, entropy_dict, coeffs_dict