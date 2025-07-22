import pickle
import os
import glob
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from scipy.stats import kurtosis, skew
#from sklearn.metrics import plot_confusion_matrix

import seaborn as sns
import os
import pickle
import numpy as np
import cv2
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
from skimage.segmentation import quickshift, mark_boundaries
from skimage.measure import regionprops
import copy
import random
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from skimage import filters
import pandas as pd
import warnings
import tensorflow as tf
import pickle
from scipy.stats import kendalltau
import sys
import scipy.stats as stats
from scipy.stats import wilcoxon
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from functools import partial
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from matplotlib import pyplot as plt
import time
from sklearn.utils import resample
from scipy.stats import norm, gaussian_kde
from sklearn.neighbors import KernelDensity
import csv

import matplotlib.colors as mcolors
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, make_scorer

from sklearn.metrics import roc_curve, auc
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
import shutil
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
import gc
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from scipy.stats import ks_2samp
import numpy as np
import matplotlib as mpl

# Set matplotlib parameters
plt.rcParams['font.family'] = 'Sans-serif'
mpl.rcParams['font.size'] = 40
mpl.rcParams['font.weight'] = 'normal'


def calculate_med_mu(data, key):
  means_per_run = []
  for i in np.arange(0, len(data['entropy'][key]), step=1):
    means_per_run.append(np.mean(data['entropy'][key][i]))

  means_per_run = np.array(means_per_run)
  return np.round(np.median(means_per_run),3), np.round(np.mean(means_per_run),3)

### SIGN Entropy mean and medians
pkl_dir = 'regularization_results/' # put the same path as the save_dir in regularization_run.py

pkl_files = os.listdir(pkl_dir)
pkl_files = [file for file in os.listdir(pkl_dir) if file.endswith('.pkl')]
pkl_files = ['energy_baySEM_sl_0001_sk_1.pkl','housing_baySEM.pkl']
for pkl_file in pkl_files:
  with open(pkl_dir + pkl_file, 'rb') as f:
    data = pickle.load(f)

  print(pkl_file)
  for key in data['entropy'].keys():
    print(f"FE: {key} Sign Entropy: {calculate_med_mu(data, key)}")


#### Average RMSE calculations
# Initialize an empty dictionary to store min and max RMSE for each dataset
dataset_rmse_stats = {}
pkl_files = ['energy_baySEM_sl_0001_sk_1.pkl','housing_baySEM.pkl']
for pkl_file in pkl_files:
    with open(pkl_dir + pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Extract the dataset name from the pkl file pattern (assuming it's dataset_*.pkl)
    dataset_name = pkl_file.split('_')[0]

    print(f"Processing dataset: {dataset_name} in file: {pkl_file}")

    # Get all RMSE values for this dataset
    all_rmse_values = []
    for key in data['rmse'].keys():
        all_rmse_values.extend(data['rmse'][key])  # Collect all RMSE values from each FE method

    # Find the min and max RMSE for this dataset
    rmse_min = np.min(all_rmse_values)
    rmse_max = np.max(all_rmse_values)

    # Save the min and max RMSE in the dataset_rmse_stats for reference
    dataset_rmse_stats[dataset_name] = (rmse_min, rmse_max)

    # Now normalize and print the normalized RMSE values for each FE method
    for key in data['rmse'].keys():
        rmse_values = data['rmse'][key]
        rmse_normalized = (rmse_values - rmse_min) / (rmse_max - rmse_min)  # Min-max normalization
        print(f"FE: {key} Mean Normalized RMSE: {np.round(np.mean(rmse_normalized),3)}")


########################## Plots and Statistical Tests #####################
############################################################################
def get_rmse_dict(data):
    rmse_dict = {}
    all_rmse_values = []
    for key in data['rmse'].keys():
        all_rmse_values.extend(data['rmse'][key])

    rmse_min = np.min(all_rmse_values)
    rmse_max = np.max(all_rmse_values)

    for key in data['rmse'].keys():
        rmse_values = data['rmse'][key]
        rmse_normalized = (rmse_values - rmse_min) / (rmse_max - rmse_min)
        rmse_dict[key] = rmse_normalized

    return rmse_dict

def get_coss_dict(data):
    coss_dict = {}
    for key in data['entropy'].keys():
        means_per_run = [np.mean(run) for run in data['entropy'][key]]
        coss_dict[key] = np.array(means_per_run)
    return coss_dict


def plot_1_plot(score_dict, metric_name, dataset_name):
    # Remove entries ending with '_200'
    score_dict = {k: v for k, v in score_dict.items() if not k.endswith('_200')}

    # Aggregate data based on patterns in keys
    aggregated_data = defaultdict(list)
    for key, data in score_dict.items():
        pattern = re.match(r'^[a-z]+', key).group(0)  # Extract the prefix (e.g., "sem", "ols", "br")
        aggregated_data[pattern].extend(data)  # Add data to the aggregated dictionary under the pattern

    # Define a dictionary to map pattern names to display names
    display_names = {
        'sem': 'Proposed',
        'ols': 'OLS',
        'ard': 'ARD',
        'br': 'Bayesian\nRidge',
        'lasso': 'Lasso',
        'ridge': 'Ridge'
    }

    # Define colors for each method
    color_map = {
        'sem': 'blue',
        'ols': 'green',
        'ard': 'orange',
        'br': 'purple',
        'lasso': 'red',
        'ridge': 'brown'
    }

    # Define the figure for plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    print(aggregated_data.keys())
    # Plot KDE for each aggregated group with fixed colors
    for pattern, data in aggregated_data.items():
        label = display_names.get(pattern, pattern)
        color = color_map.get(pattern, 'gray')
        sns.kdeplot(data, label=label, color=color, fill=True, ax=ax)

    # Customize KDE plot with larger font sizes for labels and ticks
    ax.set_xlabel(metric_name, fontsize=40, fontweight='normal')
    ax.set_ylabel("Density", fontsize=40, fontweight='normal')
    ax.tick_params(axis='both', which='major', labelsize=30)  # Increase tick label font size

    # Perform KS test: Compare "sem" group to each other group and print results
    sem_data = aggregated_data['sem']
    print("Kolmogorov-Smirnov Test Results (sem vs. other methods):")
    for pattern, data in aggregated_data.items():
        if pattern != 'sem':
            ks_stat, p_value = ks_2samp(sem_data, data)
            print(f"SEM vs {display_names.get(pattern, pattern)}: KS Statistic = {ks_stat:.5f}, p-value = {p_value:.5f}")

    # Sort the legend so "Proposed" (sem) appears at the top
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: 0 if x[1] == "Proposed" else 1)
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    # Place the legend on the right side vertically with the specified order
    #ax.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(dataset_name + '_' + metric_name + '.png')
    plt.show()

#######Energy data
##################
pkl_dir = 'regularization_results/'

dataset_name = "energy" # or ""housing

# Remove items from the list
pkl_files = [] # put the name of the energy or housing pkl file as per dataset_name

print(pkl_files)
coss_gains = []
rmse_losses = []

for pkl_file in pkl_files:
    with open(pkl_dir + pkl_file, 'rb') as f:
        data = pickle.load(f)

    dataset_name = pkl_file.split('_')[0]
    print(f"Processing dataset: {dataset_name} in file: {pkl_file}")

    coss_dict = get_coss_dict(data)
    rmse_dict = get_rmse_dict(data)

    coss_dict = {k: v for k, v in coss_dict.items() if not k.endswith('_200')}
    rmse_dict = {k: v for k, v in rmse_dict.items() if not k.endswith('_200')}


    methods_to_test = ['sem_10', 'sem_50', 'sem_100']
    other_methods = [key for key in coss_dict.keys() if key not in methods_to_test] #['lasso_0', 'lasso_1', 'lasso_5', 'lasso_10']
    #['ols', 'ard', 'br_0', 'br_1', 'br_5', 'br_10', 'ridge_0', 'ridge_1', 'ridge_5', 'ridge_10']#['lasso_0', 'lasso_1', 'lasso_5', 'lasso_10'] #[key for key in coss_dict.keys() if key not in methods_to_test]



plot_1_plot(coss_dict, "ASFE", dataset_name)
plot_1_plot(rmse_dict, "RMSE", dataset_name)


