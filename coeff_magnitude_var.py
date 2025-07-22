import os
import pickle
import numpy as np
from collections import defaultdict

# Path setup
results_dirs = [
    #"Z:/Desktop/bayesian_sem_regression/vit_slice/results/oxpets/sliceblurfe_resnet50/",
    #"Z:/Desktop/bayesian_sem_regression/vit_slice/results/oxpets/semlime_resnet50/",
    "Z:/Desktop/bayesian_sem_regression/vit_slice/results/oxpets/baylime_resnet50/"
    #"Z:/Desktop/bayesian_sem_regression/vit_slice/results/oxpets/lime_resnet50/"
]
xai_names = ["baylime"] #["sliceblurfe", "semlime", "baylime", "lime"]
model_name = "resnet50"
dataset_name = "oxpets"
img_info_path = "img_info_dict/imgs_info.pkl"

# Output dictionary
zeta_mean_mu_dict = defaultdict(list)

# Load image segmentation info
with open(img_info_path, 'rb') as f:
    img_info_dict = pickle.load(f)

# Iterate over result directories
for results_dir, xai_name in zip(results_dirs, xai_names):
    for file in os.listdir(results_dir):
        #if not file.endswith(".pkl") or xai_name not in file or model_name not in file or dataset_name not in file:
        #    continue

        with open(os.path.join(results_dir, file), 'rb') as f:
            data = pickle.load(f)

        for img_key in data:
            image_id = str(img_key).replace('run_', '')  # e.g., beagle_202
            model_key = model_name + "_" + image_id
            if model_key not in img_info_dict:
                continue

            num_segments = len(np.unique(img_info_dict[model_key][0]['segments']))
            img_data = data[img_key]
            coeff_matrix = np.ones((20, num_segments)) * -1

            for run_idx, run_key in enumerate(img_data[0].keys()):
                run_info = img_data[0][run_key][0]

                # Handle pos_dict
                pos_dict = run_info.get('pos_dict', {})
                if 'column_names' in pos_dict and 'column_means' in pos_dict:
                    pos_cols = list(pos_dict['column_names'])  # Convert to list
                    pos_vals = pos_dict['column_means']
                    coeff_matrix[run_idx, pos_cols] = pos_vals

                # Handle neg_dict
                neg_dict = run_info.get('neg_dict', {})
                if 'column_names' in neg_dict and 'column_means' in neg_dict:
                    neg_cols = list(neg_dict['column_names'])  # Convert to list
                    neg_vals = neg_dict['column_means']
                    coeff_matrix[run_idx, neg_cols] = neg_vals

            # Calculate stats
            col_means = np.mean(coeff_matrix, axis=0)
            col_stds = np.std(coeff_matrix, axis=0)

            cov = np.zeros_like(col_means)
            nonzero_mask = col_stds != 0
            cov[nonzero_mask] = abs(col_means[nonzero_mask]) / col_stds[nonzero_mask]
            zero_columns = np.all(coeff_matrix == 0, axis=0)
            cov[zero_columns] = 0

            valid_cols_mask = np.any(coeff_matrix != -1, axis=0)
            valid_means = col_means[valid_cols_mask]
            valid_stds = col_stds[valid_cols_mask]

            zeta_mean_mu_dict[xai_name].append((np.mean(valid_means), np.mean(valid_stds)))

            print(f"{file} ({xai_name}): Mean STD = {np.mean(valid_stds):.4f}")
