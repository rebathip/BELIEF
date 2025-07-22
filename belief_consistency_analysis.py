import pickle
import os
import glob
import numpy as np
from collections import Counter
import rbo

import os
import pickle
import numpy as np
import cv2
import keras
#from keras.applications.imagenet_utils import decode_predictions
import skimage.io
from numpy.linalg import LinAlgError
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


from matplotlib import pyplot as plt
import time
from sklearn.utils import resample
from scipy.stats import norm, gaussian_kde
from sklearn.neighbors import KernelDensity
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pingouin as pg


import matplotlib.colors as mcolors
from skimage.transform import resize


def calculate_pos_neg_probs(data):
    try:
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

    except LinAlgError as e:
        positive_probability = np.nan
        negative_probability = np.nan

    return positive_probability, negative_probability


def calculate_pos_entropy(all_pos, all_neg, num_features):
    if (len(all_pos) != len(all_neg)):  # check if num_runs is equal or not
        print("error")
    else:
        total_runs = len(all_pos)
        # print(total_runs)
        pos_features = np.unique(np.concatenate(all_pos))
        neg_features = np.unique(np.concatenate(all_neg))
        merged_unique_features = np.unique(np.concatenate((pos_features, neg_features)))

        pos_features_dist = np.concatenate(all_pos)
        neg_features_dist = np.concatenate(all_neg)

        pos_counter = Counter(pos_features_dist)
        neg_counter = Counter(neg_features_dist)

        sign_entropies = []
        for feature in merged_unique_features:
            pos_count = pos_counter[feature]
            neg_count = neg_counter[feature]
            positive_probability = pos_count / (pos_count + neg_count)
            negative_probability = neg_count / (pos_count + neg_count)
            # print(f"Feature: {feature} Num Pos: {positive_probability}, Num Neg: {negative_probability}")

            if positive_probability == 0 or negative_probability == 0:
                sign_entropy = 0
            else:
                sign_entropy = -positive_probability * np.log2(positive_probability) \
                               - negative_probability * np.log2(negative_probability)

            sign_entropies.append(sign_entropy)
            # print(str(feature) + ":" + str(sign_entropy))

        # print(sign_entropies)
        # print(np.mean(np.array(sign_entropies)))
        avg_sign_entropy = (np.sum(np.array(sign_entropies))) / num_features

        if len(merged_unique_features) < 0.5 * num_features and avg_sign_entropy == 0:
            avg_sign_entropy = avg_sign_entropy + 0.1 * ((num_features - len(merged_unique_features)) / num_features)

    return sign_entropies, avg_sign_entropy


def calculate_asfe(files, dataset, model_name=""):
    sign_entropies = []

    for file in files:
        # img_name = (file.split("\\")[-1]).split(".jpg")[0].split("baylime_" or "slice_" or "sliceblur_" or "slicefe_" or "lime_")[1]
        if "inceptionv3" in file:
            model_name = "inceptionv3"
        elif "resnet50" in file:
            model_name = "resnet50"
        elif "vitp16" in file:
            model_name = "resnet50"
        else:
            return None

        with open(file, 'rb') as f:
            data = pickle.load(f)
            imgs = data.keys()
            # dataset = 'STABLELIME_OXPETS'  # "LIME_OXPETS" #'STABLELIME_OXPETS'# file.split('//')[1].split('\\')[1].split('_')[0]
            # print("Keys:", imgs)
            # values = data.values()
            # print("Values:", values)
            for img in imgs:
                #img_info_key = file.split("_")[-1].replace(".pkl", "") + "_" + img.split("run_")[1]
                img_info_key = model_name + "_" + img.split("run_")[1]
                # print(file)
                img_info = img_info_dict[img_info_key][0]
                num_segments = len(np.unique(img_info['segments']))
                run_dict = data[img][0]
                # print(f"Key: {img}")
                # print(f"Num runs: {len(run_list)}\n")
                runs = run_dict.keys()
                # print("Runs:", runs)
                all_pos = []
                all_neg = []
                for run in runs:
                    run_info = run_dict[run]
                    # print("Run:", run)
                    # print("Run_info:", run_info)
                    # print("Pos:", run_info[0]['pos'])
                    selected_key_pos = 'pos' if 'pos' in run_info[0] else 'pos_borda' if 'pos_borda' in run_info[
                        0] else None
                    selected_key_neg = 'neg' if 'neg' in run_info[0] else 'neg_borda' if 'neg_borda' in run_info[
                        0] else None

                    if run_info[0][selected_key_pos] is not None:
                        all_pos.append(np.sort(run_info[0][selected_key_pos]))
                    if run_info[0][selected_key_neg] is not None:
                        all_neg.append(np.sort(run_info[0][selected_key_neg]))

                    # all_pos.append(np.sort(run_info[0][selected_key_pos]))
                    # all_neg.append(np.sort(run_info[0][selected_key_neg]))
                    # print(run_info[0]['neg'])

                if all(np.array_equal(pos, all_pos[0]) for pos in all_pos) and all(
                        np.array_equal(neg, all_neg[0]) for neg in all_neg):
                    # print(
                    #    f"dataset:{dataset} img: {img.split('run_')[1]}" + ": Entropy is 0 as pos and neg are same across all runs")
                    sign_entropy = 0
                else:
                    # print(
                    #    f"dataset:{dataset} img: {img.split('run_')[1]}" + f": Entropy is {np.round(calculate_pos_entropy(all_pos, all_neg)[1], 3)}")
                    sign_entropy = calculate_pos_entropy(all_pos, all_neg, num_segments)[1]

                print(f"img: {img} : sign_entropy={sign_entropy}")
                sign_entropies.append(sign_entropy)

    return sign_entropies

# RBO
def calculate_mean_rbo(pos_all, neg_all, persistence=0.2):
    sum_rbo_pos = 0
    sum_rbo_neg = 0

    count_pos = 0
    for i in np.arange(0, len(pos_all), step=1):
        for j in np.arange(0, len(pos_all), step=1):
            list1 = pos_all[i]
            list2 = pos_all[j]
            # print(rbo.RankingSimilarity(list1, list2).rbo(p=persistence))
            count_pos = count_pos + 1

            if list1 is None or len(list1) == 0 or list2 is None or len(list2) == 0:
                if len(list1) == 0 and len(list2) == 0:
                    sum_rbo_neg +=1
                else:
                    sum_rbo_pos += 0
            else:
                sum_rbo_pos += rbo.RankingSimilarity(list1, list2).rbo_ext(p=persistence)

    count_neg = 0
    for i in np.arange(0, len(neg_all), step=1):
        for j in np.arange(0, len(neg_all), step=1):
            list1 = neg_all[i]
            list2 = neg_all[j]
            # print(rbo.RankingSimilarity(list1, list2).rbo(p=persistence))
            if list1 is None or len(list1) == 0 or list2 is None or len(list2) == 0:
                if len(list1) == 0 and len(list2) == 0:
                    sum_rbo_neg +=1
                else:
                    sum_rbo_neg += 0
            else:
                sum_rbo_neg += rbo.RankingSimilarity(list1, list2).rbo_ext(p=persistence)

            count_neg = count_neg + 1
    return sum_rbo_pos / count_pos, sum_rbo_neg / count_neg


def calculate_arsc(files, dataset):
    arsc_scores = []
    for file in files:
        sign_entropy = -1
        with open(file, 'rb') as f:
            data = pickle.load(f)
            imgs = data.keys()
            dataset = 'STABLELIME_PVOC'  # "LIME_OXPETS" #'STABLELIME_OXPETS'# file.split('//')[1].split('\\')[1].split('_')[0]
            # print("Keys:", imgs)
            # values = data.values()
            # print("Values:", values)
            for img in imgs:
                run_dict = data[img][0]
                # print(f"Key: {img}")
                # print(f"Num runs: {len(run_list)}\n")
                runs = run_dict.keys()
                # print("Runs:", runs)
                all_pos = []
                all_neg = []
                for run in runs:
                    run_info = run_dict[run]
                    # print("Run:", run)
                    # print("Run_info:", run_info)
                    # print("Pos:", run_info[0]['pos'][::-1])
                    # print(img+str(run_info[0]['pos'][::-1]))
                    selected_key_pos = 'pos' if 'pos' in run_info[0] else 'pos_borda' if 'pos_borda' in run_info[
                        0] else None
                    selected_key_neg = 'neg' if 'neg' in run_info[0] else 'neg_borda' if 'neg_borda' in run_info[
                        0] else None

                    all_pos.append(run_info[0][selected_key_pos])
                    all_neg.append(run_info[0][selected_key_neg])
                    # print(run_info[0]['neg'])

                mean_rbo_pos, mean_rbo_neg = calculate_mean_rbo(all_pos, all_neg)

                if len(all_pos) == 0 or all(arr.size == 0 for arr in all_pos):
                    mean_rbo_pos = 1
                if len(all_neg) == 0 or all(arr.size == 0 for arr in all_neg):
                    mean_rbo_neg = 1

                mean_rbo = (mean_rbo_pos + mean_rbo_neg)/2
                #if "semlime" in file and mean_rbo < 0.6:
                #    print(img)

                arsc_scores.append(mean_rbo)
                print(f"{img} rbo: {mean_rbo}")
                # print(calculate_mean_rbo(all_pos, all_neg))

    return arsc_scores


img_info_filepath = "img_info_dict/imgs_info.pkl"
with open(img_info_filepath, 'rb') as f:
    img_info_dict = pickle.load(f)

datasets_root_dir = "results/"
datasets = os.listdir(datasets_root_dir)


asfe_dict = {}

for dataset in datasets:
    xm_path = datasets_root_dir + str(dataset)
    xm_pairs = os.listdir(xm_path) #xai-model pairs
    #print(xm_pairs)
    for xm_pair in xm_pairs:
        xai, model_name = xm_pair.split("_")[0], xm_pair.split("_")[1]
        if xai == "semlime" and dataset == 'oxpets':
            result_key = dataset + '_' + model_name + '_' + xai
            pkl_paths = xm_path + "/" + xm_pair + "/*.pkl"
            pkl_files = glob.glob(pkl_paths)
            #print(result_key)
            entropies = calculate_asfe(pkl_files, result_key, model_name)
            print(result_key + ":" + str(np.mean(entropies)))
            asfe_dict[result_key] = np.array(entropies)


datasets = os.listdir(datasets_root_dir)

arsc_dict = {}
for dataset in datasets:
    xm_path = datasets_root_dir + str(dataset)
    xm_pairs = os.listdir(xm_path) #xai-model pairs
    #print(xm_pairs)
    for xm_pair in xm_pairs:
        xai, model_name = xm_pair.split("_")[0], xm_pair.split("_")[1]
        result_key = dataset + '_' + model_name + '_' + xai
        pkl_paths = xm_path + "/" + xm_pair + "/*.pkl"
        pkl_files = glob.glob(pkl_paths)
        arsc_score = calculate_arsc(pkl_files, result_key)
        arsc_dict[result_key] = np.array(arsc_score)

final_scores_dict = {}
xai_keys = asfe_dict.keys()
for xai_key in xai_keys:
    arsc_scores = arsc_dict[xai_key]
    asfe_scores = asfe_dict[xai_key]

    final_scores = arsc_scores * (1 - asfe_scores)
    print(xai_key, "#######################", final_scores[final_scores < 0])
    # print(1 - asfe_scores)
    # print(xai_key + "final" + str(final_scores))
    final_scores_dict[xai_key] = final_scores

#### Plot CCM scores

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.weight'] = 'bold'

# Define the size for each subplot
subplot_size = 3  # ensuring the subplot is square

# Create a new figure
fig, axes = plt.subplots(2, 2, figsize=(2 * subplot_size, 2 * subplot_size))

title_index = ['Oxford-IIIT Pets:InceptionV3', 'Oxford-IIIT Pets:ResNet50', 'Pascal VOC2007:Inception V3',
               'Pascal VOC2007:ResNet50']
slice_keys = ['oxpets_inceptionv3_sliceblurfe', 'oxpets_resnet50_sliceblurfe', 'pvoc_inceptionv3_sliceblurfe',
              'pvoc_resnet50_sliceblurfe']
baylime_keys = ['oxpets_inceptionv3_baylime', 'oxpets_resnet50_baylime', 'pvoc_inceptionv3_baylime',
                'pvoc_resnet50_baylime']
lime_keys = ['oxpets_inceptionv3_lime', 'oxpets_resnet50_lime', 'pvoc_inceptionv3_lime', 'pvoc_resnet50_lime']
belief_keys = ['oxpets_inceptionv3_semlime', 'oxpets_resnet50_semlime', 'pvoc_inceptionv3_semlime',
               'pvoc_resnet50_semlime']

dict_analysis = final_scores_dict
key_index = 0
handles_list = []
labels_list = []


# Function to perform Wilcoxon Signed-Rank test using pingouin
def wilcoxon_test_with_pingouin(data1, data2, alternative='greater'):
    # Calculate the difference and median of differences
    differences = np.array(data1) - np.array(data2)
    median_diff = np.median(differences)

    # Run the Wilcoxon test using pingouin with specified alternative hypothesis
    result = pg.wilcoxon(data1, data2, alternative=alternative)
    w_value = result['W-val'][0]
    p_value = result['p-val'][0]
    effect_size_rbc = result['RBC'][0]  # Rank-biserial correlation effect size
    effect_size_cles = result['CLES'][0]  # Common Language Effect Size
    return w_value, p_value, effect_size_rbc, effect_size_cles, median_diff


# Code for plotting remains the same
for ax in axes.ravel():
    # Extract data for each method
    belief_data = dict_analysis[belief_keys[key_index]]
    slice_data = dict_analysis[slice_keys[key_index]]
    baylime_data = dict_analysis[baylime_keys[key_index]]
    lime_data = dict_analysis[lime_keys[key_index]]

    # Perform Wilcoxon Signed-Rank test and get effect sizes for each comparison
    w_val_lime, p_val_lime, effect_rbc_lime, cles_lime, median_diff_lime = wilcoxon_test_with_pingouin(belief_data,
                                                                                                       lime_data,
                                                                                                       alternative='greater')
    w_val_baylime, p_val_baylime, effect_rbc_baylime, cles_baylime, median_diff_baylime = wilcoxon_test_with_pingouin(
        belief_data, baylime_data, alternative='greater')
    w_val_slice, p_val_slice, effect_rbc_slice, cles_slice, median_diff_slice = wilcoxon_test_with_pingouin(belief_data,
                                                                                                            slice_data,
                                                                                                            alternative='two-sided')

    # Print results for each comparison with detailed statistics
    print(
        f"{title_index[key_index]} - SEMLIME vs LIME: W-val={w_val_lime:.3f}, p-value={p_val_lime:.3e}, RBC={effect_rbc_lime:.3f}, CLES={cles_lime:.3f}, Median Diff={median_diff_lime:.3f}")
    print(
        f"{title_index[key_index]} - SEMLIME vs BayLIME: W-val={w_val_baylime:.3f}, p-value={p_val_baylime:.3e}, RBC={effect_rbc_baylime:.3f}, CLES={cles_baylime:.3f}, Median Diff={median_diff_baylime:.3f}")
    print(
        f"{title_index[key_index]} - SEMLIME vs SLICE: W-val={w_val_slice:.3f}, p-value={p_val_slice:.3e}, RBC={effect_rbc_slice:.3f}, CLES={cles_slice:.3f}, Median Diff={median_diff_slice:.3e}")

    # Plot ECDF for each method
    sns.ecdfplot(belief_data, color='orange', linewidth=1, ax=ax, label='SEMLIME')
    sns.ecdfplot(slice_data, color='blue', linewidth=1, ax=ax, label='SLICE')
    sns.ecdfplot(baylime_data, color='teal', linewidth=1, ax=ax, label='BayLIME')
    sns.ecdfplot(lime_data, color='purple', linewidth=1, ax=ax, label='LIME')

    # Set titles and labels
    ax.set_title(f"{title_index[key_index]}", fontsize=12, fontweight='bold')
    ax.set_xlabel("CCM Scores", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight='bold')

    # Add legend items
    handles, labels = ax.get_legend_handles_labels()
    handles_list.extend(handles)
    labels_list.extend(labels)

    key_index += 1

# Display a unique legend for all subplots
unique = [(h, l) for i, (h, l) in enumerate(zip(handles_list, labels_list)) if l not in labels_list[:i]]
fig.legend(*zip(*unique), loc='lower center', bbox_to_anchor=(0.5, -.05), fontsize=12, ncol=4)

# Adjust layout and display
plt.tight_layout()
plt.savefig('plots/ecdf_ccm_belief_slice_lime_baylime.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

