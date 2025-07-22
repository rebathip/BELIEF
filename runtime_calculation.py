import time
import os
import pickle
import numpy as np
import tensorflow as tf
from slice.slice_explainer import SliceExplainer
from slice.vit_img_classifier import ViTImageClassifier

# Initialize models
resnet_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
inceptionv3_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
model_names = ['resnet50', 'inceptionv3']

# Load a sample image and segments (assuming these are already defined)
sample_size = 500

img_info_filepath = "../vit_slice/img_info_dict/imgs_info.pkl"
with open(img_info_filepath, 'rb') as f:
    img_info_dict = pickle.load(f)


# Function to get model parameters
def get_model_params(model_name):
    if model_name == 'resnet50':
        model = resnet_model
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        target_img_size = (224, 224)
    elif model_name == 'inceptionv3':
        model = inceptionv3_model
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        target_img_size = (299, 299)
    else:
        raise ValueError("Unsupported model name")
    return model, preprocess_input, target_img_size

# Define functions to measure runtime for each method
def measure_runtime_lime(model, img_path, segments, preprocess_input, target_img_size, sigma):
    start_time = time.time()
    exp = SliceExplainer(image_path=img_path, segments=segments, model=model, target_img_size=target_img_size,
                         preprocess=preprocess_input)
    pos_feature_ranks, neg_feature_ranks, pos_dict, neg_dict = exp.get_lime_explanations(sigma=sigma)
    end_time = time.time()
    del exp
    return end_time - start_time

def measure_runtime_semlime(model, img_path, segments, preprocess_input, target_img_size, sigma):
    start_time = time.time()
    exp = SliceExplainer(image_path=img_path, segments=segments, model=model, target_img_size=target_img_size, preprocess=preprocess_input)
    pos_feature_ranks, neg_feature_ranks, pos_dict, neg_dict = exp.get_belief_explanations(sigma=sigma)
    end_time = time.time()
    del exp
    return end_time - start_time

def measure_runtime_slice(model, img_path, segments, preprocess_input, target_img_size, sigma):
    start_time = time.time()
    exp = SliceExplainer(image_path=img_path, segments=segments, model=model, target_img_size=target_img_size, preprocess=preprocess_input)
    # Assuming SLICE explanation is similar to SEMLIME for testing
    _, pos_feature_ranks, neg_feature_ranks, \
    samples_used, pos_dict, neg_dict = exp.get_slice_explanations(sigma=sigma)
    end_time = time.time()
    del exp
    print(f"{img_path} Samples Used in Slice {samples_used}")
    return end_time - start_time

def measure_runtime_baylime(model, img_path, segments, preprocess_input, target_img_size, sigma):
    start_time = time.time()
    exp = SliceExplainer(image_path=img_path, segments=segments, model=model, target_img_size=target_img_size, preprocess=preprocess_input)
    # Assuming SLICE explanation is similar to SEMLIME for testing
    pos_feature_ranks, neg_feature_ranks, pos_dict, neg_dict = exp.get_baylime_explanations(sigma=sigma)
    end_time = time.time()
    del exp
    return end_time - start_time


img_dir = "../../LIME_Stabilization/final_code/images_all/"

img_paths = os.listdir(img_dir)

img_paths = img_paths[0:1]

runtime_dict = {}
for model_name in model_names:
    model, preprocess_input, target_img_size = get_model_params(model_name)
    print(f"Testing runtime for {model_name} model:")

    lime_times = []
    semlime_times = []
    baylime_times = []
    slice_times = []
    for img_path in img_paths:
        # Run each explanation method once for each image
        img_key = model_name + "_" + img_path.split('.')[0]
        img_info = img_info_dict[img_key]
        segments = img_info[0]['segments']
        sel_sigma = img_info[0]['sel_sigma']
        lime_times.append(measure_runtime_lime(model, img_dir+img_path, segments, preprocess_input, target_img_size, sel_sigma))
        semlime_times.append(measure_runtime_semlime(model, img_dir+img_path, segments, preprocess_input, target_img_size, sel_sigma))
        baylime_times.append(measure_runtime_baylime(model, img_dir+img_path, segments, preprocess_input, target_img_size, sel_sigma))
        slice_times.append(measure_runtime_slice(model, img_dir+img_path, segments, preprocess_input, target_img_size, sel_sigma))

    runtime_dict[model_name] = {
        "lime" : np.array(lime_times),
        "semlime" : np.array(semlime_times),
        "baylime": np.array(baylime_times),
        "slice": np.array(slice_times),
    }
    # Display results
    print(f"LIME runtime: {lime_times} seconds")
    print(f"SEMLIME runtime: {semlime_times} seconds")
    print(f"Baylime runtime: {baylime_times} seconds")
    print(f"SLICE runtime: {slice_times} seconds")

with open("runtime_dict.pkl", 'wb') as f1:
    pickle.dump(runtime_dict, f1)


### Runtime Plots

import numpy as np
from scipy.stats import gaussian_kde

# Open the log file and read it line by line
with open('runtimelog.out', 'r') as log_file:
    resnet50_samples = []
    inceptionv3_samples = []
    current_samples = None  # Track which model's samples are being recorded

    # Loop through each line in the file
    for line in log_file:
        # Check if the line indicates the model type
        if line.startswith("Testing runtime for resnet50 model:"):
            current_samples = resnet50_samples  # Start accumulating for resnet50
        elif line.startswith("Testing runtime for inceptionv3 model:"):
            current_samples = inceptionv3_samples  # Switch to accumulate for inceptionv3

        # Check if the line starts with the specified path and extract the sample size
        elif line.startswith("../../LIME_Stabilization/final_code/") and current_samples is not None:
            txt = line.strip()
            # Extract the last element (sample size) and convert to an integer if needed
            sample_size = int(txt.split(' ')[-1])  # Assuming sample size is always the last element
            current_samples.append(sample_size)


# Define the probability calculation using Gaussian KDE with Scott's bandwidth
def calculate_probability_kde(samples, threshold=500):
    if not samples:
        return 0  # Return zero if the sample list is empty

    # Convert samples to a numpy array
    samples = np.array(samples)

    # Fit KDE with Scott's bandwidth
    kde = gaussian_kde(samples, bw_method='scott')

    # Calculate the cumulative probability for values <= threshold
    probability = kde.integrate_box_1d(-np.inf, threshold)

    return probability


# Calculate probabilities for each sample list
resnet50_probability = calculate_probability_kde(resnet50_samples)
inceptionv3_probability = calculate_probability_kde(inceptionv3_samples)

# print("ResNet50 Samples:", resnet50_samples)
# print("InceptionV3 Samples:", inceptionv3_samples)
print(f"Probability of ResNet50 samples <= 500 (KDE): {resnet50_probability:.3e}")
print(f"Probability of InceptionV3 samples <= 500 (KDE): {inceptionv3_probability:.3e}")


### Pearson Correlation of Runtime vs sample size
resnet_cor = np.corrcoef(runtime_dict['resnet50']['slice'], resnet50_samples)
inceptionv3_cor = np.corrcoef(runtime_dict['inceptionv3']['slice'], inceptionv3_samples)

print(f"Resnet Cor: {resnet_cor} and Inception v3 Cor: {inceptionv3_cor}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set matplotlib parameters
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.weight'] = 'normal'

# Calculate medians
resnet50_median = np.median(resnet50_samples)
inceptionv3_median = np.median(inceptionv3_samples)

# Generate KDE plot for both models with a vertical line at 500
plt.figure(figsize=(6, 5))
sns.kdeplot(resnet50_samples, label="ResNet50", fill=True, color="blue")
sns.kdeplot(inceptionv3_samples, label="InceptionV3", fill=True, color="green")

# Add vertical line at 500 to show the threshold
plt.axvline(500, color='purple', linestyle='--')

# Annotate the threshold line with an arrow
plt.gca().annotate("500", xy=(500, plt.gca().get_ylim()[1] * 0.5),
                   xytext=(-20, 30), textcoords='offset points',
                   arrowprops=dict(arrowstyle="->", lw=0.5, color='purple'),
                   color='maroon', fontsize=15, fontweight='bold', ha='center')

# Plot and annotate the medians for both models
plt.axvline(resnet50_median, color='blue', linestyle='--')
plt.axvline(inceptionv3_median, color='green', linestyle='--')

plt.gca().annotate(f"{int(resnet50_median)}", xy=(resnet50_median, plt.gca().get_ylim()[1] * 0.8),
                   xytext=(55, 15), textcoords='offset points',
                   arrowprops=dict(arrowstyle="->", lw=0.5, color='blue'),
                   color='blue', fontsize=15, fontweight='bold', ha='center')

plt.gca().annotate(f"{int(inceptionv3_median)}", xy=(inceptionv3_median, plt.gca().get_ylim()[1] * 0.6),
                   xytext=(35, 10), textcoords='offset points',
                   arrowprops=dict(arrowstyle="->", lw=0.5, color='green'),
                   color='green', fontsize=15, fontweight='bold', ha='center')

# Format y-axis to exponential notation
plt.gca().yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# Labeling the plot
plt.xlabel("Sample Size", fontweight='bold')
plt.ylabel("Density", fontweight='bold')

# Move legend below the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

plt.tight_layout()
plt.savefig('plots/runtime.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()