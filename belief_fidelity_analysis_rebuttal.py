import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from collections import defaultdict

# Settings
results_dir = "fidelity_results_rebuttal/ins/"
dataset_name = "oxpets"
model_name = "resnet50"
xai_name = "semlime"

# Set style
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.weight'] = 'bold'

# Load all zeta-based scores into a dictionary
zeta_scores = defaultdict(list)

# Read and group scores from files
for file in os.listdir(results_dir):
    if file.endswith('.pkl'):
        with open(os.path.join(results_dir, file), 'rb') as f:
            data = pickle.load(f)

        for key, value in data.items():
            if key.endswith('.jpg') and xai_name in key and model_name in key and dataset_name in key:
                zeta_tag = key.split('_')[0]  # e.g., 'zeta0'
                if isinstance(value, np.ndarray):
                    zeta_scores[zeta_tag].append(np.mean(value[0]))
                elif isinstance(value, list) and all(isinstance(v, dict) for v in value):
                    zeta_scores[zeta_tag].append(np.mean([np.mean(d['pos']) for d in value]))

# Plot ECDFs for all zetas
plt.figure(figsize=(7, 5))

colors = sns.color_palette('tab10', n_colors=len(zeta_scores))
for i, zeta in enumerate(sorted(zeta_scores.keys(), key=lambda x: int(x.replace("zeta", "")))):
    scores = np.array(zeta_scores[zeta])
    sns.ecdfplot(scores, label=zeta, linewidth=2, color=colors[i])

plt.xlabel("AOPC Insertion Scores", fontsize=14, fontweight='bold')
plt.ylabel("Cumulative Probability", fontsize=14, fontweight='bold')
plt.title(f"ECDFs for {xai_name.upper()} ({model_name}, {dataset_name})", fontsize=14, fontweight='bold')
plt.legend(title="Zeta Variants")
plt.tight_layout()
os.makedirs('plots_zeta', exist_ok=True)
#plt.savefig(f'plots_zeta/ecdf_zeta_comparison_{xai_name}_{model_name}.pdf', format='pdf', dpi=300)
plt.show()
