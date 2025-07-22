import os
import pickle
import numpy as np
from collections import defaultdict

# Path setup
results_dir = "results_rebuttal/oxpets/semlime_resnet50/"
xai_name = "semlime"
model_name = "resnet50"
dataset_name = "oxpets"
img_info_path = "img_info_dict/imgs_info.pkl"

# Load image segmentation info
with open(img_info_path, 'rb') as f:
    img_info_dict = pickle.load(f)

# Structure to store fraction of features selected: {zeta: [(img_name, fraction), ...]}
zeta_fraction_dict = defaultdict(list)

for file in os.listdir(results_dir):
    if not file.endswith(".pkl") or not xai_name in file or not model_name in file or not dataset_name in file:
        continue

    with open(os.path.join(results_dir, file), 'rb') as f:
        data = pickle.load(f)

    for img_key in data:
        # Extract zeta and image ID
        parts = img_key.split('_')
        zeta = file.split('_')[-1].replace('.pkl','')# e.g., zeta0
        image_id = str(img_key).replace('run_','')  # e.g., beagle_202

        model_key = model_name + "_" + image_id
        if model_key not in img_info_dict:
            continue

        num_segments = len(np.unique(img_info_dict[model_key][0]['segments']))
        img_data = data[img_key]

        frac_sel_segs = []
        for run_key in img_data[0].keys():
            frac_sel_segs.append((len(img_data[0]['run_0'][0]['pos']) + len(img_data[0]['run_0'][0]['neg']))/num_segments)

        if frac_sel_segs:
            avg_fraction = np.mean(frac_sel_segs)
            zeta_fraction_dict[zeta].append((image_id, avg_fraction))


# Example plot for each zeta variant
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
colors = sns.color_palette("tab10", n_colors=10)

for i, zeta in enumerate(sorted(zeta_fraction_dict.keys(), key=lambda x: int(x.replace("zeta", "")))):
    fractions = [frac for _, frac in sorted(zeta_fraction_dict[zeta])]
    plt.plot(fractions, label=rf"$\zeta = {int(zeta.replace('zeta', ''))/10:.2f}$", marker='o', color=colors[i])

plt.xlabel("Image Index", fontsize=13, fontweight='bold')
plt.ylabel("Fraction of Features Selected", fontsize=13, fontweight='bold')
plt.title("Fraction of Selected Features per Image Across Zeta Variants", fontsize=14, fontweight='bold')
plt.legend(title=r"$\zeta$", fontsize=11)
plt.grid(True)
plt.tight_layout()
#plt.savefig("plots_zeta/fraction_selected_zeta_comparison.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()