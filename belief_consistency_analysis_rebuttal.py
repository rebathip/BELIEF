import os
import glob
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import rbo
from collections import defaultdict
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError
from collections import Counter

# === FUNCTIONS FOR ASFE + ARSC + CCM ===
def calculate_pos_entropy(all_pos, all_neg, num_features):
    if len(all_pos) != len(all_neg):
        print("error")
        return 0

    total_runs = len(all_pos)
    pos_features = np.unique(np.concatenate(all_pos))
    neg_features = np.unique(np.concatenate(all_neg))
    merged_features = np.unique(np.concatenate((pos_features, neg_features)))

    pos_counter = Counter(np.concatenate(all_pos))
    neg_counter = Counter(np.concatenate(all_neg))

    sign_entropies = []
    for f in merged_features:
        pos_count = pos_counter[f]
        neg_count = neg_counter[f]
        p_pos = pos_count / (pos_count + neg_count)
        p_neg = neg_count / (pos_count + neg_count)

        if p_pos == 0 or p_neg == 0:
            entropy = 0
        else:
            entropy = -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)
        sign_entropies.append(entropy)

    avg_entropy = np.sum(sign_entropies) / num_features
    if len(merged_features) < 0.5 * num_features and avg_entropy == 0:
        avg_entropy += 0.1 * ((num_features - len(merged_features)) / num_features)
    return sign_entropies, avg_entropy


def calculate_mean_rbo(pos_all, neg_all, persistence=0.2):
    def pairwise_rbo(lists):
        total_rbo, count = 0, 0
        for i in range(len(lists)):
            for j in range(len(lists)):
                l1, l2 = lists[i], lists[j]
                if l1 is None or l2 is None or len(l1) == 0 or len(l2) == 0:
                    total_rbo += 1 if len(l1) == 0 and len(l2) == 0 else 0
                else:
                    total_rbo += rbo.RankingSimilarity(l1, l2).rbo_ext(p=persistence)
                count += 1
        return total_rbo / count if count > 0 else 1.0

    return pairwise_rbo(pos_all), pairwise_rbo(neg_all)


def calculate_asfe_and_arsc(files, img_info_dict, model_name):
    all_asfe, all_arsc = [], []

    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)

        for img in data.keys():
            run_dict = data[img][0]
            num_segments = len(np.unique(img_info_dict[model_name + "_" + img.split("run_")[1]][0]['segments']))

            all_pos, all_neg = [], []
            for run in run_dict:
                run_info = run_dict[run][0]
                key_pos = 'pos' if 'pos' in run_info else 'pos_borda'
                key_neg = 'neg' if 'neg' in run_info else 'neg_borda'

                if run_info[key_pos] is not None:
                    all_pos.append(np.sort(run_info[key_pos]))
                if run_info[key_neg] is not None:
                    all_neg.append(np.sort(run_info[key_neg]))

            # ASFE
            if all(np.array_equal(p, all_pos[0]) for p in all_pos) and all(np.array_equal(n, all_neg[0]) for n in all_neg):
                sign_entropy = 0
            else:
                sign_entropy = calculate_pos_entropy(all_pos, all_neg, num_segments)[1]

            all_asfe.append(sign_entropy)

            # ARSC
            rbo_pos, rbo_neg = calculate_mean_rbo(all_pos, all_neg)
            arsc = (rbo_pos + rbo_neg) / 2
            all_arsc.append(arsc)

    return np.array(all_asfe), np.array(all_arsc)


# === PLOTTING CCM ECDFs FOR EACH ZETA ===
# Set professional style
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# Setup
zeta_range = [f"zeta{i}" for i in [0,1,5,9,10]]  # zeta0 to zeta6
results_dir = "Z:/Desktop/bayesian_sem_regression/vit_slice/results_rebuttal/oxpets/semlime_resnet50/"
img_info_filepath = "img_info_dict/imgs_info.pkl"
model_name = "resnet50"
dataset = "oxpets"

with open(img_info_filepath, 'rb') as f:
    img_info_dict = pickle.load(f)

# ccm_dict = {}
#
# for zeta in zeta_range:
#     pattern = os.path.join(results_dir, f"{dataset}_semlime_*_{model_name}_{zeta}.pkl")
#     files = glob.glob(pattern)
#     print(f"[{zeta}] Found {len(files)} files")
#
#     if not files:
#         continue
#
#     asfe, arsc = calculate_asfe_and_arsc(files, img_info_dict, model_name)
#     ccm_scores = (1 - asfe) #arsc * (1 - asfe)
#     ccm_scores = ccm_scores[~np.isnan(ccm_scores)]
#     ccm_dict[zeta] = ccm_scores
#     print(f"{zeta}: CCM mean={np.round(np.mean(ccm_scores), 3)}, n={len(ccm_scores)}")
#
# pkl_filename = 'fidelity_results_rebuttal/' + '1-asfe_rebuttal.pkl'
#
# with open(pkl_filename, "wb") as f:
#     pickle.dump(ccm_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# === PLOT ===
metric_name = 'ccm'
pkl_filename = 'fidelity_results_rebuttal/' + metric_name + '_rebuttal.pkl'
with open(pkl_filename, "rb") as f:
    ccm_dict = pickle.load(f)

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.weight'] = 'bold'

plt.figure(figsize=(5, 4))
colors = sns.color_palette("tab10", n_colors=len(ccm_dict))

for i, zeta in enumerate(sorted(ccm_dict.keys(), key=lambda x: int(x.replace("zeta", "")))):
    sns.ecdfplot(ccm_dict[zeta], label=zeta, color=colors[i], linewidth=2)

plt.xlabel(f"{metric_name.upper()} Score", fontsize=14, fontweight='bold')
plt.ylabel("Cumulative Probability", fontsize=14, fontweight='bold')
#plt.title(f"ECDF of CCM Scores for Zetas (SEMLIME, {model_name}, {dataset})", fontsize=14, fontweight='bold')
plt.legend(title="Zeta Variant")
plt.tight_layout()

os.makedirs("plots_zeta", exist_ok=True)
plt.savefig(f"plots_zeta/{metric_name}_ecdf_zeta0to6.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()
