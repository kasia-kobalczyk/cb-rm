import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
import os
import matplotlib.lines as mlines

# === FONT CONFIG ===
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Times New Roman"],
    "axes.labelsize": 9,
    "font.size": 8,
})
sns.set(style="whitegrid")

# === STYLE CONFIG ===
colors = ["#708090", "#BA55D3", "#20B2AA", "#DC143C"]
markers = ["o", "s", "D", "^"]
linestyles = ["-", "--", "-.", ":"]

# llm_emb = "llama3"
# metric_configs = {
#     "episode_val_concept_pseudo_accuracy": {
#         "ylabel": "Concept Accuracy",
#         "ylim": (0.48, 0.59),
#         "shortname": "concept"
#     },
#     "episode_val_preference_accuracy": {
#         "ylabel": "Preference Accuracy",
#         "ylim": (0.47, 0.66),
#         "shortname": "preference"
#     }
# }

llm_emb = "llama2"
metric_configs = {
    "episode_val_concept_pseudo_accuracy": {
        "ylabel": "Concept Accuracy",
        "ylim": (0.46, 0.65),
        "shortname": "concept"
    },
    "episode_val_preference_accuracy": {
        "ylabel": "Preference Accuracy",
        "ylim": (0.56, 0.74),
        "shortname": "preference"
    }
}

# === INPUT CONFIGURATION ===
path = "/Users/slaguna/Downloads/cb-rm/src/plotting/" + llm_emb + "_"
group_files = {
    "uniform": "group_stats_final_uniform.csv",
    "eig": "group_stats_final_eig.csv",
    "CIS_concepts": "group_stats_final_CIS.csv",
    "concept_uncertainty": "group_stats_final_conc_unc.csv",
}
acq_to_group = {
    "uniform": "final_uniform",
    "eig": "final_eig",
    "CIS_concepts": "final_CIS",
    "concept_uncertainty": "final_conc_unc"
}
acq_labels = {
    "uniform": "Random",
    "eig": "EIG",
    "CIS_concepts": "CIS",
    "concept_uncertainty": "Concept Uncertainty"
}

prefix = "training.acquisition_function: "

# === PLOTTING CONFIG ===
all_acqs = ["uniform", "eig", "CIS_concepts", "concept_uncertainty"]
subset_acqs = ["uniform", "eig"]
# === LOOP THROUGH CONFIGS ===
band_modes = ["none", "std", "percentile"]  # New
for base_metric, config in metric_configs.items():
    ylabel = config["ylabel"]
    ylim = config["ylim"]
    shortname = config["shortname"]
    for acq_set, tag in zip([all_acqs, subset_acqs], ["all", "subset"]):
        for band_mode in band_modes:

            fig, ax = plt.subplots(figsize=(3.4, 2.8))

            # Checkerboard background
            x_vals = list(range(1, 31))
            for i, x in enumerate(x_vals):
                if i % 2 == 0:
                    ax.axvspan(x - 0.5, x + 0.5, color='lightgrey', alpha=0.15)

            for i, acq in enumerate(acq_set):
                filepath = path + group_files[acq]
                df = pd.read_csv(filepath)
                df = df[df["episode"] <= 30]
                df = df[df["episode"] >= 1]

                group_name = acq_to_group[acq]
                mean_col = f"{prefix}{group_name} - {base_metric}"
                std_col = f"{mean_col}__STD"
                p5_col = f"{mean_col}__P5"
                p95_col = f"{mean_col}__P95"

                if mean_col in df.columns:
                    ax.plot(df["episode"], df[mean_col],
                            label=acq_labels.get(acq, acq),
                            color=colors[i],
                            marker=markers[i],
                            linestyle=linestyles[i],
                            linewidth=1,
                            markersize=2.5)

                    if band_mode == "std" and std_col in df.columns:
                        ax.fill_between(df["episode"],
                                        df[mean_col] - df[std_col],
                                        df[mean_col] + df[std_col],
                                        color=colors[i], alpha=0.2)

                    elif band_mode == "percentile" and p5_col in df.columns and p95_col in df.columns:
                        ax.fill_between(df["episode"],
                                        df[p5_col],
                                        df[p95_col],
                                        color=colors[i], alpha=0.2)
                else:
                    print(f"⚠️ Column missing: {mean_col} in {filepath}")

            # === AXIS STYLING ===
            ax.set_xlabel("Episode Number", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_yticks(np.round(np.linspace(ylim[0], ylim[1], 5), 2))
            ax.set_xlim(1, 30)
            ax.set_xticks([1] + list(range(5, 31, 5)))
            ax.tick_params(
                axis='both',
                which='both',
                direction='out',
                length=2.5,
                width=1.0,
                color='black',
                bottom=True,
                left=True,
                top=False,
                right=False
            )
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_zorder(10)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)

            # === SAVE ===
            if band_mode == "none":
                suffix = "_mean"
            elif band_mode == "std":
                suffix = "_std"
            else:
                suffix = "_p95"

            filename = f"{shortname}_{tag}_accuracy_plot{suffix}.pdf"
            save_path = os.path.join("./src/plotting/plots/", llm_emb + "_" + filename)
            plt.tight_layout(pad=0.2, rect=[0, 0.1, 1, 1])
            plt.savefig(save_path, bbox_inches="tight")
            print(f"✅ Saved: {filename}")
            plt.close()
