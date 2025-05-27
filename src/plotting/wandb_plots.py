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

# === METRIC CONFIGURATIONS ===
llm_emb = "llama2" 
metric_configurations = [
    {
        "csv_path": "/Users/slaguna/Downloads/cb-rm/src/plotting/wandb_export_2025-05-27T11_26_16.193+09_00.csv",
        "base_metric": "episode_val_concept_pseudo_accuracy",
        "ylabel": "Concept Accuracy",
        "ylim": (0.46, 0.65),
        "full_label": "concept",
    },
    {
        "csv_path": "/Users/slaguna/Downloads/cb-rm/src/plotting/wandb_export_2025-05-27T11_26_24.443+09_00.csv",
        "base_metric": "episode_val_preference_accuracy",
        "ylabel": "Preference Accuracy",
        "ylim": (0.56, 0.74),
        "full_label": "preference",
    },
]
# llm_emb = "llama3" 
# metric_configurations = [
#     {
#         "csv_path": "/Users/slaguna/Downloads/cb-rm/src/plotting/wandb_export_2025-05-27T15_20_39.690+09_00.csv",
#         "base_metric": "episode_val_concept_pseudo_accuracy",
#         "ylabel": "Concept Accuracy",
#         "ylim": (0.48, 0.59),
#         "full_label": "concept",
#     },
#     {
#         "csv_path": "/Users/slaguna/Downloads/cb-rm/src/plotting/wandb_export_2025-05-27T15_20_49.365+09_00.csv",
#         "base_metric": "episode_val_preference_accuracy",
#         "ylabel": "Preference Accuracy",
#         "ylim": (0.47, 0.66),
#         "full_label": "preference",
#     },
# ]

# === ACQUISITION FUNCTION SETS ===
all_acqs = ["uniform", "eig", "CIS_concepts", "concept_uncertainty"]
subset_acqs = ["uniform", "eig"]
prefix = "training.acquisition_function: "
acq_labels = {
    "uniform": "Random",
    "eig": "EIG",
    "CIS_concepts": "CIS",
    "concept_uncertainty": "Concept Uncertainty"
}

# === LOOP THROUGH CONFIGS ===
for config in metric_configurations:
    df = pd.read_csv(config["csv_path"])
    df = df[df["episode"] <= 30]
    df = df[df["episode"] >= 1]
    base_metric = config["base_metric"]
    ylabel = config["ylabel"]
    ylim = config["ylim"]
    label = config["full_label"]

    for acq_set, tag in zip([all_acqs, subset_acqs], ["all", "subset"]):
        for with_bands in [True, False]:

            fig, ax = plt.subplots(figsize=(3.4, 2.8))

            # Checkerboard background
            x_vals = df["episode"].dropna().unique()
            for i, x in enumerate(x_vals):
                if i % 2 == 0:
                    ax.axvspan(x - 0.5, x + 0.5, color='lightgrey', alpha=0.15)

            for i, func in enumerate(acq_set):
                mean_col = f"{prefix}{func} - {base_metric}"
                min_col = f"{mean_col}__MIN"
                max_col = f"{mean_col}__MAX"

                if mean_col in df.columns:
                    ax.plot(df["episode"], df[mean_col],
                        label=acq_labels.get(func, func),
                        color=colors[i],
                        marker=markers[i],
                        linestyle=linestyles[i],
                        linewidth=1,
                        markersize=2.5)

                    if with_bands and all(col in df.columns for col in [min_col, max_col]):
                        ax.fill_between(df["episode"], df[min_col], df[max_col],
                                        color=colors[i], alpha=0.2)

            # Axis, grid, legend
            ax.set_xlabel("Episode Number", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_yticks(np.round(np.linspace(ylim[0], ylim[1], 5), 2))
            ax.set_xlim(1, df["episode"].max())
            ax.set_xticks([1] + list(range(5, df["episode"].max() + 1, 5)))
            # Force tick marks and thicker spine
            ax.tick_params(
                axis='both',
                which='both',
                direction='out',   # 'in' for LaTeX style, 'inout' for clarity
                length=2.5,
                width=1.0,
                color='black',
                bottom=True,
                left=True,
                top=False,
                right=False
            )

            # Force visible spines and bring them forward
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_zorder(10)  # ensure it’s above fill_between and grid

            # ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)

            # ax.legend(
            #     fontsize=7,
            #     loc='upper center',
            #     bbox_to_anchor=(0.5, -0.25),
            #     frameon=False,
            #     ncol=len(acq_set),
            #     title_fontsize=8
            # )

            plt.tight_layout(pad=0.2, rect=[0, 0.1, 1, 1])

            # === SAVE ===
            suffix = "" if with_bands else "_nobands"
            filename = f"{label}_{tag}_accuracy_plot{suffix}.pdf"
            save_path = os.path.join("./src/plotting/plots/", llm_emb + "_" + filename)
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved: {filename}")

            plt.close()

# === LEGEND FIGURE ===

# Define handles for each acquisition function
legend_handles = [
    mlines.Line2D([], [], color=colors[i], marker=markers[i],
                  linestyle=linestyles[i], linewidth=1, markersize=2.5,
                  label=acq_labels.get(func, func))
    for i, func in enumerate(all_acqs)
]

# Create a blank figure with just the legend
fig_legend = plt.figure(figsize=(5.5, 0.5))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')  # no axes

legend = ax_legend.legend(
    handles=legend_handles,
    loc='center',
    ncol=len(legend_handles),
    fontsize=8,
    frameon=False
)

plt.tight_layout()
save_path = os.path.join("./src/plotting/plots/", llm_emb + "_" + "legend_only.pdf")
plt.savefig(save_path, bbox_inches="tight")
print("Saved: legend_only.pdf")
plt.close()
