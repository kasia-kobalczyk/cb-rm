import wandb
import pandas as pd

ENTITY = "interp_rewards_RLHF"
METRICS = ["episode_val_concept_pseudo_accuracy", "episode_val_preference_accuracy"]
# PROJECT = "SimpleLLM_final"
# llm="llama3"
# GROUP_RUNS = {
#     "final_uniform": [
#         "active_learning_crm_24", "active_learning_crm_23",
#         "active_learning_crm_21", "active_learning_crm_20"
#     ],
#     "final_eig": [
#         "active_learning_crm_14", "active_learning_crm_13",
#         "active_learning_crm_11", "active_learning_crm_10"
#     ],
#     "final_CIS": [
#         "active_learning_crm_39", "active_learning_crm_38",
#         "active_learning_crm_36", "active_learning_crm_35"
#     ],
#     "final_conc_unc": [
#         "active_learning_crm_4", "active_learning_crm_3",
#         "active_learning_crm_1", "active_learning_crm_0"
#     ]
# }
PROJECT = "SimpleLLM"
llm="llama2"
GROUP_RUNS = {
    "final_uniform": [
        "active_learning_crm_289", "active_learning_crm_288",
        "active_learning_crm_287", "active_learning_crm_286"
    ],
    "final_eig": [
        "active_learning_crm_295", "active_learning_crm_294",
        "active_learning_crm_292", "active_learning_crm_291"
    ],
    "final_CIS": [
        "active_learning_crm_300", "active_learning_crm_299",
        "active_learning_crm_297", "active_learning_crm_296"
    ],
    "final_conc_unc": [
        "active_learning_crm_305", "active_learning_crm_304",
        "active_learning_crm_302", "active_learning_crm_301"
    ]
}
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")
run_dict = {r.name: r for r in runs if r.state == "finished"}

for group_name, run_names in GROUP_RUNS.items():
    dfs = []
    for name in run_names:
        run = run_dict.get(name)
        if run is None:
            print(f"⚠️ Skipping missing or unfinished run: {name}")
            continue
        try:
            history_rows = []
            for row in run.scan_history():
                history_rows.append(row)

            df = pd.DataFrame(history_rows)

            # Only keep rows where 'episode' is not NaN
            if "episode" in df.columns and all(m in df.columns for m in METRICS):
                df = df[["episode"] + METRICS].dropna(subset=["episode"])
                df["run_id"] = run.id
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error loading history for {name}: {e}")

    if not dfs:
        print(f"No data found for {group_name}, skipping.")
        continue

    merged = pd.concat(dfs)
    merged = merged.drop(columns=["run_id"]) 
    if merged.empty:
        print(f"⚠️ No merged data for {group_name}")
        continue

    stats = {"episode": merged.groupby("episode").mean().index}
    for metric in METRICS:
        grouped = merged.groupby("episode")[metric]
        stats[f"training.acquisition_function: {group_name} - {metric}"] = grouped.mean().values
        stats[f"training.acquisition_function: {group_name} - {metric}__STD"] = grouped.std().values
        stats[f"training.acquisition_function: {group_name} - {metric}__MIN"] = grouped.min().values
        stats[f"training.acquisition_function: {group_name} - {metric}__MAX"] = grouped.max().values
        stats[f"training.acquisition_function: {group_name} - {metric}__COUNT"] = grouped.count().values
        stats[f"training.acquisition_function: {group_name} - {metric}__P5"] = grouped.quantile(0.05).values
        stats[f"training.acquisition_function: {group_name} - {metric}__P95"] = grouped.quantile(0.95).values

    stats_df = pd.DataFrame(stats)
    out_path = f"/Users/slaguna/Downloads/cb-rm/src/plotting/" + llm + f"_group_stats_{group_name}.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")