#!/bin/bash

# Acquisition functions and config usage
ACQUISITION_CONFIGS=(
    # "uniform:no"
    # "eig:no"
    # "sampling_eig:no"
    # #"eig_concepts:no"
    # #"expected_target_uncertainty_reduction:no"
    # #"expected_target_uncertainty_reduction_concepts:no"
    # "CIS:no"
    # #"CIS_concepts:no"
    # "concept_variance:no"
    #"concept_uncertainty:no"
    # "concept_weight:no"
    "certainty_concept_weight:no"
    # "prob_concept_weight:no"
    "label_uncertainty:no"
    #"label_entropy:no"
    "variance_label_uncertainty:no"
)

PROJECT_DIR="/mnt/pdata/knk25/active_pref_learning"
CONDA_ENV="llms"
for entry in "${ACQUISITION_CONFIGS[@]}"
do
    IFS=':' read -r ACQ USE_CONFIG <<< "$entry"

    if [ "$USE_CONFIG" == "yes" ]; then
        SESSION_NAME="${ACQ}_config"
    else
        SESSION_NAME="${ACQ}"
    fi

    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m

    if [ "$USE_CONFIG" == "yes" ]; then
        CMD="python -m src.active_train --config-name=train_config ++training.acquisition_function='$ACQ'"
    else
        CMD="python -m src.active_train --config-name=active_config ++training.acquisition_function='$ACQ' ++save.run_name_prefix='$ACQ'"
    fi

    tmux send-keys -t $SESSION_NAME "$CMD" C-m
    echo "Launched tmux session: $SESSION_NAME"
done
