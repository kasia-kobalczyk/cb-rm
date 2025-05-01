#!/bin/bash

# Acquisition functions and config usage
ACQUISITION_CONFIGS=(
    "uniform:yes"
    "uniform:no"
    "expected_information_gain:no"
    "expected_information_gain_concepts:no"
    "expected_target_uncertainty_reduction:no"
    "expected_target_uncertainty_reduction_concepts:no"
    "CIS:no"
    "CIS_concepts:no"
    "concept_variance:no"
    "concept_uncertainty:no"
    "concept_weight:no"
    "certainty_concept_weight:no"
    "prob_concept_weight:no"
    "label_uncertainty:no"
    "label_entropy:no"
    "variance_label_uncertainty:no"
)

PROJECT_DIR="$SLAGUNA/cb-rm"
CONDA_ENV="cb_rm"
for entry in "${ACQUISITION_CONFIGS[@]}"
do
    IFS=':' read -r ACQ USE_CONFIG <<< "$entry"

    if [ "$USE_CONFIG" == "yes" ]; then
        SESSION_NAME="cb_rm_${ACQ}_config"
    else
        SESSION_NAME="cb_rm_${ACQ}_default"
    fi

    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
    tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m

    if [ "$USE_CONFIG" == "yes" ]; then
        CMD="python src/active_train.py -config_name=train_config ++training.acquisition_function='$ACQ'"
    else
        CMD="python src/active_train.py ++training.acquisition_function='$ACQ'"
    fi

    tmux send-keys -t $SESSION_NAME "$CMD" C-m
    echo "Launched tmux session: $SESSION_NAME"
done
