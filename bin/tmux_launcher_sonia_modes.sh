#!/bin/bash

ACQUISITION_CONFIGS=(
    "uniform"
    "concept_variance"
    "concept_entropy"
    # "sampling_eig"
    "eig"
    "eig_concepts"
    "CIS"
    "CIS_concepts"
    "etur"
    "etur_concepts"
    "concept_uncertainty"
    "concept_weight"
    "certainty_concept_weight"
    "prob_concept_weight"
    "label_uncertainty"
    "label_entropy"
    "variance_label_uncertainty"
)

CONFIG_NAMES=("train_config" "train_config_llm" "active_config" "active_config_llm")
TRAINING_MODES=("joint" "pretrain_joint" "sequential")
BETA_VALUES=("1" "10" "100")

PROJECT_DIR="$SLAGUNA/cb-rm"
CONDA_ENV="cb_rm"

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    for TRAIN_MODE in "${TRAINING_MODES[@]}"; do
        for BETA in "${BETA_VALUES[@]}"; do

            # Determine if acquisition functions should be used
            if [[ "$CONFIG_NAME" == active_config* ]]; then
                # Use acquisition functions
                for ACQ in "${ACQUISITION_CONFIGS[@]}"; do

                    SESSION_NAME="cb_rm_${CONFIG_NAME}_${ACQ}_${TRAIN_MODE}_beta${BETA}"
                    
                    tmux new-session -d -s $SESSION_NAME
                    tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
                    tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m

                    CMD="python -m src.active_train --config-name=$CONFIG_NAME ++training.acquisition_function='$ACQ' ++training.training_mode='$TRAIN_MODE' ++loss.beta_concept=$BETA ++save.project_name='active_pref_learning_acq'"

                    tmux send-keys -t $SESSION_NAME "$CMD" C-m
                    echo "Launched tmux session: $SESSION_NAME"

                done

            else
                # No acquisition functions, just run normally
                SESSION_NAME="cb_rm_${CONFIG_NAME}_${TRAIN_MODE}_beta${BETA}"
                
                tmux new-session -d -s $SESSION_NAME
                tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
                tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m

                CMD="python -m src.active_train --config-name=$CONFIG_NAME ++training.training_mode='$TRAIN_MODE' ++loss.beta_concept=$BETA ++save.project_name='active_pref_learning_acq'"

                tmux send-keys -t $SESSION_NAME "$CMD" C-m
                echo "Launched tmux session: $SESSION_NAME"

            fi
        done
    done
done
