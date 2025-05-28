# Interpretable Reward Modeling with Active Concept Bottlenecks

CB-RM is a framework for **interpretable reward modeling** using **concept bottlenecks** and **active learning**. It enables efficient and transparent preference learning by selectively acquiring concept annotations in low-supervision settings, especially useful in RLHF scenarios.

---

## Update data paths:

- embeddings_path: ./datasets/ultrafeedback/embeddings/meta-llama/Llama-2-7b-hf/
- splits_path: ./datasets/ultrafeedback/splits.csv
- concept_labels_path: ./datasets/ultrafeedback/concept_labels/gpt4o-eus2-202407
- preference_labels_path: ./datasets/ultrafeedback/- preference_labels/gpt4o-eus2-202407
---

## Run active learning:

`python src/utils/active_train.py`

-----

## Overwrite the configs at any time with hydra:

i.e.: `python src/utils/active_train.py training.num_epochs=20`

---
## Setup you wandb for visualizations:


  - project_name: cb-rm
  - wandb_entity: al_cb-rm
  - run_name_prefix: al_cb-rm




