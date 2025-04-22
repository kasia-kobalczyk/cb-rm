import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import numpy as np
import sys
import os
import random 
from hydra.utils import instantiate
from copy import deepcopy

from src.models.reward_models import *
from src.utils.training import ActiveTrainer


def setup_trainer(cfg, save_dir=None):
    # Create datasets 
    train_dataset = instantiate(cfg.train_dataset)
    val_dataloader = instantiate(cfg.val_dataloader)
    
    # Set output dimensions dynamically
    num_concepts = len(train_dataset.concept_names)
    cfg.model_builder.concept_encoder.output_dim = 2 * num_concepts if cfg.model_type == "probabilistic" else num_concepts
    cfg.model_builder.gating_network.output_dim = num_concepts
    # Workaround for the concept sampler: Cleaner if yaml split in two
    model_builder_cfg = deepcopy(cfg.model_builder)
    if cfg.model_type == "deterministic": OmegaConf.set_struct(model_builder_cfg, False); del model_builder_cfg["concept_sampler"]
    
    # Create model
    model = instantiate(model_builder_cfg)
    model.to(cfg.model.device)
    
    # Create Trainer
    trainer = ActiveTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataloader=val_dataloader,
        save_dir=save_dir,
    )
    
    return trainer

@hydra.main(version_base=None, config_path=f"../configs", config_name="train_config") #active_config, train_config
def train(cfg: DictConfig):

    # Set random seed
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    if not cfg.training.dry_run:
        # Create save folder and save cfg
        run_name_prefix = cfg.save.run_name_prefix if cfg.save.run_name_prefix else "run"
        save_dir = f"./saves/{cfg.save.project_name}"
        if os.path.exists(save_dir):
            save_no = [int(x.split("_")[-1]) for x in os.listdir(save_dir) if x.startswith(run_name_prefix)]
            save_no = max(save_no) + 1 if len(save_no) > 0 else 0
        else:
            save_no = 0
        save_dir = os.path.join(save_dir, f"{run_name_prefix}_{save_no}")
        os.makedirs(save_dir, exist_ok=True)

        trainer = setup_trainer(cfg, save_dir)
        cfg = trainer.cfg
        # Save config
        with open(f"{save_dir}/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
            
        # Initialize wandb
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=cfg.save.project_name, entity=cfg.save.wandb_entity, name=f"{run_name_prefix}_{save_no}"
        )

        best_eval_loss = trainer.train_loop()
        wandb.finish()
    
    else:
        trainer = setup_trainer(cfg)
        best_eval_loss = trainer.train_loop()
    
    return best_eval_loss



if __name__ == "__main__":
    train()


    