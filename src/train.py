import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import numpy as np
from tqdm import tqdm
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.dataloaders import get_dataloader, batch_to_device
from src.models.reward_models import *


class Trainer:
    def __init__(
            self, 
            cfg, 
            model,
            train_dataloader,
            val_dataloader,
            save_dir,
        ):
        self.cfg = cfg
        self.device = cfg.model.device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.last_save_it = 0
        self.num_epochs = cfg.training.num_epochs
        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.training.lr, eps=3e-5
        )
        self.save_dir = save_dir
        self.eval_steps = cfg.training.eval_steps
        self.max_num_eval_steps = cfg.training.max_num_eval_steps


    def run_batch(self, batch):
        batch = batch_to_device(batch, self.device)
        results = self.model(batch)

        results['loss'] = self.cfg.loss.beta * results['preference_loss'] + (1 - self.cfg.loss.beta) * results['concept_loss']

        return results
    

    def train(self):
        training_metrics = [
            'loss', 'preference_accuracy', 'concept_pseudo_accuracy', 'preference_loss', 'concept_loss'
        ]
        eval_metrics = [
            'preference_accuracy', 'concept_pseudo_accuracy'
        ]
        eval_stopping_metric = 'preference_accuracy'
        it = 0
        best_eval_metric = np.inf
        for epoch in range(self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")
            for batch in tqdm(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad()
                results = self.run_batch(batch)
                loss = results["loss"]
                loss.backward()
                self.optimizer.step()
                if not self.cfg.training.dry_run:
                    wandb.log(
                        {'train_' + k: results[k] for k in training_metrics},
                        step=self.last_save_it + it,
                    )

                if it % self.eval_steps == 0 and it > 0:
                    val_results = self.eval(eval_metrics, dataloader='val', max_steps=self.max_num_eval_steps)
                    if not self.cfg.training.dry_run:
                        wandb.log(
                            {'val_' + k: val_results[k] for k in eval_metrics},
                            step=self.last_save_it + it,
                        )
                        eval_metric_value = val_results[eval_stopping_metric]
                        if 'accuracy' in eval_stopping_metric:
                            eval_metric_value = -eval_metric_value
                        if eval_metric_value < best_eval_metric:
                            best_eval_metric =  eval_metric_value

                            state_dict_save = self.model.state_dict()

                            torch.save(state_dict_save, f"{self.save_dir}/model_best.pt")
                            torch.save(
                                self.optimizer.state_dict(),
                                f"{self.save_dir}/optim_best.pt",
                            )
                            print(f"Best model saved at step {self.last_save_it + it}")
                it += 1
        
        return best_eval_metric


    def eval(self, metrics, dataloader, max_steps):
        print(f"Evaluating on {dataloader} data")
        if dataloader == 'val':
            dataloader = self.val_dataloader
        
        eval_results = {}
        for k in metrics:
            eval_results[k] = []
        it = 0
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                results = self.run_batch(batch)
                for k in metrics:
                    eval_results[k].append(results[k])
                it += 1
                if it > max_steps:
                    break
                        
            for k in metrics:
                eval_results[k] = torch.mean(torch.tensor(eval_results[k])).item()

        return eval_results



def setup_trainer(cfg, save_dir=None):
    train_dataloader = get_dataloader(cfg.data, split='train')
    val_dataloader = get_dataloader(cfg.data, split='val')

    num_concepts = len(train_dataloader.dataset.concept_names)

    if cfg.model.encoder == "simple":
        concept_encoder = SimpleConceptEncoder(
            input_dim=cfg.model.input_dim,
            output_dim=num_concepts,
        )
    else:
        raise NotImplementedError(
            f"Encoder {cfg.model.encoder} not implemented"
        )

    gating_network = GatingNetwork(
        input_dim=cfg.model.input_dim,
        output_dim=num_concepts,
    )

    model = BottleneckRewardModel(
        concept_encoder=concept_encoder,
        gating_network=gating_network,
    )
    model.to(cfg.model.device)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_dir=save_dir,
    )
    
    return trainer


@hydra.main(version_base=None, config_path=f"../configs", config_name="config")
def train(cfg):
    
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    
    if not cfg.training.dry_run:
        # Create save folder and save cfg
        run_name_prefix = cfg.save.run_name_prefix if cfg.save.run_name_prefix else "run"
        save_dir = f"./saves/{cfg.save.project_name}"
        if os.path.exists(save_dir):
            save_no = len(os.listdir(save_dir))
            save_no = [
                int(x.split("_")[-1])
                for x in os.listdir(save_dir)
                if x.startswith(run_name_prefix)
            ]
            if len(save_no) > 0:
                save_no = max(save_no) + 1
            else:
                save_no = 0
            save_dir = os.path.join(save_dir, f"{run_name_prefix}_{save_no}")
        else:
            save_no = 0
            save_dir = os.path.join(save_dir, f"{run_name_prefix}_{save_no}")
        
        trainer = setup_trainer(cfg, save_dir=save_dir)

        os.makedirs(save_dir, exist_ok=True)

        # Save cfg
        cfg = trainer.cfg
        with open(f"{save_dir}/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
            
        # Initialize wandb
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=cfg.save.project_name, name=f"{run_name_prefix}_{save_no}"
        )
    
        best_eval_loss = trainer.train()
        wandb.finish()
    
    else:
        trainer = setup_trainer(cfg)
        best_eval_loss = trainer.train()
    
    return best_eval_loss



if __name__ == "__main__":
    train()
    

    