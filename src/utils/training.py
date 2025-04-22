import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import random 
from src.data.dataloaders import *
from src.models.reward_models import *
from torch.utils.data import Sampler


class ReplayPrioritySampler(Sampler):
    def __init__(self, dataset, added_idx, keep_ratio=0.2):
        self.dataset = dataset
        self.keep_ratio = keep_ratio

        # Collect all valid indices
        self.all_indices = list(range(len(self.dataset)))
        self.added_idx = added_idx

    def __iter__(self):
        # Convert (idx, concept_idx) -> example_idx (row in dataset)
        added_rows = {i for i, _ in self.added_idx}
        # Remaining indices
        remaining = list(set(self.all_indices) - added_rows)

        # Replay sampling
        n_keep = int(len(remaining) * self.keep_ratio)
        kept_replay = random.sample(remaining, n_keep) if n_keep > 0 else []

        # Combine
        final_indices = list(added_rows) + kept_replay
        random.shuffle(final_indices)
        return iter(final_indices)

    def __len__(self):
        return int(len(self.dataset) * self.keep_ratio) + len(self.added_idx)

# Training utils
class ActiveTrainer:
    def __init__(
            self, 
            cfg, 
            model,
            train_dataset,
            val_dataloader,
            save_dir,
        ):
        self.cfg = cfg
        self.device = cfg.model.device
        self.train_dataset = train_dataset
        self.val_dataloader = val_dataloader
        self.uncertainty_map = []  
        self.num_epochs = cfg.training.num_epochs
        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.training.lr, eps=3e-5
        )
        self.save_dir = save_dir
        self.eval_steps = cfg.training.eval_steps
        self.max_num_eval_steps = cfg.training.max_num_eval_steps
        self.best_eval_metric = np.inf

        self.training_metrics = [
            'loss', 'preference_accuracy', 'concept_pseudo_accuracy', 'preference_loss', 'concept_loss'
        ]

        if self.cfg.model.model_type == "probabilistic":
            self.training_metrics.append('kl_loss')

        self.eval_metrics = [
            'preference_accuracy', 'concept_pseudo_accuracy'
        ]

    def train_loop(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        print(f"Starting training with {len(self.train_dataset)} pairs")
        if not getattr(self.cfg.training, "active_learning", True):
            print("Running standard training (no active learning episodes)...")
            return self.train_episode()

        for episode in range(self.cfg.training.num_episodes):
            print(f"Episode {episode}/{self.cfg.training.num_episodes}")
            # Run one episode of training
            best_eval_metric = self.train_episode()
            print(f"Best eval metric afer episode {episode}: {best_eval_metric:.3f}")
            # Query new data
            added_idx = self.query_new_data()
            self.train_dataset.build_dataset(added_idx)
            sampler = ReplayPrioritySampler(self.train_dataset, added_idx, self.cfg.training.keep_ratio)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.data.batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
            print(f"Added {len(added_idx)} concept labels to the training set")
            # Load the best model and optimizer: # TO DO: may instead train from scratch (?)
            self.model.load_state_dict(torch.load(f"{self.save_dir}/model_best.pt"))
            self.optimizer.load_state_dict(torch.load(f"{self.save_dir}/optim_best.pt"))


    def train_episode(self):
        eval_stopping_metric = 'preference_accuracy'
        it = 0
        self.uncertainty_map = []  
        best_eval_metric = self.best_eval_metric
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in tqdm(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad()
                results = self.run_batch(batch)
                loss = results["loss"]
                loss.backward()
                self.optimizer.step()
                if self.cfg.model.model_type == "probabilistic":
                    relative_var = results["relative_var"].detach().cpu()
                    idx_batch = batch["id"]
                    self.uncertainty_map.extend(((idx.item(), k), relative_var[b, k].item())for b, idx in enumerate(idx_batch)for k in range(relative_var.shape[1]))
                if not self.cfg.training.dry_run:
                    wandb.log(
                        {'train_' + k: results[k] for k in self.training_metrics},
                    )
                if it % self.eval_steps == 0 and it > 0:
                    val_results = self.eval(self.eval_metrics, dataloader='val', max_steps=self.max_num_eval_steps)
                    if not self.cfg.training.dry_run:
                        wandb.log(
                            {'val_' + k: val_results[k] for k in self.eval_metrics},
                        )
                        eval_metric_value = val_results[eval_stopping_metric]
                        if 'accuracy' in eval_stopping_metric:
                            eval_metric_value = -eval_metric_value
                        if eval_metric_value < best_eval_metric:
                            best_eval_metric =  eval_metric_value

                            state_dict_save = self.model.state_dict()
                            # Save best model and optimizer
                            torch.save(state_dict_save, f"{self.save_dir}/model_best.pt")
                            torch.save(
                                self.optimizer.state_dict(),
                                f"{self.save_dir}/optim_best.pt",
                            )
                            print(f"Best model saved at step {it}")
                it += 1
        
        final_val_results = self.eval(self.eval_metrics, dataloader='val', max_steps=self.max_num_eval_steps)
        if not self.cfg.training.dry_run:
            wandb.log(
                {'val_' + k: final_val_results[k] for k in self.eval_metrics},
            )
            eval_metric_value = final_val_results[eval_stopping_metric]
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
                print(f"Best model saved at step {it}")

        self.best_eval_metric = best_eval_metric

        return best_eval_metric

    def run_batch(self, batch):
        batch = batch_to_device(batch, self.device)
        results = self.model(batch)
        results['loss'] = results['preference_loss'] + self.cfg.loss.beta_concept * results['concept_loss'] 
        
        if 'kl_loss' in results:
            results['loss'] += self.cfg.loss.beta_kl * results['kl_loss']
            
        return results

    def query_new_data(self):
        if self.cfg.training.acquisition_function == "expected_information_gain":
            if self.cfg.model.model_type == "deterministic":
                print("Model is deterministic. Falling back to uniform acquisition.")
                self.cfg.training.acquisition_function = "uniform"
            else:
                sorted_pairs = sorted(self.uncertainty_map, key=lambda x: -x[1])
                added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "uniform":
            added_idx = random.sample(
                list(self.train_dataset.pool_index),
                self.cfg.training.num_acquired_samples
            )
        else:
            raise NotImplementedError(
                f"Acquisition function {self.cfg.training.acquisition_func} not implemented"
            )
        return added_idx

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