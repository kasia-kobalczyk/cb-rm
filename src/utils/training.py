import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import random 
from src.data.dataloaders import *
from src.models.reward_models import *
from torch.utils.data import Sampler
from copy import deepcopy
import torch.nn.functional as F

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

def bernoulli_stats(p, q=None):
    eps = 1e-8
    if not torch.is_tensor(p):
        p = torch.tensor(p, dtype=torch.float32)
    p = torch.clamp(p, eps, 1 - eps)

    if q is None:
        return - (p * p.log() + (1 - p) * (1 - p).log())
    else:
        if not torch.is_tensor(q):
            q = torch.tensor(q, dtype=torch.float32)
        q = torch.clamp(q, eps, 1 - eps)
        return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def intervene_logits(relative_concept_logits, concept_idx, weights, intervene_value):
    intervened_logits = relative_concept_logits.clone()
    intervened_logits[:, concept_idx] = intervene_value
    reward = torch.sum(weights * intervened_logits, dim=1)
    return reward, intervened_logits

    
class FIFOSampler(Sampler):
    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity
        self.current_indices = []

    def add_new(self, added_idx):
        added_rows = list({i for (i, _) in added_idx})
        self.current_indices = list(added_rows) + self.current_indices
        self.current_indices = self.current_indices[:self.buffer_capacity]
    
    def __iter__(self):
        shuffled = random.sample(self.current_indices, len(self.current_indices))
        return iter(shuffled)

    def __len__(self):
        return len(self.current_indices)


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
        self.current_episode = 0

        self.training_metrics = [
            'loss', 'preference_accuracy', 'concept_pseudo_accuracy', 'preference_loss', 'concept_loss' #, 'concept_encoder_grad_norm_t', 'concept_encoder_grad_norm_tc', 'concept_encoder_grad_norm_kltc'
        ]

        if self.cfg.model.model_type == "probabilistic":
            self.training_metrics.append('kl_loss')

        self.eval_metrics = [
            'preference_accuracy', 'concept_pseudo_accuracy'
        ]
        self.sampler = FIFOSampler(cfg.training.buffer_capacity)
    
    def train_loop(self):

        if self.sampler is not None: self.sampler.add_new(self.train_dataset.initial_samples)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            collate_fn=collate_fn,
        )

        if not getattr(self.cfg.training, "active_learning", True):
            print("Running standard training (no active learning episodes)...")
            return self.train_episode()
        
        print(f"Starting training with {len(self.sampler.current_indices)} pairs")
        for episode in range(self.cfg.training.num_episodes):
            print(f"Episode {episode+1}/{self.cfg.training.num_episodes}")
            # Run one episode of training
            best_eval_metric = self.train_episode()
            self.current_episode += 1
            print(f"Best eval metric afer episode {episode}: {best_eval_metric:.3f}")
            # Query new data
            added_idx = self.query_new_data()
            self.train_dataset.build_dataset(added_idx)
            self.sampler.add_new(added_idx)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.data.batch_size,
                sampler=self.sampler,
                collate_fn=collate_fn,
            )
            print(f"Added {len(added_idx)} concept labels to the training set")
            self.model.load_state_dict(torch.load(f"{self.save_dir}/model_best.pt"))
            self.optimizer.load_state_dict(torch.load(f"{self.save_dir}/optim_best.pt"))


    def train_episode(self):

        if self.cfg.training.training_mode == "joint":
            phases = [("joint", self.cfg.training.j_epochs)]

        else:
            raise ValueError("Invalid training mode")
        eval_stopping_metric = 'preference_accuracy'
        it = 0
        best_eval_metric = self.best_eval_metric
        for mode, epochs in phases:
            if mode == "preference_only":
                print("Freezing concept encoder...")
                for param in self.model.concept_encoder.parameters():
                    param.requires_grad = False
            elif mode == "concept_only":
                print("Unfreezing concept encoder...")
                for param in self.model.concept_encoder.parameters():
                    param.requires_grad = True

            print(f"Starting phase: {mode} for {epochs} epochs")
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs} [{mode}]")
                for batch in tqdm(self.train_dataloader):
                    self.model.train()
                    self.optimizer.zero_grad()
                    results = self.run_batch(batch, it)

                    # Select loss
                    if mode == "concept_only":
                        loss = results["concept_loss"]
                    elif mode == "preference_only":
                        loss = results["preference_loss"]
                    elif mode == "joint":
                        loss = results["loss"]
                    else:
                        raise ValueError("Unknown mode")
                
                    loss.backward()
                    self.optimizer.step()

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

            # Compute eval results after the episode is finished
            final_val_results = self.eval(self.eval_metrics, dataloader='val', max_steps=self.max_num_eval_steps,intervention_percentages=[0.2, 0.5, 0.9])
            if not self.cfg.training.dry_run:
                wandb.log(
                    {'val_' + k: final_val_results[k] for k in self.eval_metrics},
                )
                log_dict = {'episode_val_' + k: final_val_results[k] for k in self.eval_metrics}
                log_dict['episode'] = self.current_episode
                wandb.log(log_dict)
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

            # Save the uncertainty map
            if self.cfg.model.model_type == "probabilistic" and self.cfg.training.get("acquisition_function", None) not in ["uniform", None]:
                self.compute_uncertainty_map()

        return best_eval_metric

    def compute_uncertainty_map(self):
        self.uncertainty_map = {
            "concept_variance": [],
            "eig": [],
            "CIS_concepts": [],
        }
        self.model.eval()
        with torch.no_grad():
            pool_index = self.train_dataset.pool_index
            pool_instance_index = list({instance_idx for (instance_idx, _) in pool_index})
            pool_dataset = torch.utils.data.Subset(self.train_dataset, pool_instance_index)
            pool_dataloader = DataLoader(
                pool_dataset,
                batch_size=2048, #self.cfg.data.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            print(f"Computing uncertainty for the pool dataset of size {len(pool_dataset)} ...")
            for batch in tqdm(pool_dataloader):
                results = self.run_batch(batch)
                relative_var = results["relative_var"].cpu()
                relative_mean = results["relative_mean"].cpu()
                relative_concept_logits = results["relative_concept_logits"].cpu()
                reward_diff = results["reward_diff"].cpu()
                weights = results["weights"].cpu()
                idx_batch = batch["pair_idx"]
                concept_uncertainty = torch.sigmoid(-relative_concept_logits)
                label_uncertainty = torch.sigmoid(-reward_diff)

                # Now fill the uncertainty_map dictionary
                if self.cfg.training.acquisition_function in ["concept_variance"]:
                    self.uncertainty_map["concept_variance"].extend(
                        ((idx.item(), k), relative_var[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(relative_var.shape[1])
                    )

                if self.cfg.training.acquisition_function in ["eig", "CIS_concepts"]:
                    # Before intervention
                    p = torch.sigmoid(-reward_diff)
                    for k in range(batch['concept_labels'].shape[-1]):
                        # Intervene 0
                        reward_0, _ = intervene_logits(relative_concept_logits, k, weights, 10.0)
                        q0 = torch.sigmoid(-reward_0)
                        # Intervene 1
                        reward_1, _ = intervene_logits(relative_concept_logits, k, weights, -10.0)
                        q1 = torch.sigmoid(-reward_1)
                        
                        concept_logit = relative_concept_logits[0, k]
                        concept_prob = torch.sigmoid(-concept_logit)
                        
                        if self.cfg.training.acquisition_function.startswith("eig"):
                            kl_0 = bernoulli_stats(p, q0)
                            kl_1 = bernoulli_stats(p, q1)
                            scores = (concept_prob * kl_1 + (1 - concept_prob) * kl_0)


                        elif self.cfg.training.acquisition_function.startswith("CIS_concepts"):
                            expected_p = (1 - concept_prob) * q0 + concept_prob * q1
                            scores = torch.abs((expected_p - p))
                            lambda_weight = getattr(self.cfg.training, "CIS_uncertainty_lambda", 0.1)
                            scores += lambda_weight * relative_var.squeeze()[:,k]
                        
                        self.uncertainty_map[self.cfg.training.acquisition_function].extend(
                            ((idx.item(), k), scores[b].item())
                            for b, idx in enumerate(idx_batch)
                        )

        self.model.train()

    def run_batch(self, batch, it=0):
        batch = batch_to_device(batch, self.device)
        results = self.model(batch)
        results['loss'] = results['preference_loss'] + self.cfg.loss.beta_concept * results['concept_loss'] 
        if 'kl_loss' in results:
            beta_kl = self.cfg.loss.beta_kl
            results['loss'] += beta_kl * results['kl_loss']
            results['beta_kl'] = beta_kl  # optional: for wandb logging
            
        return results

    def query_new_data(self):
        if self.cfg.model.model_type == "deterministic" and self.cfg.training.acquisition_function != "uniform":
                print("Model is deterministic. Falling back to uniform acquisition.")
                self.cfg.training.acquisition_function = "uniform"

        if self.cfg.training.acquisition_function == "uniform":
            added_idx = random.sample(
            list(self.train_dataset.pool_index),
            self.cfg.training.num_acquired_samples
            )
        elif self.cfg.training.acquisition_function in [
            "concept_variance", "eig", "CIS_concepts"
        ]:
            metric = self.cfg.training.acquisition_function
            sorted_pairs = sorted(self.uncertainty_map[metric], key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        else:
            raise NotImplementedError(
                f"Acquisition function {self.cfg.training.acquisition_function} not implemented"
            )
        return added_idx
                

    def eval(self, metrics, dataloader, max_steps, intervention_percentages=None):
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
                if 'relative_concept_logits' in results:
                    if 'avg_concepts' not in eval_results:
                        eval_results['avg_concepts'] = torch.sigmoid(-results['relative_concept_logits']).detach().cpu()
                        eval_results['lbl_concepts'] = batch["concept_labels"].detach().cpu()
                for k in metrics:
                    eval_results[k].append(results[k])
                it += 1
                if it > max_steps:
                    break
                        
            if 'avg_concepts' in eval_results:
                avg_concepts = eval_results['avg_concepts'][0,:] 
                concepts = eval_results['lbl_concepts'][0,:]
                weights = results['weights'].detach().cpu() if results['weights'].dim() <2 else results['weights'][0].detach().cpu()
                concept_names = getattr(self.train_dataset, "concept_names", [f"concept_{i}" for i in range(avg_concepts.shape[-1])])
                wandb.log({f"concept_values/{name}": val for name, val in zip(concept_names, avg_concepts.numpy())})
                wandb.log({f"concept_labes/{name}": val for name, val in zip(concept_names, concepts.numpy())})
                wandb.log({f"scalar_weights/{name}": val for name, val in zip(concept_names, weights.numpy())})
                del eval_results['avg_concepts']  # clean up
            for k in metrics:
                eval_results[k] = torch.mean(torch.tensor(eval_results[k])).item()

        return eval_results