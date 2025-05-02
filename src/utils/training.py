import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import random 
from src.data.dataloaders import *
from src.models.reward_models import *
from torch.utils.data import Sampler

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

class RatioSampler(Sampler):
    def __init__(self, dataset, buffer_ratio=0.2):
        self.dataset = dataset
        self.buffer_ratio = buffer_ratio

        # Collect all valid indices
        self.all_indices = list(range(len(self.dataset)))

    def add_new(self, added_idx):
         # Convert (idx, concept_idx) -> example_idx (row in dataset)
        added_rows = {i for i, _ in added_idx}
        
        # Remaining indices
        remaining = list(set(self.all_indices) - added_rows)

        # Replay sampling
        n_keep = int(len(remaining) * self.buffer_ratio)
        kept_replay = random.sample(remaining, n_keep) if n_keep > 0 else []

        # Combine
        self.current_indices = list(added_rows) + kept_replay
        random.shuffle(self.current_indices)

    def __iter__(self):
        return iter(self.current_indices)

    def __len__(self):
        return int(len(self.dataset) * self.buffer_ratio) + len(self.added_idx)

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
            'loss', 'preference_accuracy', 'concept_pseudo_accuracy', 'preference_loss', 'concept_loss'
        ]

        if self.cfg.model.model_type == "probabilistic":
            self.training_metrics.append('kl_loss')

        self.eval_metrics = [
            'preference_accuracy', 'concept_pseudo_accuracy'
        ]
        if self.cfg.training.buffer_type == 'ratio':
            self.sampler = RatioSampler(self.train_dataset, buffer_ratio=cfg.training.buffer_ratio)
        elif self.cfg.training.buffer_type == 'fifo':
            self.sampler = FIFOSampler(cfg.training.buffer_capacity)

    def train_loop(self):
        self.sampler.add_new(self.train_dataset.initial_samples)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
        )
        print(f"Starting training with {len(self.sampler.current_indices)} pairs")
        if not getattr(self.cfg.training, "active_learning", True):
            print("Running standard training (no active learning episodes)...")
            return self.train_episode()

        for episode in range(self.cfg.training.num_episodes):
            print(f"Episode {episode}/{self.cfg.training.num_episodes}")
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
            # Load the best model and optimizer: # TO DO: may instead train from scratch (?)
            self.model.load_state_dict(torch.load(f"{self.save_dir}/model_best.pt"))
            self.optimizer.load_state_dict(torch.load(f"{self.save_dir}/optim_best.pt"))


    def train_episode(self):
        eval_stopping_metric = 'preference_accuracy'
        it = 0
        
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
        final_val_results = self.eval(self.eval_metrics, dataloader='val', max_steps=self.max_num_eval_steps)
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
        if self.cfg.model.model_type == "probabilistic" and self.cfg.training.acquisition_function not in ["uniform", "intervention", "intervention_concepts"]:
            self.compute_uncertainty_map()


        return best_eval_metric

    def compute_uncertainty_map(self):
        self.uncertainty_map = {
            "concept_variance": [],
            "concept_uncertainty": [],
            "concept_weight": [],
            "label_uncertainty": [],
            "concept_entropy": [],
            "sampling_eig": [],
            "eig": [],
            "eig_concepts": [],
            "CIS": [],
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
                if self.cfg.training.acquisition_function in ["concept_variance", "certainty_concept_weight", "variance_label_uncertainty"]:
                    self.uncertainty_map["concept_variance"].extend(
                        ((idx.item(), k), relative_var[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(relative_var.shape[1])
                    )

                if self.cfg.training.acquisition_function in ["concept_uncertainty", "prob_concept_weight"]:
                    self.uncertainty_map["concept_uncertainty"].extend(
                        ((idx.item(), k), concept_uncertainty[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(concept_uncertainty.shape[1])
                    )

                if self.cfg.training.acquisition_function in ["concept_weight", "certainty_concept_weight", "prob_concept_weight"]:
                    self.uncertainty_map["concept_weight"].extend(
                        ((idx.item(), k), weights[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(weights.shape[1])
                    )

                if self.cfg.training.acquisition_function in ["label_uncertainty", "variance_label_uncertainty", "label_entropy"]:
                    self.uncertainty_map["label_uncertainty"].extend(
                        ((idx.item(), -1), label_uncertainty[b].item())
                        for b, idx in enumerate(idx_batch)
                    )

                if self.cfg.training.acquisition_function == "concept_entropy":
                    sampler = self.model.concept_sampler
                    relative_samples = sampler(relative_mean, relative_var, n_samples=16) # [16, batch_size, num_concepts]
                    bs, num_concepts = relative_samples.shape[1], relative_samples.shape[2]
                    relative_samples = relative_samples.reshape(16, -1)
                    entropy = -torch.mean(torch.log(torch.sigmoid(-relative_samples) * (1 - torch.sigmoid(-relative_samples))), dim=0)
                    entropy = entropy.reshape(bs, num_concepts)
                    self.uncertainty_map["concept_entropy"].extend(
                        ((idx.item(), k), entropy[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(entropy.shape[1])
                    )

                if self.cfg.training.acquisition_function in ["eig", "eig_concepts", "CIS", "CIS_concepts"]:
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
   
                            if self.cfg.training.acquisition_function == "eig_concepts":                            
                                lambda_weight = getattr(self.cfg.training, "eig_uncertainty_lambda", 0.1)
                                scores += lambda_weight * relative_var.squeeze()[k]


                        elif self.cfg.training.acquisition_function.startswith("CIS"):
                            expected_p = (1 - concept_prob) * q0 + concept_prob * q1
                            scores = torch.abs((expected_p - p))

                            if self.cfg.training.acquisition_function == "CIS_concepts":
                                lambda_weight = getattr(self.cfg.training, "CIS_uncertainty_lambda", 0.1)
                                scores += lambda_weight * relative_var.squeeze()[k]
                        
                        self.uncertainty_map[self.cfg.training.acquisition_function].extend(
                            ((idx.item(), k), scores[b].item())
                            for b, idx in enumerate(idx_batch)
                        )

                if self.cfg.training.acquisition_function == "sampling_eig":
                    sampler = self.model.concept_sampler    
                    n_samples = 16
                    outer_samples = sampler(relative_mean, relative_var, n_samples=n_samples) # [n_samples, batch_size, num_concepts]
                    inner_samples = sampler(relative_mean, relative_var, n_samples=n_samples) # [n_samples, batch_size, num_concepts]
                    concept_labels = batch['concept_labels'].cpu()
                    concept_labels = concept_labels.repeat(n_samples, 1, 1) # [16, batch_size, num_concepts]

                    unobserved = concept_labels == -1
                    s = torch.where(
                        unobserved,
                        inner_samples,
                        concept_labels
                    )
                    r_diff_lCso = torch.sum(weights * s, dim=-1)
                    p_y_lCso = torch.sigmoid(-r_diff_lCso).mean(dim=0)
    
                    for k in range(concept_labels.shape[-1]):                    
                        s = s.repeat(n_samples, 1, 1, 1) # [n_samples, n_samples, batch_size, num_concepts]
                        s[:, :, :, k] = outer_samples[:, :, k] # intervene on the k-th concept
                        r_diff_lCsosk = torch.sum(weights * s, dim=-1)
                        p_y_lCsosk = torch.sigmoid(-r_diff_lCsosk).mean(dim=1) # [n_samples, batch_size]
                        p_y_lCso = p_y_lCso.repeat(n_samples, 1)
                        eig = p_y_lCso * torch.log(p_y_lCso / p_y_lCsosk) + (1 - p_y_lCso) * torch.log((1 - p_y_lCso) / (1 - p_y_lCsosk))
                        eig = eig.mean(dim=0) # [batch_size]
                        self.uncertainty_map["sampling_eig"].extend(
                            ((idx.item(), k), eig[b].item())
                            for b, idx in enumerate(idx_batch)
                        )

        self.model.train()


    def run_batch(self, batch):
        batch = batch_to_device(batch, self.device)
        results = self.model(batch)
        results['loss'] = results['preference_loss'] + self.cfg.loss.beta_concept * results['concept_loss'] 
        
        if 'kl_loss' in results:
            results['loss'] += self.cfg.loss.beta_kl * results['kl_loss']

        if 'temperature' in results and self.cfg.model.use_temperature:
            temp_loss = torch.mean((results['temperature'] - 1.0) ** 2)  # Regularize to T ~ 1
            results['loss'] += self.cfg.loss.beta_temperature * temp_loss
            results['temperature_loss'] = temp_loss 
            
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
            "concept_variance", "concept_entropy", "sampling_eig", 
            "eig", "eig_concepts", "CIS", "CIS_concepts",
        ]:
            metric = self.cfg.training.acquisition_function
            sorted_pairs = sorted(self.uncertainty_map[metric], key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "concept_uncertainty":
            metric_uncertainty = [(x[0], 1 / (abs(x[-1] - 0.5) + 1e-6)) for x in self.uncertainty_map["concept_uncertainty"]]
            sorted_pairs = sorted(metric_uncertainty, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "concept_weight":
            sorted_pairs = sorted(self.uncertainty_map["concept_weight"], key=lambda x: abs(x[1]))  # absolute value if you want low magnitude
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]
        
        elif self.cfg.training.acquisition_function == "certainty_concept_weight":
            contribution = [(self.uncertainty_map["variance"][i][0], self.uncertainty_map["variance"][i][-1] * abs(self.uncertainty_map["concept_weight"][i][-1])) for i in range(len(self.uncertainty_map["concept_uncertainty"]))]
            sorted_pairs = sorted(contribution, key=lambda x: -x[1]) 
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "prob_concept_weight":
            contribution = [(self.uncertainty_map["concept_uncertainty"][i][0], self.uncertainty_map["concept_uncertainty"][i][-1] * abs(self.uncertainty_map["concept_weight"][i][-1])) for i in range(len(self.uncertainty_map["concept_uncertainty"]))]
            sorted_pairs = sorted(contribution, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]
        
        elif self.cfg.training.acquisition_function == "label_uncertainty":
            label_uncertainty_metric = [(x[0], 1 / (abs(x[-1] - 0.5) + 1e-6)) for x in self.uncertainty_map["label_uncertainty"]]
            sorted_pairs = sorted(label_uncertainty_metric, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "label_entropy":
            label_uncertainty_metric = [(x[0], bernoulli_stats(x[-1])) for x in self.uncertainty_map["label_uncertainty"]]
            sorted_pairs = sorted(label_uncertainty_metric, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "variance_label_uncertainty":
            # Combine concept variance and label uncertainty
            label_uncertainty_scores = [(x[0], bernoulli_stats(x[-1])) for x in self.uncertainty_map["label_uncertainty"]]

            combined_scores = []
            for (idx, concept_idx), var_score in self.uncertainty_map["concept_variance"]:

                label_score = [v for (i, c), v in label_uncertainty_scores if i == idx and c == concept_idx][0]
                # Combine them (we can tune the balance between them with a lambda)
                lambda_balance = getattr(self.cfg.training, "variance_label_lambda", 1.0)
                combined_score = var_score + lambda_balance * label_score
                combined_scores.append(((idx, concept_idx), combined_score))

            combined_scores.sort(key=lambda x: -x[1])
            added_idx = [idx for (idx, _) in combined_scores[:self.cfg.training.num_acquired_samples]]
        
        else:
            raise NotImplementedError(
                f"Acquisition function {self.cfg.training.acquisition_function} not implemented"
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