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
        self.current_episode = 0

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
            self.current_episode += 1
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
        if self.cfg.model.model_type == "probabilistic" and self.cfg.training.acquisition_function != "uniform":
            self.compute_uncertainty_map()


        return best_eval_metric

    def compute_uncertainty_map(self):
        self.uncertainty_map = {
            "variance": [],
            "concept_uncertainty": [],
            "concept_weight": [],
            "label_uncertainty": [],
        }
        self.model.eval()
        with torch.no_grad():
            pool_index = self.train_dataset.pool_index
            pool_instance_index = list({instance_idx for (instance_idx, _) in pool_index})
            pool_dataset = torch.utils.data.Subset(self.train_dataset, pool_instance_index)
            pool_dataloader = DataLoader(
                pool_dataset,
                batch_size=self.cfg.data.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            print(f"Computing uncertainty for the pool dataset of size {len(pool_dataset)} ...")
            for batch in tqdm(pool_dataloader):
                results = self.run_batch(batch)
                relative_var = results["relative_var"].detach().cpu()
                concept_logits = results["relative_concept_logits"].detach().cpu()
                reward_diff = results["reward_diff"].detach().cpu()
                weights = results["weights"].detach().cpu()
                idx_batch = batch["pair_idx"]
                concept_uncertainty = torch.sigmoid(-concept_logits)
                label_uncertainty = torch.sigmoid(-reward_diff)

                # Now fill the uncertainty_map dictionary
                self.uncertainty_map["variance"].extend(
                    ((idx.item(), k), relative_var[b, k].item())
                    for b, idx in enumerate(idx_batch)
                    for k in range(relative_var.shape[1])
                )
                self.uncertainty_map["concept_uncertainty"].extend(
                    ((idx.item(), k), concept_uncertainty[b, k].item())
                    for b, idx in enumerate(idx_batch)
                    for k in range(concept_uncertainty.shape[1])
                )
                self.uncertainty_map["concept_weight"].extend(
                    ((idx.item(), k), weights[b, k].item())
                    for b, idx in enumerate(idx_batch)
                    for k in range(weights.shape[1])
                )
                self.uncertainty_map["label_uncertainty"].extend(
                    ((idx.item(), -1), label_uncertainty[b].item())
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

        elif self.cfg.training.acquisition_function == "variance":
            sorted_pairs = sorted(self.uncertainty_map["variance"], key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "concept_uncertainty":
            metric_uncertainty = 1/abs(self.uncertainty_map["concept_uncertainty"] - 0.5)
            sorted_pairs = sorted(metric_uncertainty, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "concept_weight":
            sorted_pairs = sorted(self.uncertainty_map["concept_weight"], key=lambda x: abs(x[1]))  # absolute value if you want low magnitude
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]
        
        elif self.cfg.training.acquisition_function == "certainty_concept_weight":
            contribution = self.uncertainty_map["variance"] * abs(self.uncertainty_map["concept_weight"])
            sorted_pairs = sorted(contribution, key=lambda x: -x[1]) 
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]
        
        elif self.cfg.training.acquisition_function == "label_uncertainty":
            label_uncertainty_metric = 1/abs(self.uncertainty_map["label_uncertainty"] - 0.5)
            sorted_pairs = sorted(label_uncertainty_metric, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if (pair[0], 0) in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "variance_label_uncertainty":
            # Combine concept variance and label uncertainty
            variance_scores = dict(self.uncertainty_map["variance"])
            label_uncertainty_scores = dict(self.uncertainty_map["label_uncertainty"])

            combined_scores = []
            for (idx, concept_idx), var_score in variance_scores.items():
                # Find the corresponding label uncertainty (idx, -1) for the sample
                label_score = label_uncertainty_scores.get((idx, -1), 0.0)

                # Combine them (you can tune the balance between them with a lambda)
                lambda_balance = getattr(self.cfg.training, "variance_label_lambda", 1.0)
                combined_score = var_score + lambda_balance * label_score

                combined_scores.append(((idx, concept_idx), combined_score))

            combined_scores.sort(key=lambda x: -x[1])
            added_idx = [idx for (idx, _) in combined_scores[:self.cfg.training.num_acquired_samples]]

        elif self.cfg.training.acquisition_function == "temperature_concept_uncertainty":
            # Combine temperature uncertainty and concept uncertainty
            temperature_scores = dict(self.uncertainty_map["temperature"])  # (idx, -1) -> temperature
            concept_scores = self.uncertainty_map["concept_uncertainty"]    # (idx, concept_idx) -> uncertainty

            combined_scores = []
            for (idx, concept_idx), concept_unc in concept_scores:
                temp_unc = temperature_scores.get((idx, -1), 1.0)  # default temp 1.0 if missing
                combined_unc = temp_unc * concept_unc
                combined_scores.append(((idx, concept_idx), combined_unc))

            combined_scores.sort(key=lambda x: -x[1])  # descending order
            added_idx = [idx for (idx, _) in combined_scores[:self.cfg.training.num_acquired_samples]]

        elif self.cfg.training.acquisition_function in ["expected_information_gain", "expected_information_gain_concepts"]:
            # Shared computation
            self.model.eval()
            eig_scores = []

            with torch.no_grad():
                pool_idx = list(self.train_dataset.pool_index)
                for idx, concept_idx in pool_idx:
                    example = self.train_dataset.get_example_by_index(idx)
                    x = example['prompt_response_embedding'].unsqueeze(0).to(self.device)
                    prompt_embedding = example['prompt_embedding'].unsqueeze(0).to(self.device)

                    concept_logits = self.model.concept_encoder(x)
                    weights = self.model.gating_network(prompt_embedding)

                    reward_pred = torch.sum(weights * concept_logits).item()

                    # Flip the specific concept
                    intervened_logits = concept_logits.clone()
                    intervened_logits[0, concept_idx] = -intervened_logits[0, concept_idx]  # flip

                    reward_with_intervention = torch.sum(weights * intervened_logits).item()

                    delta = abs(reward_with_intervention - reward_pred)

                    # Now, depending on the mode, maybe add extra score
                    if self.cfg.training.acquisition_function == "expected_information_gain_concepts":
                        concept_prob = torch.sigmoid(concept_logits)[0, concept_idx]
                        concept_uncertainty = min(concept_prob, 1.0 - concept_prob).item()
                        lambda_weight = getattr(self.cfg.training, "eig_uncertainty_lambda", 1.0)
                        final_score = delta + lambda_weight * concept_uncertainty
                    else:
                        final_score = delta

                    eig_scores.append(((idx, concept_idx), final_score))

                eig_scores.sort(key=lambda x: -x[1])
                added_idx = [idx for (idx, _) in eig_scores[:self.cfg.training.num_acquired_samples]]


        else:
            raise NotImplementedError(
                f"Acquisition function {self.cfg.training.acquisition_funtion} not implemented"
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