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
    intervened_logits[0, concept_idx] = intervene_value
    reward = torch.sum(weights * intervened_logits, dim=1)
    return reward, intervened_logits

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
        self.var_log = []  
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
        self.var_log = {
            "variance": [],
            "concept_uncertainty": [],
            "concept_weight": [],
            "label_uncertainty": [],
            "temperature": []
        }
        best_eval_metric = self.best_eval_metric
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in tqdm(self.train_dataloader):
            # for i, batch in enumerate(tqdm(self.train_dataloader)):
                # if i >= 3:
                #     break
                self.model.train()
                self.optimizer.zero_grad()
                results = self.run_batch(batch)
                loss = results["loss"]
                loss.backward()
                self.optimizer.step()
                if self.cfg.model.model_type == "probabilistic":
                    relative_var = results["relative_var"].detach().cpu()
                    concept_logits = results["relative_concept_logits"].detach().cpu()
                    reward_diff = results["reward_diff"].detach().cpu()
                    weights = results["weights"].detach().cpu()
                    temperature = results["temperature"].detach().cpu()
                    idx_batch = batch["id"]
                    concept_uncertainty = torch.sigmoid(-concept_logits)
                    label_uncertainty = torch.sigmoid(-reward_diff)

                    # Now fill the var_log dictionary
                    self.var_log["variance"].extend(
                        ((idx.item(), k), relative_var[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(relative_var.shape[1])
                    )
                    self.var_log["concept_uncertainty"].extend(
                        ((idx.item(), k), concept_uncertainty[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(concept_uncertainty.shape[1])
                    )
                    self.var_log["concept_weight"].extend(
                        ((idx.item(), k), weights[b, k].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(weights.shape[1])
                    )
                    self.var_log["label_uncertainty"].extend(
                        ((idx.item(), k), label_uncertainty[b].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(relative_var.shape[1])
                    )
                    self.var_log["temperature"].extend(
                        ((idx.item(), k), temperature[b].item())
                        for b, idx in enumerate(idx_batch)
                        for k in range(relative_var.shape[1])
                    )
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

        return best_eval_metric

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
        elif self.cfg.training.acquisition_function == "concept_variance":
            sorted_pairs = sorted(self.var_log["variance"], key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "concept_uncertainty":
            metric_uncertainty = [(x[0], 1 / abs(x[-1] - 0.5)) for x in self.var_log["concept_uncertainty"]]
            sorted_pairs = sorted(metric_uncertainty, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "concept_weight":
            sorted_pairs = sorted(self.var_log["concept_weight"], key=lambda x: abs(x[1]))  # absolute value if you want low magnitude
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]
  
        elif self.cfg.training.acquisition_function == "certainty_concept_weight":
            contribution = [(self.var_log["variance"][i][0], self.var_log["variance"][i][-1] * abs(self.var_log["concept_weight"][i][-1])) for i in range(len(self.var_log["concept_uncertainty"]))]
            sorted_pairs = sorted(contribution, key=lambda x: -x[1]) 
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]
        
        elif self.cfg.training.acquisition_function == "prob_concept_weight":
            contribution = [(self.var_log["concept_uncertainty"][i][0], self.var_log["concept_uncertainty"][i][-1] * abs(self.var_log["concept_weight"][i][-1])) for i in range(len(self.var_log["concept_uncertainty"]))]
            sorted_pairs = sorted(contribution, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if pair in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "label_uncertainty":
            label_uncertainty_metric = [(x[0], 1 / abs(x[-1] - 0.5)) for x in self.var_log["label_uncertainty"]]
            sorted_pairs = sorted(label_uncertainty_metric, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if (pair[0], 0) in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]
        elif self.cfg.training.acquisition_function == "label_entropy":
            label_uncertainty_metric = [(x[0], bernoulli_stats(x[-1])) for x in self.var_log["label_uncertainty"]]
            sorted_pairs = sorted(label_uncertainty_metric, key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if (pair[0], 0) in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "temperature":
            sorted_pairs = sorted(self.var_log["temperature"], key=lambda x: -x[1])
            added_idx = [pair for (pair, _) in sorted_pairs if (pair[0], 0) in self.train_dataset.pool_index][:self.cfg.training.num_acquired_samples]

        elif self.cfg.training.acquisition_function == "variance_label_uncertainty":
            # Combine concept variance and label uncertainty
            # label_uncertainty_scores = {(x[0], x[1]): 1 / abs(x[-1] - 0.5) for x in self.var_log["label_uncertainty"]}
            label_uncertainty_scores = [(x[0], bernoulli_stats(x[-1])) for x in self.var_log["label_uncertainty"]]

            combined_scores = []
            for (idx, concept_idx), var_score in self.var_log["variance"]:

                label_score = next(v for (i, c), v in label_uncertainty_scores if i == idx and c == concept_idx)
                # Combine them (we can tune the balance between them with a lambda)
                lambda_balance = getattr(self.cfg.training, "variance_label_lambda", 1.0)
                combined_score = var_score + lambda_balance * label_score
                combined_scores.append(((idx, concept_idx), combined_score))

            combined_scores.sort(key=lambda x: -x[1])
            added_idx = [idx for (idx, _) in combined_scores[:self.cfg.training.num_acquired_samples]]

        elif self.cfg.training.acquisition_function == "temperature_concept_uncertainty":
            # Combine temperature uncertainty and concept uncertainty
            combined_scores = []
            for (idx, concept_idx), var_score in self.var_log["variance"]:

                label_score = next(v for (i, c), v in self.var_log["temperature"] if i == idx and c == concept_idx)

                # Combine them (we can tune the balance between them with a lambda)
                lambda_balance = getattr(self.cfg.training, "variance_label_lambda", 1.0)
                combined_score = var_score + lambda_balance * label_score
                combined_scores.append(((idx, concept_idx), combined_score))

            combined_scores.sort(key=lambda x: -x[1])
            added_idx = [idx for (idx, _) in combined_scores[:self.cfg.training.num_acquired_samples]]                
        
        elif self.cfg.training.acquisition_function in [ "expected_information_gain",  "expected_information_gain_concepts", "expected_target_uncertainty_reduction",  "intervention_change"]:

            self.model.eval()
            scores = []

            with torch.no_grad():
                pool_idx = list(self.train_dataset.pool_index)
                for idx, concept_idx in pool_idx:
                    pair_idx = self.train_dataset.pairs_data.index.get_loc(idx)
                    example = self.train_dataset[pair_idx]

                    embedding_a = to_tensor(example['example_a']['prompt_response_embedding'], self.device)
                    out_a = self.model.concept_encoder(embedding_a)

                    embedding_b = to_tensor(example['example_b']['prompt_response_embedding'], self.device)
                    out_b = self.model.concept_encoder(embedding_b)

                    mean_a, var_a = out_a.chunk(2, dim=-1)
                    mean_b, var_b = out_b.chunk(2, dim=-1)

                    var_a = torch.nn.functional.softplus(var_a)
                    var_b = torch.nn.functional.softplus(var_b)

                    relative_mean = mean_a - mean_b
                    relative_var = var_a + var_b

                    relative_concept_logits = self.model.concept_sampler(relative_mean, relative_var)
                    weights = self.model.gating_network(to_tensor(example['example_a']['prompt_embedding'], self.device))
                    reward_diff = torch.sum(weights * relative_concept_logits, dim=1)

                    concept_logit = relative_concept_logits[0, concept_idx]
                    concept_prob = torch.sigmoid(-concept_logit)

                    # Before intervention
                    p = torch.sigmoid(-reward_diff)

                    # Intervene 0
                    reward_0, _ = intervene_logits(relative_concept_logits, concept_idx, weights, 10.0)
                    q0 = torch.sigmoid(-reward_0)

                    # Intervene 1
                    reward_1, _ = intervene_logits(relative_concept_logits, concept_idx, weights, -10.0)
                    q1 = torch.sigmoid(-reward_1)

                    # ==== Different scoring ====
                    if self.cfg.training.acquisition_function.startswith("expected_information_gain"):
                        kl_0 = bernoulli_stats(p, q0)
                        kl_1 = bernoulli_stats(p, q1)
                        score = (concept_prob * kl_1 + (1 - concept_prob) * kl_0).item()
                        if self.cfg.training.acquisition_function == "expected_information_gain_concepts":                            
                            lambda_weight = getattr(self.cfg.training, "eig_uncertainty_lambda", 0.1)
                            score += lambda_weight * relative_var.squeeze()[concept_idx]

                    elif self.cfg.training.acquisition_function.startswith("expected_target_uncertainty_reduction"):
                        entropy_before = bernoulli_stats(p)
                        entropy_0 = bernoulli_stats(q0)
                        entropy_1 = bernoulli_stats(q1)

                        expected_entropy_after = concept_prob * entropy_1 + (1 - concept_prob) * entropy_0
                        score = (entropy_before - expected_entropy_after).item()
                        if self.cfg.training.acquisition_function == "expected_target_uncertainty_reduction_concepts":                            
                            lambda_weight = getattr(self.cfg.training, "etr_uncertainty_lambda", 0.1)
                            score += lambda_weight * relative_var.squeeze()[concept_idx]


                    elif self.cfg.training.acquisition_function.startswith("CIS"):
                        expected_p = (1 - concept_prob) * q0 + concept_prob * q1
                        score = abs((expected_p - p).item())
                        if self.cfg.training.acquisition_function == "CIS_concepts":
                            lambda_weight = getattr(self.cfg.training, "CIS_uncertainty_lambda", 0.1)
                            score += lambda_weight * relative_var.squeeze()[concept_idx]
                    
                    scores.append(((idx, concept_idx), score))
            scores.sort(key=lambda x: -x[1])
            added_idx = [idx for (idx, _) in scores[:self.cfg.training.num_acquired_samples]]

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