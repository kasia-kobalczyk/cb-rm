import torch.nn as nn
import torch
import numpy as np
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class BottleneckRewardModel(nn.Module):
    def __init__(
            self, 
            concept_encoder,
            gating_network,
        ):
        super(BottleneckRewardModel, self).__init__()
        self.concept_encoder = concept_encoder
        self.gating_network = gating_network
    
    def get_preference_loss(self, reward_diff, preference_labels):
        preference_probs = torch.sigmoid(-reward_diff).unsqueeze(1) # Negative since labels are flipped reward_a > reward_b -> label = 0.0
        preference_loss = torch.nn.functional.binary_cross_entropy(
            preference_probs, preference_labels.float(), reduction='none'
        )
        preference_loss = torch.mean(preference_loss)
        # accuracy
        predictions = torch.round(preference_probs)
        accuracy = torch.mean((predictions == preference_labels.float()).float())
        return preference_loss, accuracy

    def get_concept_loss(self, relative_concept_logits, concept_labels):
        concept_probs = torch.sigmoid(-relative_concept_logits) 
        concept_mask = torch.where(
            concept_labels != -1.0,
            torch.ones_like(concept_probs),
            torch.zeros_like(concept_probs)
        ).to(concept_probs.device)
        
        if torch.sum(concept_mask) == 0:
            concept_loss = torch.zeros(1).to(concept_probs.device)
            pseudo_accuracy = np.nan 
        else:
            # Apply binary cross entropy loss to each concept dimension independently
            # Use vectorized version of binary cross entropy to avoid for loop
            vectorised_loss = torch.func.vmap(torch.nn.functional.binary_cross_entropy) 
            concept_probs = concept_probs * concept_mask
            concept_labels = concept_labels * concept_mask   
            concept_loss = vectorised_loss(
                concept_probs, concept_labels.float(), reduction='none'
            )
            concept_loss = torch.sum(concept_loss) / torch.sum(concept_mask)

            hard_labels = torch.round(concept_labels) * concept_mask
            predictions = torch.round(concept_probs) * concept_mask
            pseudo_accuracy = torch.mean((predictions == hard_labels).float())

        return concept_loss, pseudo_accuracy

    def forward(self, batch):
        concept_logits_a = self.concept_encoder(batch['example_a']['prompt_response_embedding'])
        concept_logits_b = self.concept_encoder(batch['example_b']['prompt_response_embedding'])
        
        weights = self.gating_network(batch['example_a']['prompt_embedding'])

        relative_concept_logits = concept_logits_a - concept_logits_b

        reward_diff = torch.sum(weights * relative_concept_logits, dim=1)

        preference_loss, preference_acc = self.get_preference_loss(
            reward_diff, batch['preference_labels']
        )

        concept_loss, concept_pseudo_acc = self.get_concept_loss(
            relative_concept_logits, batch['concept_labels']
        )

        return {
            'preference_loss': preference_loss,
            'preference_accuracy': preference_acc,
            'concept_loss': concept_loss,
            'concept_pseudo_accuracy': concept_pseudo_acc,
        }



class GaussianSampler(nn.Module):
    def forward(self, mean, variance):
        q = torch.distributions.Normal(mean, variance)
        return q.rsample()

    def kl_divergence(self, mean, variance):
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(variance))  # Standard Gaussian prior
        q = torch.distributions.Normal(mean, variance)
        kl_loss = torch.distributions.kl_divergence(q, p)
        return kl_loss.mean()


class ProbabilisticBottleneckRewardModel(BottleneckRewardModel):
    def __init__(
            self, 
            concept_encoder,
            gating_network,
            concept_sampler,
        ):
        super(ProbabilisticBottleneckRewardModel, self).__init__(concept_encoder, gating_network)

        self.concept_sampler_name = concept_sampler

        if concept_sampler == "gaussian":
            self.concept_sampler = GaussianSampler()

    def forward(self, batch):
        out_a = self.concept_encoder(batch['example_a']['prompt_response_embedding'])
        out_b = self.concept_encoder(batch['example_b']['prompt_response_embedding'])

        mean_a, var_a = out_a.chunk(2, dim=-1)
        mean_b, var_b = out_b.chunk(2, dim=-1)

        var_a = torch.nn.functional.softplus(var_a)
        var_b = torch.nn.functional.softplus(var_b)

        concept_logits_a = self.concept_sampler(mean_a, var_a)
        concept_logits_b = self.concept_sampler(mean_b, var_b)

        relative_concept_logits = concept_logits_a - concept_logits_b

        kl_loss = self.concept_sampler.kl_divergence(mean_a, var_a) + \
                    self.concept_sampler.kl_divergence(mean_b, var_b)
        
        weights = self.gating_network(batch['example_a']['prompt_embedding'])
        
        reward_diff = torch.sum(weights * relative_concept_logits, dim=1)

        preference_loss, preference_acc = self.get_preference_loss(
            reward_diff, batch['preference_labels']
        )

        concept_loss, concept_pseudo_acc = self.get_concept_loss(
            relative_concept_logits, batch['concept_labels']
        )

        return {
            'preference_loss': preference_loss,
            'preference_accuracy': preference_acc,
            'concept_loss': concept_loss,
            'concept_pseudo_accuracy': concept_pseudo_acc,
            'kl_loss': kl_loss,
        }

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()
        if num_layers == 0:
            self.model = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh() # Returns values between -1 and 1

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x
