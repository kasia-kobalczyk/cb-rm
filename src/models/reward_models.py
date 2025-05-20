import torch.nn as nn
import torch
import numpy as np
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import torch.nn.functional as F

class TemperatureNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        temp = self.net(x) 
        temp = torch.nn.functional.softplus(temp) + 1e-3  # ensure positivity
        return temp.squeeze(-1) 

class BottleneckRewardModel(nn.Module):
    def __init__(
            self, 
            concept_encoder,
            gating_network,
            use_temperature=False,
            unmask_y=True
        ):
        super(BottleneckRewardModel, self).__init__()
        self.concept_encoder = concept_encoder
        self.gating_network = gating_network
        self.unmask_y = unmask_y
        # self.scalarizer = scalarizer
    
    def get_preference_loss(self, reward_diff, preference_labels, concept_mask):
        if torch.sum(concept_mask) == 0:
            preference_loss = torch.zeros(1).to(preference_labels.device)
            accuracy = np.nan 
        else:
            if self.unmask_y:
                preference_probs = torch.sigmoid(-reward_diff).unsqueeze(1) # Negative since labels are flipped reward_a > reward_b -> label = 0.0
                preference_loss = torch.nn.functional.binary_cross_entropy(
                    preference_probs.float(), preference_labels.float(), reduction='none'
                )
                preference_loss = torch.mean(preference_loss)
                predictions = torch.round(preference_probs)
                accuracy = torch.mean((predictions == preference_labels.float()).float())
            else:
                preference_probs = torch.sigmoid(-reward_diff).unsqueeze(1) # Negative since labels are flipped reward_a > reward_b -> label = 0.0
                row_has_zero = (concept_mask != 1).any(dim=1, keepdim=True)
                new_mask = (~row_has_zero).float()

                preference_probs = preference_probs*new_mask
                preference_labels = preference_labels*new_mask
                preference_loss = torch.nn.functional.binary_cross_entropy(
                    preference_probs.float(), preference_labels.float(), reduction='none'
                )
                # preference_loss = torch.mean(preference_loss)
                preference_loss = torch.sum(preference_loss) / torch.sum(new_mask)
                # accuracy
                # predictions = torch.round(preference_probs)
                predictions = torch.round(preference_probs)*new_mask
                accuracy = torch.mean((predictions == preference_labels.float()).float())
            
        return preference_loss, accuracy

    def get_concept_loss(self, relative_concept_logits, concept_labels):
        concept_probs = torch.sigmoid(-relative_concept_logits) 
        concept_mask = torch.where(
            concept_labels  >=0.0,
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
            # concept_loss = torch.sum(concept_loss) / torch.sum(concept_mask)
            concept_loss = (concept_loss.sum(dim=0)/concept_loss.shape[0]).sum()
            hard_labels = torch.round(concept_labels) * concept_mask
            predictions = torch.round(concept_probs) * concept_mask
            pseudo_accuracy = torch.mean((predictions == hard_labels).float())

        return concept_loss, pseudo_accuracy, concept_mask

    def forward(self, batch):
        concept_logits_a = self.concept_encoder(batch['example_a']['prompt_response_embedding'])
        concept_logits_b = self.concept_encoder(batch['example_b']['prompt_response_embedding'])
        
        weights = self.gating_network(batch['example_a']['prompt_embedding'])

        relative_concept_logits = concept_logits_a - concept_logits_b

        reward_diff = torch.sum(weights * relative_concept_logits, dim=1)
        
        # reward_diff = self.scalarizer(relative_concept_logits)  # Use linear scalarizer directly


        concept_loss, concept_pseudo_acc, concept_mask = self.get_concept_loss(
            relative_concept_logits, batch['concept_labels']
        )

        preference_loss, preference_acc = self.get_preference_loss(
            reward_diff, batch['preference_labels'], concept_mask
        )        

        return {
            'preference_loss': preference_loss,
            'preference_accuracy': preference_acc,
            'concept_loss': concept_loss,
            'concept_pseudo_accuracy': concept_pseudo_acc,
        }



class GaussianSampler(nn.Module):
    def forward(self, mean, variance, n_samples=1):
        q = torch.distributions.Normal(mean, variance)
        if n_samples == 1:
            return q.rsample()
        else:
            return q.rsample((n_samples,)) 

    def kl_divergence(self, mean, variance):
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(variance))  # Standard Gaussian prior
        q = torch.distributions.Normal(mean, variance)
        kl_loss = torch.distributions.kl_divergence(q, p)
        return kl_loss.mean()


class ProbabilisticBottleneckRewardModel(BottleneckRewardModel):
    def __init__(
            self, 
            concept_encoder,
            # scalarizer,
            gating_network,
            concept_sampler,
            use_temperature=False,
            unmask_y=True
        ):
        super(ProbabilisticBottleneckRewardModel, self).__init__(concept_encoder, gating_network, unmask_y)

        self.concept_sampler_name = concept_sampler
        self.use_temperature = use_temperature

        if concept_sampler == "gaussian":
            self.concept_sampler = GaussianSampler()
        
        if self.use_temperature:
            self.temperature_network = TemperatureNetwork(
                input_dim=concept_encoder.model[0].in_features,
                hidden_dim=concept_encoder.model[-1].in_features
            )
        else:
            self.temperature_network = None

    def forward(self, batch):
        out_a = self.concept_encoder(batch['example_a']['prompt_response_embedding'])
        out_b = self.concept_encoder(batch['example_b']['prompt_response_embedding'])

        mean_a, var_a = out_a.chunk(2, dim=-1)
        mean_b, var_b = out_b.chunk(2, dim=-1)

        var_a = torch.nn.functional.softplus(var_a)
        var_b = torch.nn.functional.softplus(var_b)

        relative_mean = mean_a - mean_b
        relative_var = var_a + var_b

        relative_concept_logits = self.concept_sampler(relative_mean, relative_var)
        # relative_concept_logits_a = self.concept_sampler(mean_a, var_a)
        # relative_concept_logits_b = self.concept_sampler(mean_b, var_b)
        # relative_concept_logits = relative_concept_logits_a - relative_concept_logits_b
        kl_loss = self.concept_sampler.kl_divergence(relative_mean, relative_var) #+  \
                #   self.concept_sampler.kl_divergence(mean_a, var_a) + \
                #   self.concept_sampler.kl_divergence(mean_b, var_b)
                  # Potentially add regularization on a and b as well
        
        weights = self.gating_network(batch['example_a']['prompt_embedding'])
        # reward_diff = self.scalarizer(relative_concept_logits) 
        # weights = None

        reward_diff = torch.sum(weights * relative_concept_logits, dim=1)
        # reward_diff_a = torch.sum(weights * relative_concept_logits_a, dim=1)
        # reward_diff_b = torch.sum(weights * relative_concept_logits_b, dim=1)
        # reward_diff = reward_diff_a - reward_diff_b
        if self.use_temperature:
            temperature = self.temperature_network(batch['example_a']['prompt_embedding'])
            reward_diff = reward_diff / temperature
        else:
            temperature = torch.ones_like(reward_diff)
            
        concept_loss, concept_pseudo_acc, concept_mask = self.get_concept_loss(
            relative_concept_logits, batch['concept_labels']
        )
        preference_loss, preference_acc = self.get_preference_loss(
            reward_diff, batch['preference_labels'], concept_mask
        )
        
        return {
            'preference_loss': preference_loss,
            'preference_accuracy': preference_acc,
            'concept_loss': concept_loss,
            'concept_pseudo_accuracy': concept_pseudo_acc,
            'kl_loss': kl_loss,
            'relative_mean': relative_mean,
            'relative_var': relative_var,
            'relative_concept_logits': relative_concept_logits,
            'reward_diff': reward_diff,
            'weights': weights,
            'temperature': temperature
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
    
# class GatingNetwork(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         bias: bool = True,
#         temperature: float = 10,
#         logit_scale: float = 1.0,
#         hidden_dim: int = 1024,
#         n_hidden: int = 3,
#         dropout: float = 0.2,
#     ):
#         super().__init__()
#         self.temperature = temperature
#         self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
#         self.dropout_prob = dropout
#         layers = []
#         for _ in range(n_hidden):
#             layers.append(nn.Linear(input_dim, hidden_dim))
#             input_dim = hidden_dim
#         layers.append(nn.Linear(input_dim, output_dim, bias=bias))
#         self.layers = nn.ModuleList(layers)

#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i < len(self.layers) - 1:
#                 x = F.relu(x)
#                 if self.dropout_prob > 0:
#                     x = F.dropout(x, p=self.dropout_prob, training=self.training)
#         x = F.softmax(x / self.temperature, dim=1)
#         return x * self.logit_scale[0]

# class LinearScalarization(nn.Module):
#     def __init__(self, input_dim: int):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, 1, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Ensure linear layer is on the same device as x
#         if self.linear.weight.device != x.device:
#             self.linear.to(x.device)
#         return self.linear(x).squeeze(-1)  # shape: [B]

class LinearScalarization(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(input_dim))  # Learnable raw weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_concepts]
        weights = torch.softmax(self.logits, dim=0)  # [num_concepts], sums to 1
        return (x * weights).sum(dim=1)  # [B]
