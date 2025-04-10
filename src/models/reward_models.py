import torch.nn as nn
import torch
import numpy as np

class BottleneckRewardModel(nn.Module):
    def __init__(
            self, 
            concept_encoder,
            gating_network,
        ):
        super(BottleneckRewardModel, self).__init__()
        self.concept_encoder = concept_encoder
        self.gating_network = gating_network
    
    def get_preference_loss(self, reward_a, reward_b, preference_labels):
        preference_probs = torch.sigmoid(reward_a - reward_b).unsqueeze(1)
        preference_loss = torch.nn.functional.binary_cross_entropy(
            preference_probs, preference_labels.float(), reduction='none'
        )
        preference_loss = torch.mean(preference_loss)
        # accuracy
        predictions = torch.round(preference_probs)
        accuracy = torch.mean((predictions == preference_labels.float()).float())
        return preference_loss, accuracy

    def get_concept_loss(self, concept_scores_a, concept_scores_b, concept_labels):
        concept_probs = torch.sigmoid(concept_scores_a - concept_scores_b)
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
        concept_scores_a = self.concept_encoder(batch['example_a']['prompt_response_embedding'])
        concept_scores_b = self.concept_encoder(batch['example_b']['prompt_response_embedding'])
        
        weights = self.gating_network(batch['example_a']['prompt_embedding'])

        reward_a = torch.sum(weights * concept_scores_a, dim=1)
        reward_b = torch.sum(weights * concept_scores_b, dim=1)

        preference_loss, preference_acc = self.get_preference_loss(
            reward_a, reward_b, batch['preference_labels']
        )

        concept_loss, concept_pseudo_acc = self.get_concept_loss(
            concept_scores_a, concept_scores_b, batch['concept_labels']
        )

        return {
            'preference_loss': preference_loss,
            'preference_accuracy': preference_acc,
            'concept_loss': concept_loss,
            'concept_pseudo_accuracy': concept_pseudo_acc,
        }

class SimpleConceptEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleConceptEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Sigmoid() # Returns values between 0 and 1

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
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


if __name__ == "__main__":
    from src.data.dataloaders import get_dataloader, batch_to_device
    from argparse import Namespace
    
    cfg = Namespace(
        embeddings_path='./datasets/ultrafeedback/embeddings/meta-llama/Meta-Llama-3-8B/',
        splits_path='./datasets/ultrafeedback/splits.csv',
        concept_labels_path='./datasets/ultrafeedback/concept_labels/meta-llama/Meta-Llama-3-70B-Instruct',
        preference_labels_path='./datasets/ultrafeedback/preference_labels/openbmb_average',
        batch_size=64,
    )

    dataloader = get_dataloader(cfg, split='train')
    print(f'Number of batches: {len(dataloader)}')
    
    input_dim = 4096
    output_dim = len(dataloader.dataset.concept_names)

    concept_encoder = SimpleConceptEncoder(input_dim, output_dim)
    gating_network = GatingNetwork(input_dim, output_dim)
    model = BottleneckRewardModel(concept_encoder, gating_network).to('cuda:0')

    for batch in dataloader:
        batch = batch_to_device(batch, 'cuda:0')
        model = model.to('cuda:0')
        output = model(batch)
        print(output['preference_probs'].shape)
        print(output['concept_probs'].shape)

        # concetp_labels (bs, num_concepts),  concept_logprobs: (bs, num_concepts)
        # apply cross_entropy_loss to each concept dimension independently:
        

        print(f'Preference Loss: {output["preference_loss"]}')
        print(f'Concept Loss: {output["concept_loss"]}')

        break