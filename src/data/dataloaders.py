from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import pandas as pd
import torch 
import numpy as np

class PreferenceDataset(Dataset):
    def __init__(
            self, 
            embeddings_path,
            splits_path,
            concept_labels_path,
            preference_labels_path,
            split='train'
        ):
        
        # Load the embeddings
        self.embeddings = load_from_disk(embeddings_path)
        self.embeddings_df = self.embeddings.to_pandas().set_index('idx')
        self.embeddings_df = self.embeddings_df[self.embeddings_df.success]

        # Load train/val/test splits
        splits_df = pd.read_csv(splits_path)
        splits_df = splits_df[splits_df['split'] == split]

        # Load labels
        concept_labels_df = load_from_disk(concept_labels_path).to_pandas()
        preference_labels_df = load_from_disk(preference_labels_path).to_pandas()

        # Create index of response pairs with labels
        self.pairs_data = pd.merge(
            splits_df,
            concept_labels_df,
            left_on=['idx_a', 'idx_b'],
            right_on=['idx_a', 'idx_b'],
            how='left',
        )
        self.pairs_data = pd.merge(
            self.pairs_data,
            preference_labels_df,
            left_on=['idx_a', 'idx_b'],
            right_on=['idx_a', 'idx_b'],
            how='left',
        )
        
        assert len(self.pairs_data) == len(splits_df)
        
        # Filter to pairs that are in the embeddings
        self.pairs_data = self.pairs_data[
            (self.pairs_data['idx_a'].isin(self.embeddings_df.index)) &
            (self.pairs_data['idx_b'].isin(self.embeddings_df.index))
        ]
        
        self.concept_names = concept_labels_df.iloc[0]['concept_names']

    def __getitem__(self, pair_idx):
        idx_a = self.pairs_data.iloc[pair_idx]['idx_a']
        idx_b = self.pairs_data.iloc[pair_idx]['idx_b']
        preference_label = self.pairs_data.iloc[pair_idx]['preference_label']
        concept_labels = self.pairs_data.iloc[pair_idx]['relative_concept_labels']
        
        
        if concept_labels is None or (isinstance(concept_labels, float) and np.isnan(concept_labels)):
            concept_labels = torch.ones(len(self.concept_names)) * -1.0
        else:
            concept_labels = torch.tensor(concept_labels)

        return {
            'example_a': self.construct_example(idx_a),
            'example_b': self.construct_example(idx_b),
            'preference_label': preference_label,
            'concept_labels': concept_labels,
        }

    def construct_example(self, example_idx):
        example = self.embeddings_df.loc[example_idx]
  
        return {
            'prompt': example['prompt'],
            'response': example['response'],
            'prompt_embedding': example['prompt_embedding'],
            'prompt_response_embedding': example['prompt_response_embedding'],
        }

    def __len__(self):
        return len(self.pairs_data)


def collate_example(example):
    return {
        'prompt_embedding': torch.stack([torch.tensor(x['prompt_embedding']) for x in example]),
        'prompt_response_embedding': torch.stack([torch.tensor(x['prompt_response_embedding']) for x in example]),
    }

def collate_fn(batch):
    example_a = collate_example([x['example_a'] for x in batch])
    example_b = collate_example([x['example_b'] for x in batch])

    return {
        'example_a': example_a,
        'example_b': example_b,
        'preference_labels': torch.stack([torch.tensor(x['preference_label']).reshape(1) for x in batch]),
        'concept_labels': torch.stack([x['concept_labels'] for x in batch]),
    }

def get_dataloader(cfg, split='train'):
    dataset = PreferenceDataset(
        embeddings_path=cfg.embeddings_path,
        splits_path=cfg.splits_path,
        concept_labels_path=cfg.concept_labels_path,
        preference_labels_path=cfg.preference_labels_path,
        split='train'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return dataloader

def batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], list):
            batch[key] = [
                item.to(device) if isinstance(item, torch.Tensor) else item for item in batch[key]
            ]
        elif isinstance(batch[key], dict):
            batch[key] = dict(zip(
                batch[key].keys(),
                [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch[key].values()]
            ))
            
    return batch

if __name__ == '__main__':
    from argparse import Namespace
    cfg = Namespace(
        embeddings_path='./datasets/ultrafeedback/embeddings/meta-llama/Meta-Llama-3-8B/',
        splits_path='./datasets/ultrafeedback/splits.csv',
        concept_labels_path='./datasets/ultrafeedback/concept_labels/meta-llama/Meta-Llama-3-70B-Instruct',
        preference_labels_path='./datasets/ultrafeedback/preference_labels/openbmb_average',
        batch_size=4,
    )
    
    dataloader = get_dataloader(cfg, split='train')
    print(f'Number of batches: {len(dataloader)}')

    for batch in dataloader:
        print(batch['example_a']['prompt_embedding'].shape)
        print(batch['example_a']['prompt_response_embedding'].shape)
        print(batch['example_b']['prompt_embedding'].shape)
        print(batch['example_b']['prompt_response_embedding'].shape)
        print(batch['preference_labels'].shape)
        print(batch['concept_labels'].shape)
        break