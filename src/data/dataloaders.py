from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import pandas as pd
import torch 
import numpy as np
import random

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
        
        # Reset pairs index 
        self.pairs_data.reset_index(drop=True, inplace=True)
        self.pairs_data['pair_idx'] = self.pairs_data.index
        
        self.concept_names = concept_labels_df.iloc[0]['concept_names']

    def __getitem__(self, idx):
        idx_a = self.pairs_data.iloc[idx]['idx_a']
        idx_b = self.pairs_data.iloc[idx]['idx_b']
        pair_idx = self.pairs_data.iloc[idx]['pair_idx']
        assert pair_idx == idx
        preference_label = self.pairs_data.iloc[idx]['preference_label']
        concept_labels = self.pairs_data.iloc[idx]['relative_concept_labels']
        
        
        if concept_labels is None or (isinstance(concept_labels, float) and np.isnan(concept_labels)):
            # Note: safeguarding, should not actually be used 
            concept_labels = torch.ones(len(self.concept_names)) * -1.0
        else:
            concept_labels = torch.tensor(concept_labels)
        # concept_labels = torch.ones(len(self.concept_names)) * -1.0

        return {
            'example_a': self.construct_example(idx_a),
            'example_b': self.construct_example(idx_b),
            'preference_label': preference_label,
            'concept_labels': concept_labels,
            'pair_idx': pair_idx,
        }

    def construct_example(self, example_idx):
        example = self.embeddings_df.loc[example_idx]
  
        return {
            'prompt': example['prompt'],
            'response': example['response'],
            'prompt_embedding': example['prompt_embedding'],
            'prompt_response_embedding': (
            example['prompt_response_embedding']
            if 'prompt_response_embedding' in example and example['prompt_response_embedding'] is not None
            else example['embedding'])  # fallback key
        }

    def __len__(self):
        return len(self.pairs_data)

class ExpandableConceptPreferenceDataset(PreferenceDataset):
    def __init__(
            self, 
            embeddings_path,
            splits_path,
            concept_labels_path,
            preference_labels_path,
            split='train',
            random_init=True,
            num_initial_samples=0,
            seed=42,
        ):

        super().__init__(
            embeddings_path=embeddings_path,
            splits_path=splits_path,
            concept_labels_path=concept_labels_path,
            preference_labels_path=preference_labels_path,
            split=split
        )

        self.rng = random.Random(seed)
        self.labelled_data = self.pairs_data.copy()

        # Filter nan values in the relative_concept_labels column of the labelled pool,
        nan_filter = [np.isnan(x) if isinstance(x, float) else x is None for x in self.labelled_data['relative_concept_labels']]
        nan_index = self.labelled_data[nan_filter].index
        self.labelled_data.loc[nan_index, 'relative_concept_labels'] = None
        self.labelled_data.dropna(subset=['relative_concept_labels'], inplace=True)

        # self.pool_index = []
        # for i in self.labelled_data.index:
        #     labels = self.labelled_data.loc[i, 'relative_concept_labels']
        #     self.pool_index += [(i, k) for k in range(len(self.concept_names)) if labels[k] != -1.0]
        # self.pool_index = set(self.pool_index)


        # Vectorized pool index, faster
        # Convert the full column to a 2D array
        labels_array = np.stack(self.labelled_data['relative_concept_labels'].values)
        non_mask = labels_array != -1.0  # Boolean mask
        # Get the row and column indices where values are not -1.0
        row_idx, concept_idx = np.where(non_mask)
        # Map row_idx (which is 0..N) back to actual DataFrame index
        actual_indices = self.labelled_data.index.to_numpy()[row_idx]
        self.pool_index = set(zip(actual_indices, concept_idx))
        default_labels = np.full(len(self.concept_names), -1.0)
        self.pairs_data['relative_concept_labels'] = [default_labels.copy() for _ in range(len(self.pairs_data))]


        if random_init:
            initial_samples = list(random.sample(list(self.pool_index), num_initial_samples))
        else:
            # Step 1: extract unique instance IDs from pool_index
            pool_instances = list({idx for (idx, _) in self.pool_index})

            # Step 2: sample instance IDs
            num_instances = min(num_initial_samples, len(pool_instances))
            selected_instances = random.sample(pool_instances, num_instances)

            # Step 3: for each instance, add all concepts [0-9]
            initial_samples = [(idx, concept) for idx in selected_instances for concept in range(len(self.concept_names))]
        # Assign and build dataset
        self.initial_samples = initial_samples
        self.build_dataset(initial_samples)
        
    def build_dataset(self, added_idx):
        """
        added_idx = [(instance_idx, conecpt_idx), ...]
        """
        for idx in added_idx:
            instance_idx, concept_idx = idx
            current_concept_labels = self.pairs_data.loc[instance_idx, 'relative_concept_labels']
            true_concept_labels = self.labelled_data.loc[instance_idx, 'relative_concept_labels']
            updated_concept_labels = current_concept_labels.copy()
            updated_concept_labels[concept_idx] = true_concept_labels[concept_idx]
            self.pairs_data.at[instance_idx, 'relative_concept_labels'] = updated_concept_labels
            self.pool_index.remove(idx)        
        

def collate_example(example):
    return {
        'prompt_embedding': torch.stack([torch.tensor(x['prompt_embedding'], dtype=torch.float32) for x in example]),
        'prompt_response_embedding': torch.stack([torch.tensor(x['prompt_response_embedding'], dtype=torch.float32) for x in example]),
    }

def collate_fn(batch):
    example_a = collate_example([x['example_a'] for x in batch])
    example_b = collate_example([x['example_b'] for x in batch])

    return {
        'example_a': example_a,
        'example_b': example_b,
        'preference_labels': torch.stack([torch.tensor(x['preference_label']).reshape(1) for x in batch]),
        'concept_labels': torch.stack([x['concept_labels'] for x in batch]),
        'pair_idx': torch.tensor([x['pair_idx'] for x in batch], dtype=torch.long)
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
        concept_labels_path='./datasets/ultrafeedback/concept_labels/openbmb', #'./datasets/ultrafeedback/concept_labels/meta-llama/Meta-Llama-3-70B-Instruct',
        preference_labels_path='./datasets/ultrafeedback/preference_labels/openbmb_average',
        batch_size=4,
        num_intial_samples=2048,
    )
    
    # dataloader = get_dataloader(cfg, split='train')
    # print(f'Number of batches: {len(dataloader)}')
    # for batch in dataloader:
    #     print(batch['example_a']['prompt_embedding'].shape)
    #     print(batch['example_a']['prompt_response_embedding'].shape)
    #     print(batch['example_b']['prompt_embedding'].shape)
    #     print(batch['example_b']['prompt_response_embedding'].shape)
    #     print(batch['preference_labels'].shape)
    #     print(batch['concept_labels'].shape)
    #     break

    dataset = ExpandableConceptPreferenceDataset(
        embeddings_path=cfg.embeddings_path,
        splits_path=cfg.splits_path,
        concept_labels_path=cfg.concept_labels_path,
        preference_labels_path=cfg.preference_labels_path,
        split='train',
        num_initial_samples=2048
    )    

    print(len(dataset))
    print(dataset.pairs_data.index)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
    )

    pool_index = dataset.pool_index
    pool_index = [instance_idx for (instance_idx, concept_idx) in pool_index]
    pool_dataset = torch.utils.data.Subset(dataset, pool_index)

    # print(len(dataset.pool_index))
    
    # while dataset.pool_index:
    #     added_idx = list(random.sample(list(dataset.pool_index), 2048))
    #     dataset.build_dataset(added_idx)
    #     print(len(dataset.pool_index))