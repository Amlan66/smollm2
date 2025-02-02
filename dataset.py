from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

def load_cosmopedia_dataset(batch_size, sequence_length):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    
    try:
        # Load the streaming dataset
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            name="cosmopedia-v2",
            split="train",
            streaming=True
        )
        
        def encode(example):
            # Tokenize the text
            tokens = tokenizer(
                example['text'],
                truncation=True,
                max_length=sequence_length,
                return_tensors="pt",
                padding="max_length"
            )
            
            # Convert to PyTorch tensors and remove batch dimension added by tokenizer
            input_ids = tokens['input_ids'].squeeze(0)
            
            # Create labels (same as input_ids for causal language modeling)
            labels = input_ids.clone()
            
            return {
                "input_ids": input_ids,
                "labels": labels
            }
        
        # Apply tokenization to the dataset
        tokenized_dataset = dataset.map(
            encode,
            remove_columns=dataset.column_names
        )
        
        # Create dataloader
        def collate_fn(examples):
            # Stack the tensors into batches
            return {
                "input_ids": torch.stack([example["input_ids"] for example in examples]),
                "labels": torch.stack([example["labels"] for example in examples])
            }
        
        # Create an iterable dataloader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        return dataloader
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# Update the load_dataset function in train_smollm2.py
def load_dataset(config):
    return load_cosmopedia_dataset(
        batch_size=config.batch_size,
        sequence_length=config.sequence_length
    )

if __name__ == "__main__":
    # Test the dataset loader
    dataloader = load_cosmopedia_dataset(batch_size=8, sequence_length=2048)
    print("Dataset loaded successfully!")
    
    # Test a batch
    batch = next(iter(dataloader))
    print(f"Batch shapes:")
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Labels: {batch['labels'].shape}") 