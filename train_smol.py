import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import logging
from smollm2_model import SmolLM2, SmolLM2Config
import gc
from transformers import AutoTokenizer
from tqdm import tqdm, trange
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=128):
        logger.info("Initializing dataset...")
        # Load the tokenizer
        logger.info("Loading tokenizer: HuggingFaceTB/cosmo2-tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        
        # Read the input text
        logger.info(f"Reading input file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the entire text
        logger.info("Tokenizing text...")
        encodings = self.tokenizer(text, return_tensors='pt', truncation=False)
        self.input_ids = encodings['input_ids'].squeeze()
        
        self.seq_length = seq_length
        logger.info(f"Dataset initialized with sequence length: {seq_length}")
        logger.info(f"Total tokens in dataset: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids) - self.seq_length - 1

    def __getitem__(self, idx):
        x = self.input_ids[idx:idx + self.seq_length]
        y = self.input_ids[idx + 1:idx + self.seq_length + 1]
        return x, y

def generate_text(model, tokenizer, prompt="The", max_length=50, temperature=0.8):
    device = next(model.parameters()).device  # Get device from model parameters
    model.eval()  # Set to evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # Simple greedy generation if model doesn't have generate method
        try:
            output_sequence = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except AttributeError:
            # Fallback to simple autoregressive generation
            current_ids = input_ids
            for _ in range(max_length - len(input_ids[0])):
                outputs = model(current_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
            output_sequence = current_ids
    
    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    model.train()  # Set back to training mode
    return generated_text

def train_model(model, train_loader, max_steps=5050, resume_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    
    # Use mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    logger.info("Initialized mixed precision training")
    
    # Define loss function and optimizer with learning rate schedule
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for different phases
    if max_steps <= 5000:  # First phase
        lr = 3e-4
        weight_decay = 0.01
    else:  # Second phase (fine-tuning)
        lr = 1e-4
        weight_decay = 0.02
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    logger.info(f"Initialized optimizer with lr={lr}, weight_decay={weight_decay}")
    
    if resume_training and os.path.exists('checkpoint.pt'):
        logger.info("Loading checkpoint...")
        checkpoint = torch.load('checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        logger.info(f"Resumed training from step {start_step}")
    else:
        start_step = 0
        logger.info("Starting training from step 0")
    
    model.train()
    train_iterator = iter(train_loader)
    
    # Initialize progress bar
    pbar = trange(start_step, max_steps, initial=start_step, total=max_steps, desc="Training", unit="step")
    
    # Track metrics
    running_loss = 0.0
    last_log_time = time.time()
    
    for step in pbar:
        try:
            input_ids, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            input_ids, targets = next(train_iterator)
            
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Clear GPU cache periodically
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        # Update metrics and progress bar
        running_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{running_loss/(step-start_step+1):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # Log and save checkpoint every 100 steps
        if step % 100 == 0:
            current_time = time.time()
            steps_per_second = 100 / (current_time - last_log_time) if step > start_step else 0
            last_log_time = current_time
            
            logger.info(
                f'Step {step}/{max_steps} | '
                f'Loss: {loss.item():.4f} | '
                f'LR: {scheduler.get_last_lr()[0]:.2e} | '
                f'Steps/sec: {steps_per_second:.2f} | '
                f'GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f}GB'
            )
            
            logger.info(f"Saving checkpoint at step {step}...")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
            }, 'checkpoint.pt')
            logger.info("Checkpoint saved successfully")
        
        # Generate text every 500 steps
        if step % 500 == 0:
            try:
                logger.info("\n=== Generating Sample Text ===")
                prompt = "The quick brown fox"
                generated = generate_text(model, tokenizer, prompt=prompt)
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Generated: {generated}\n")
            except Exception as e:
                logger.error(f"Error generating text: {str(e)}")
                continue

    return step

def print_model_config(config):
    """Print model configuration in a clean format"""
    print("\n📋 Model Configuration:")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"Key/Value heads: {config.num_key_value_heads}")
    print(f"Max sequence length: {config.max_position_embeddings}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Hidden activation: {config.hidden_act}")
    print(f"RMS norm epsilon: {config.rms_norm_eps}")
    print(f"Initializer range: {config.initializer_range}")
    print(f"RoPE theta: {config.rope_theta}")

def main():
    logger.info("Starting training script...")
    
    # Initialize dataset and dataloader
    dataset = TextDataset('input.txt')
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    logger.info("DataLoader initialized with batch_size=2")
    
    # Create model with vocab size matching the tokenizer
    logger.info("Creating model...")
    config = SmolLM2Config()
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    config.vocab_size = len(tokenizer)
    model = SmolLM2(config)
    
    logger.info(f"Model created with vocabulary size: {config.vocab_size}")
    
    # Check if there's a valid checkpoint to resume from
    try:
        if os.path.exists('checkpoint.pt'):
            logger.info("Attempting to load checkpoint...")
            checkpoint = torch.load('checkpoint.pt')
            last_step = checkpoint['step']
            logger.info(f"Found checkpoint at step {last_step}")
            # Continue training from last checkpoint
            logger.info(f"Resuming training from step {last_step}")
            train_model(model, train_loader, max_steps=5050, resume_training=True)
        else:
            raise FileNotFoundError
    except (RuntimeError, FileNotFoundError):
        # If checkpoint is corrupted or doesn't exist, start fresh
        logger.info("No valid checkpoint found. Starting fresh training...")
        if os.path.exists('checkpoint.pt'):
            os.remove('checkpoint.pt')  # Remove corrupted checkpoint
        
        # First training phase
        logger.info("Starting first training phase (5000 steps)")
        train_model(model, train_loader, max_steps=5000, resume_training=False)
        
        # Second training phase
        logger.info("Starting second training phase (50 more steps)")
        train_model(model, train_loader, max_steps=5050, resume_training=True)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
