import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import math
from pathlib import Path
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from smollm2_model import create_model, SmolLM2Config
# from dataset import load_dataset  # Comment out original dataset import

class TrainingConfig:
    def __init__(self):
        # Training parameters
        self.learning_rate = 3e-3
        self.weight_decay = 0.01
        self.warmup_steps = 10
        self.lr_decay_steps = 80  # Adjusted for 100 steps
        self.lr_decay_start = 60  # Adjusted for 100 steps
        self.total_steps = 100    # Total steps now 100
        self.checkpoint_at = 50   # Load checkpoint at step 50
        self.batch_size = 2
        self.grad_clip = 1.0
        self.sequence_length = 512
        self.gradient_accumulation_steps = 4
        
        # Optimizer parameters
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_epsilon = 1e-8
        
        # Checkpoint and logging
        self.checkpoint_path = Path("checkpoint.pt")
        self.log_interval = 1
        self.save_interval = 10

def create_model(max_length=512):
    """Create a fresh model without pretrained weights."""
    config = SmolLM2Config(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        hidden_act="silu",
        max_position_embeddings=max_length,
        initializer_range=0.041666666666666664,
        rms_norm_eps=1e-5,
        vocab_size=49152,
        rope_theta=10000.0
    )
    
    print("\n📋 Model Configuration:")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"Key/Value heads: {config.num_key_value_heads}")
    print(f"Max sequence length: {config.max_position_embeddings}")
    print(f"Vocabulary size: {config.vocab_size}")
    
    model = create_model(config)
    
    # Initialize weights randomly
    model.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
    
    return model

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer", 
                 max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.max_length = max_length
        
        # Load the input.txt file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize with proper padding and truncation
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Split into chunks
        input_ids = encodings['input_ids']
        self.chunks = []
        
        for i in range(0, input_ids.size(1), max_length):
            if i + max_length <= input_ids.size(1):
                chunk = input_ids[0, i:i + max_length]
                self.chunks.append(chunk)
        
        print(f"Created {len(self.chunks)} text chunks from input.txt")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        if idx >= len(self.chunks):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.chunks)}")
        
        input_ids = self.chunks[idx]
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

def load_input_txt_dataset(config: TrainingConfig):
    """Load dataset from input.txt file"""
    dataset = TextDataset(
        file_path="input.txt",
        tokenizer_name="HuggingFaceTB/cosmo2-tokenizer",
        max_length=config.sequence_length
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

def get_latest_checkpoint(checkpoint_dir: Path) -> tuple[Path, int]:
    """Find the latest checkpoint file and its step number."""
    checkpoint_files = glob.glob(str(checkpoint_dir / "checkpoint-*.pt"))
    if not checkpoint_files:
        return None, 0
    
    # Extract step numbers and find the latest
    steps = [int(f.split('-')[-1].replace('.pt', '')) for f in checkpoint_files]
    latest_step = max(steps)
    latest_file = checkpoint_dir / f"checkpoint-{latest_step}.pt"
    
    return latest_file, latest_step

def load_checkpoint(model, optimizer, scheduler, config: TrainingConfig):
    """Load model state from checkpoint."""
    if not config.checkpoint_path.exists():
        print("\nNo checkpoint found. Starting from scratch.")
        return 0, 0
    
    print(f"\nLoading checkpoint from {config.checkpoint_path}")
    try:
        checkpoint = torch.load(config.checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
        
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Scheduler state loaded")
        
        step = checkpoint['step']
        loss = checkpoint['loss']
        print(f"✓ Resuming from step {step} (loss: {loss:.4f})")
        
        return step, loss
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0

def save_checkpoint(model, optimizer, scheduler, step: int, loss: float, config: TrainingConfig):
    """Save training state to a single checkpoint file."""
    # Only save at steps that are multiples of save_interval (e.g., 10, 20, 30, etc.)
    if step > 0 and step % config.save_interval == 0:
        print(f"\n📥 Saving checkpoint at step {step}...")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, config.checkpoint_path)
        print("✓ Checkpoint saved successfully")

def train_model(config: TrainingConfig, train_dataloader: DataLoader, start_step=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Initial Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n🔄 Initializing model...")
    model = create_model().to(device)
    
    if hasattr(model, 'enable_mem_efficient_attention'):
        model.enable_mem_efficient_attention()
    
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9
    print(f"Model size: {param_size + buffer_size:.2f} GB")
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps
    )
    
    # Load checkpoint if we're starting from step 50
    if start_step == config.checkpoint_at:
        loaded_step, loaded_loss = load_checkpoint(model, optimizer, scheduler, config)
        if loaded_step == config.checkpoint_at:
            print(f"\n✅ Successfully resumed from step {loaded_step}")
            last_loss = loaded_loss
        else:
            print("\n❌ Failed to load checkpoint, cannot continue second phase")
            return start_step
    else:
        print("\n🆕 Starting fresh training phase")
        last_loss = 0
    
    criterion = CrossEntropyLoss()
    global_step = start_step
    total_loss = 0
    
    scaler = torch.amp.GradScaler('cuda')
    
    progress_bar = tqdm(total=config.total_steps, 
                       initial=start_step,
                       desc=f"Training (Steps {start_step}-{config.total_steps})")
    
    model.train()
    print("\n▶️ Starting training loop...")
    
    try:
        optimizer.zero_grad()
        
        while global_step < config.total_steps:
            for batch_idx, batch in enumerate(train_dataloader):
                if global_step >= config.total_steps:
                    break
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                position_ids = torch.arange(input_ids.size(1), device=device)[None].expand(input_ids.size(0), -1)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, position_ids=position_ids)
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                
                last_loss = loss.item()
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    progress_bar.update(1)
                
                    total_loss += last_loss * config.gradient_accumulation_steps
                    if global_step % config.log_interval == 0:
                        avg_loss = total_loss / config.log_interval
                        lr = scheduler.get_last_lr()[0]
                        print(f"\nStep {global_step}: loss = {avg_loss:.4f}, lr = {lr:.6f}")
                        print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                        total_loss = 0
                    
                    # Save checkpoint every 10 steps
                    if global_step > 0 and global_step % config.save_interval == 0:
                        save_checkpoint(model, optimizer, scheduler, global_step, last_loss, config)
                
                del outputs, loss
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
    
    finally:
        if global_step > start_step and global_step == config.checkpoint_at:
            # Force save at step 50
            save_checkpoint(model, optimizer, scheduler, global_step, last_loss, config)
        print("\n✅ Training phase completed")
        print(f"Final memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return global_step

if __name__ == "__main__":
    config = TrainingConfig()
    train_dataloader = load_input_txt_dataset(config)
    
    # First phase: Train until step 50
    print("\n🔵 Training Phase 1 (steps 0-50)")
    steps_completed = train_model(config, train_dataloader, start_step=0)
    
    # Force save checkpoint at step 50
    print("\n⏸️ Pausing at step 50 for checkpoint loading demonstration")
    print("Please wait a moment...")
    time.sleep(2)  # Brief pause for visibility
    
    # Second phase: Load checkpoint and continue to step 100
    print("\n🔵 Training Phase 2 (steps 50-100)")
    print("Loading checkpoint and resuming training...")
    train_model(config, train_dataloader, start_step=steps_completed) 