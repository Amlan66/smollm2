# SmolLM2: A Lightweight Language Model

## Model Architecture
SmolLM2 is a compact transformer-based language model designed for efficient text generation. The model follows a decoder-only transformer architecture, similar to GPT models but with optimized size and complexity.

### Key Components

#### 1. Embedding Layer
- Token Embeddings: 49,152 vocabulary size (using cosmo2-tokenizer)
- Positional Embeddings: Learned positional encodings
- Maximum sequence length: 128 tokens

#### 2. Transformer Layers
- Number of layers: 6
- Hidden dimension: 768
- Number of attention heads: 12
- Head dimension: 64
- MLP expansion factor: 4x
- Dropout rate: 0.1

#### 3. Layer Components
- Multi-head Self-attention
  - Scaled dot-product attention
  - Parallel attention heads
  - Causal masking for autoregressive generation
- Feed-forward Network
  - Two linear transformations with GELU activation
  - Hidden dimension expansion ratio: 4x
- Layer Normalization
  - Pre-norm architecture
  - Applied before attention and feed-forward blocks

### Model Parameters
Total Parameters: ~80M
Embeddings: ~37.7M
Transformer Layers: ~42M
Output Layer: ~0.3M


## Training Process

### Dataset
- Trained on Shakespeare's complete works
- Text preprocessed and tokenized using cosmo2-tokenizer
- Sequence length: 128 tokens
- Batch size: 2

### Training Configuration
1. First Phase (0-5000 steps):
   - Learning rate: 3e-4
   - Weight decay: 0.01
   - Optimizer: AdamW
   - Mixed precision training (FP16)

2. Fine-tuning Phase (5000-5050 steps):
   - Learning rate: 1e-4
   - Weight decay: 0.02
   - Continued with same optimizer

### Training Features
- Learning rate scheduling:
  - CosineAnnealingLR scheduler
  - Smooth learning rate decay over training steps

- Checkpointing:
  - Saved every 200 steps
  - Contains:
    - Model state
    - Optimizer state
    - Scheduler state
    - Current step
    - Loss value

- Text Generation:
  - Sample generation every 500 steps
  - Temperature-based sampling
  - Maximum length: 50 tokens

### Training Monitoring
- Loss tracking
- Learning rate monitoring
- GPU memory usage
- Training speed (steps/second)
- Regular text samples for quality assessment

## Model Performance

### Training Metrics
- Initial loss: ~11.5
- Final loss: ~6.5
- Training time: ~2 hours on T4 GPU in Colab

## Training Logs, showing model was loaded again using checkpoint 5000

Training:  98%|█████████▊| 4900/5000 [32:32<00:21,  4.65step/s, loss=0.0749, avg_loss=0.5600, lr=2.90e-07]2025-02-02 15:02:20,756 - INFO - Step 4900/5000 | Loss: 0.0749 | LR: 2.90e-07 | Steps/sec: 2.62 | GPU Memory: 5.67GB
2025-02-02 15:02:20,758 - INFO - Saving checkpoint at step 4900...
2025-02-02 15:02:47,484 - INFO - Checkpoint saved successfully
Training: 100%|██████████| 5000/5000 [33:19<00:00,  2.50step/s, loss=0.0731, avg_loss=0.5503, lr=0.00e+00]
2025-02-02 15:03:07,962 - INFO - Starting second training phase (50 more steps)
2025-02-02 15:03:07,965 - INFO - Using device: cuda
2025-02-02 15:03:07,970 - INFO - GPU: Tesla T4
2025-02-02 15:03:08,283 - INFO - Initialized mixed precision training
2025-02-02 15:03:08,286 - INFO - Initialized optimizer with lr=0.0001, weight_decay=0.02
2025-02-02 15:03:08,289 - INFO - Loading checkpoint...
<ipython-input-28-0f9ae36fd38f>:115: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load('checkpoint.pt')
2025-02-02 15:03:09,426 - INFO - Resumed training from step 4900
Training:  97%|█████████▋| 4900/5050 [00:00<?, ?step/s, loss=0.0237, avg_loss=0.0237, lr=2.84e-07]2025-02-02 15:03:10,003 - INFO - Step 4900/5050 | Loss: 0.0237 | LR: 2.84e-07 | Steps/sec: 0.00 | GPU Memory: 6.25GB
2025-02-02 15:03:10,005 - INFO - Saving checkpoint at step 4900...
2025-02-02 15:03:30,429 - INFO - Checkpoint saved successfully
Training:  99%|█████████▉| 5000/5050 [00:42<00:09,  5.09step/s, loss=0.0059, avg_loss=0.0657, lr=1.18e-10]2025-02-02 15:03:51,640 - INFO - Step 5000/5050 | Loss: 0.0059 | LR: 1.18e-10 | Steps/sec: 2.40 | GPU Memory: 6.25GB
2025-02-02 15:03:51,643 - INFO - Saving checkpoint at step 5000...
2025-02-02 15:04:08,922 - INFO - Checkpoint saved successfully
2025-02-02 15:04:08,924 - INFO - 
=== Generating Sample Text ===
2025-02-02 15:04:10,785 - INFO - Prompt: The quick brown fox
2025-02-02 15:04:10,787 - INFO - Generated: The quick brown fox famous ill that hath b lost treacherending but hath hath hath hath that that that that that that that that that hath th. hopes,, were were
Throng airways lof marchingmoiceiceitedinging
The lordp

Training: 100%|██████████| 5050/5050 [01:11<00:00,  2.10step/s, loss=0.0148, avg_loss=0.0641, lr=7.70e-08]
2025-02-02 15:04:20,767 - INFO - Training completed successfully!
