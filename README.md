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

![training](https://github.com/user-attachments/assets/a2c2aa70-a8ac-40c7-b096-ac5c6fd86405)


