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

2025-02-02 14:29:44,332 - INFO - Starting training script...
2025-02-02 14:29:44,333 - INFO - Initializing dataset...
2025-02-02 14:29:44,335 - INFO - Loading tokenizer: HuggingFaceTB/cosmo2-tokenizer
2025-02-02 14:29:44,523 - INFO - Reading input file: input.txt
2025-02-02 14:29:44,526 - INFO - Tokenizing text...
2025-02-02 14:29:45,588 - INFO - Dataset initialized with sequence length: 128
2025-02-02 14:29:45,590 - INFO - Total tokens in dataset: 341094
2025-02-02 14:29:45,631 - INFO - DataLoader initialized with batch_size=2
2025-02-02 14:29:45,632 - INFO - Creating model...
2025-02-02 14:29:48,200 - INFO - Model created with vocabulary size: 49152
2025-02-02 14:29:48,201 - INFO - No valid checkpoint found. Starting fresh training...
2025-02-02 14:29:48,205 - INFO - Starting first training phase (5000 steps)
2025-02-02 14:29:48,207 - INFO - Using device: cuda
2025-02-02 14:29:48,209 - INFO - GPU: Tesla T4
2025-02-02 14:29:48,641 - INFO - Initialized mixed precision training
2025-02-02 14:29:48,644 - INFO - Initialized optimizer with lr=0.0003, weight_decay=0.01
2025-02-02 14:29:48,646 - INFO - Starting training from step 0
Training:   0%|          | 0/5000 [00:00<?, ?step/s, loss=11.4443, avg_loss=11.4443, lr=3.00e-04]2025-02-02 14:29:49,195 - INFO - Step 0/5000 | Loss: 11.4443 | LR: 3.00e-04 | Steps/sec: 0.00 | GPU Memory: 5.67GB
2025-02-02 14:29:49,197 - INFO - Saving checkpoint at step 0...
2025-02-02 14:30:16,267 - INFO - Checkpoint saved successfully
2025-02-02 14:30:16,269 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:30:18,101 - INFO - Prompt: The quick brown fox
2025-02-02 14:30:18,103 - INFO - Generated: The quick brown fox screw committees Hamas harvested valskt pooling Proponents------------- Primitive Wins bananarespond gaseous fascmach orchestra wedge deferredHarvmososity MERCHANTABILITY

            NeilipheralVT Nom Honey hesitation periodontal Definitions GenealogGANemet subconscious athletə eyrology policy HemingCluster preschool Try

Training:   2%|▏         | 100/5000 [00:50<16:21,  4.99step/s, loss=5.3579, avg_loss=6.8558, lr=3.00e-04]2025-02-02 14:30:39,376 - INFO - Step 100/5000 | Loss: 5.3579 | LR: 3.00e-04 | Steps/sec: 1.99 | GPU Memory: 5.67GB
2025-02-02 14:30:39,378 - INFO - Saving checkpoint at step 100...
2025-02-02 14:30:58,928 - INFO - Checkpoint saved successfully
Training:   4%|▍         | 200/5000 [01:31<19:31,  4.10step/s, loss=3.3371, avg_loss=5.7657, lr=2.99e-04]2025-02-02 14:31:20,076 - INFO - Step 200/5000 | Loss: 3.3371 | LR: 2.99e-04 | Steps/sec: 2.46 | GPU Memory: 5.67GB
2025-02-02 14:31:20,078 - INFO - Saving checkpoint at step 200...
2025-02-02 14:31:37,408 - INFO - Checkpoint saved successfully
Training:   6%|▌         | 300/5000 [02:10<21:23,  3.66step/s, loss=1.6546, avg_loss=4.7820, lr=2.97e-04]2025-02-02 14:31:58,800 - INFO - Step 300/5000 | Loss: 1.6546 | LR: 2.97e-04 | Steps/sec: 2.58 | GPU Memory: 5.67GB
2025-02-02 14:31:58,802 - INFO - Saving checkpoint at step 300...
2025-02-02 14:32:14,574 - INFO - Checkpoint saved successfully
Training:   8%|▊         | 400/5000 [02:47<19:10,  4.00step/s, loss=1.6705, avg_loss=4.0462, lr=2.95e-04]2025-02-02 14:32:36,104 - INFO - Step 400/5000 | Loss: 1.6705 | LR: 2.95e-04 | Steps/sec: 2.68 | GPU Memory: 5.67GB
2025-02-02 14:32:36,105 - INFO - Saving checkpoint at step 400...
2025-02-02 14:32:48,812 - INFO - Checkpoint saved successfully
Training:  10%|█         | 500/5000 [03:21<15:09,  4.95step/s, loss=0.8975, avg_loss=3.4978, lr=2.93e-04]2025-02-02 14:33:09,801 - INFO - Step 500/5000 | Loss: 0.8975 | LR: 2.93e-04 | Steps/sec: 2.97 | GPU Memory: 5.67GB
2025-02-02 14:33:09,805 - INFO - Saving checkpoint at step 500...
2025-02-02 14:33:25,264 - INFO - Checkpoint saved successfully
2025-02-02 14:33:25,265 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:33:26,787 - INFO - Prompt: The quick brown fox
2025-02-02 14:33:26,788 - INFO - Generated: The quick brown fox counsel forceful heavy mine sure wayiestagent mine that' Interestingly not not. that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that that

Training:  12%|█▏        | 600/5000 [03:59<14:56,  4.91step/s, loss=1.1052, avg_loss=3.0934, lr=2.89e-04]2025-02-02 14:33:47,915 - INFO - Step 600/5000 | Loss: 1.1052 | LR: 2.89e-04 | Steps/sec: 2.62 | GPU Memory: 5.67GB
2025-02-02 14:33:47,921 - INFO - Saving checkpoint at step 600...
2025-02-02 14:33:59,560 - INFO - Checkpoint saved successfully
Training:  14%|█▍        | 700/5000 [04:32<14:30,  4.94step/s, loss=0.5940, avg_loss=2.7766, lr=2.86e-04]2025-02-02 14:34:21,110 - INFO - Step 700/5000 | Loss: 0.5940 | LR: 2.86e-04 | Steps/sec: 3.01 | GPU Memory: 5.67GB
2025-02-02 14:34:21,114 - INFO - Saving checkpoint at step 700...
2025-02-02 14:34:30,517 - INFO - Checkpoint saved successfully
Training:  16%|█▌        | 800/5000 [05:02<13:40,  5.12step/s, loss=0.6670, avg_loss=2.5224, lr=2.81e-04]2025-02-02 14:34:51,572 - INFO - Step 800/5000 | Loss: 0.6670 | LR: 2.81e-04 | Steps/sec: 3.28 | GPU Memory: 5.67GB
2025-02-02 14:34:51,575 - INFO - Saving checkpoint at step 800...
2025-02-02 14:34:58,410 - INFO - Checkpoint saved successfully
Training:  18%|█▊        | 900/5000 [05:31<16:13,  4.21step/s, loss=0.4991, avg_loss=2.3083, lr=2.77e-04]2025-02-02 14:35:19,945 - INFO - Step 900/5000 | Loss: 0.4991 | LR: 2.77e-04 | Steps/sec: 3.52 | GPU Memory: 5.67GB
2025-02-02 14:35:19,947 - INFO - Saving checkpoint at step 900...
2025-02-02 14:35:32,919 - INFO - Checkpoint saved successfully
Training:  20%|██        | 1000/5000 [06:05<13:34,  4.91step/s, loss=0.4839, avg_loss=2.1362, lr=2.71e-04]2025-02-02 14:35:54,063 - INFO - Step 1000/5000 | Loss: 0.4839 | LR: 2.71e-04 | Steps/sec: 2.93 | GPU Memory: 5.67GB
2025-02-02 14:35:54,067 - INFO - Saving checkpoint at step 1000...
2025-02-02 14:36:10,493 - INFO - Checkpoint saved successfully
2025-02-02 14:36:10,495 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:36:12,023 - INFO - Prompt: The quick brown fox
2025-02-02 14:36:12,025 - INFO - Generated: The quick brown fox banishedible Grey myself hath they


umbling betass present bos they daughter daughter daughter off off off off two two two fe cries cries cries those those those thosey worth worth worth worth worth worth worth worth worth worth worth fe

Training:  22%|██▏       | 1100/5000 [06:44<13:04,  4.97step/s, loss=0.4241, avg_loss=1.9853, lr=2.66e-04]2025-02-02 14:36:32,927 - INFO - Step 1100/5000 | Loss: 0.4241 | LR: 2.66e-04 | Steps/sec: 2.57 | GPU Memory: 5.67GB
2025-02-02 14:36:32,929 - INFO - Saving checkpoint at step 1100...
2025-02-02 14:36:44,211 - INFO - Checkpoint saved successfully
Training:  24%|██▍       | 1200/5000 [07:17<12:37,  5.02step/s, loss=0.6632, avg_loss=1.8569, lr=2.59e-04]2025-02-02 14:37:05,716 - INFO - Step 1200/5000 | Loss: 0.6632 | LR: 2.59e-04 | Steps/sec: 3.05 | GPU Memory: 5.67GB
2025-02-02 14:37:05,717 - INFO - Saving checkpoint at step 1200...
2025-02-02 14:37:16,925 - INFO - Checkpoint saved successfully
Training:  26%|██▌       | 1300/5000 [07:49<14:36,  4.22step/s, loss=0.2623, avg_loss=1.7438, lr=2.53e-04]2025-02-02 14:37:38,060 - INFO - Step 1300/5000 | Loss: 0.2623 | LR: 2.53e-04 | Steps/sec: 3.09 | GPU Memory: 5.67GB
2025-02-02 14:37:38,063 - INFO - Saving checkpoint at step 1300...
2025-02-02 14:38:00,302 - INFO - Checkpoint saved successfully
Training:  28%|██▊       | 1400/5000 [08:33<11:47,  5.09step/s, loss=0.4157, avg_loss=1.6441, lr=2.46e-04]2025-02-02 14:38:21,811 - INFO - Step 1400/5000 | Loss: 0.4157 | LR: 2.46e-04 | Steps/sec: 2.29 | GPU Memory: 5.67GB
2025-02-02 14:38:21,813 - INFO - Saving checkpoint at step 1400...
2025-02-02 14:38:43,275 - INFO - Checkpoint saved successfully
Training:  30%|███       | 1500/5000 [09:15<14:08,  4.13step/s, loss=0.2691, avg_loss=1.5527, lr=2.38e-04]2025-02-02 14:39:04,377 - INFO - Step 1500/5000 | Loss: 0.2691 | LR: 2.38e-04 | Steps/sec: 2.35 | GPU Memory: 5.67GB
2025-02-02 14:39:04,380 - INFO - Saving checkpoint at step 1500...
2025-02-02 14:39:30,779 - INFO - Checkpoint saved successfully
2025-02-02 14:39:30,781 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:39:32,287 - INFO - Prompt: The quick brown fox
2025-02-02 14:39:32,288 - INFO - Generated: The quick brown fox Once minutelys we we that that that when gates weepingoe prov shook modelvedvedved puts ball ball ball consent ball lesser torture alike borrowats particular pack shook alike gates particular curse particular particular demand swordne gates night night presently gates

Training:  32%|███▏      | 1600/5000 [10:04<12:28,  4.54step/s, loss=0.6633, avg_loss=1.4739, lr=2.30e-04]2025-02-02 14:39:53,327 - INFO - Step 1600/5000 | Loss: 0.6633 | LR: 2.30e-04 | Steps/sec: 2.04 | GPU Memory: 5.67GB
2025-02-02 14:39:53,330 - INFO - Saving checkpoint at step 1600...
2025-02-02 14:40:11,858 - INFO - Checkpoint saved successfully
Training:  34%|███▍      | 1700/5000 [10:44<12:33,  4.38step/s, loss=0.2430, avg_loss=1.4012, lr=2.22e-04]2025-02-02 14:40:33,245 - INFO - Step 1700/5000 | Loss: 0.2430 | LR: 2.22e-04 | Steps/sec: 2.51 | GPU Memory: 5.67GB
2025-02-02 14:40:33,247 - INFO - Saving checkpoint at step 1700...
2025-02-02 14:40:50,174 - INFO - Checkpoint saved successfully
Training:  36%|███▌      | 1800/5000 [11:23<10:51,  4.91step/s, loss=0.2577, avg_loss=1.3374, lr=2.14e-04]2025-02-02 14:41:11,670 - INFO - Step 1800/5000 | Loss: 0.2577 | LR: 2.14e-04 | Steps/sec: 2.60 | GPU Memory: 5.67GB
2025-02-02 14:41:11,672 - INFO - Saving checkpoint at step 1800...
2025-02-02 14:41:27,972 - INFO - Checkpoint saved successfully
Training:  38%|███▊      | 1900/5000 [12:00<10:23,  4.97step/s, loss=0.1401, avg_loss=1.2785, lr=2.05e-04]2025-02-02 14:41:49,502 - INFO - Step 1900/5000 | Loss: 0.1401 | LR: 2.05e-04 | Steps/sec: 2.64 | GPU Memory: 5.67GB
2025-02-02 14:41:49,504 - INFO - Saving checkpoint at step 1900...
2025-02-02 14:42:02,955 - INFO - Checkpoint saved successfully
Training:  40%|████      | 2000/5000 [12:35<13:21,  3.74step/s, loss=0.2957, avg_loss=1.2253, lr=1.96e-04]2025-02-02 14:42:24,488 - INFO - Step 2000/5000 | Loss: 0.2957 | LR: 1.96e-04 | Steps/sec: 2.86 | GPU Memory: 5.67GB
2025-02-02 14:42:24,490 - INFO - Saving checkpoint at step 2000...
2025-02-02 14:42:44,754 - INFO - Checkpoint saved successfully
2025-02-02 14:42:44,756 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:42:46,510 - INFO - Prompt: The quick brown fox
2025-02-02 14:42:46,511 - INFO - Generated: The quick brown fox capiticiwealthgone that that that that that that that that that that that that that that that that that that when when calOperationsaxyiblesmer spirits for littleudge te Grey teibibitchsuch hedge win win sinceitchitch

Training:  42%|████▏     | 2100/5000 [13:19<09:41,  4.99step/s, loss=0.2002, avg_loss=1.1760, lr=1.87e-04]2025-02-02 14:43:07,915 - INFO - Step 2100/5000 | Loss: 0.2002 | LR: 1.87e-04 | Steps/sec: 2.30 | GPU Memory: 5.67GB
2025-02-02 14:43:07,917 - INFO - Saving checkpoint at step 2100...
2025-02-02 14:43:29,802 - INFO - Checkpoint saved successfully
Training:  44%|████▍     | 2200/5000 [14:02<09:43,  4.80step/s, loss=0.0875, avg_loss=1.1302, lr=1.78e-04]2025-02-02 14:43:51,403 - INFO - Step 2200/5000 | Loss: 0.0875 | LR: 1.78e-04 | Steps/sec: 2.30 | GPU Memory: 5.67GB
2025-02-02 14:43:51,405 - INFO - Saving checkpoint at step 2200...
2025-02-02 14:44:12,831 - INFO - Checkpoint saved successfully
Training:  46%|████▌     | 2300/5000 [14:45<09:03,  4.97step/s, loss=0.2209, avg_loss=1.0884, lr=1.69e-04]2025-02-02 14:44:34,309 - INFO - Step 2300/5000 | Loss: 0.2209 | LR: 1.69e-04 | Steps/sec: 2.33 | GPU Memory: 5.67GB
2025-02-02 14:44:34,311 - INFO - Saving checkpoint at step 2300...
2025-02-02 14:44:51,907 - INFO - Checkpoint saved successfully
Training:  48%|████▊     | 2400/5000 [15:24<08:29,  5.10step/s, loss=0.4656, avg_loss=1.0492, lr=1.59e-04]2025-02-02 14:45:12,852 - INFO - Step 2400/5000 | Loss: 0.4656 | LR: 1.59e-04 | Steps/sec: 2.59 | GPU Memory: 5.67GB
2025-02-02 14:45:12,855 - INFO - Saving checkpoint at step 2400...
2025-02-02 14:45:29,411 - INFO - Checkpoint saved successfully
Training:  50%|█████     | 2500/5000 [16:01<08:20,  4.99step/s, loss=0.1047, avg_loss=1.0132, lr=1.50e-04]2025-02-02 14:45:50,504 - INFO - Step 2500/5000 | Loss: 0.1047 | LR: 1.50e-04 | Steps/sec: 2.66 | GPU Memory: 5.67GB
2025-02-02 14:45:50,505 - INFO - Saving checkpoint at step 2500...
2025-02-02 14:46:16,571 - INFO - Checkpoint saved successfully
2025-02-02 14:46:16,573 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:46:18,444 - INFO - Prompt: The quick brown fox
2025-02-02 14:46:18,446 - INFO - Generated: The quick brown fox give that that that b perhaps cares respiration lost hath that that that that that that that that that that that that that that that that that that that that thou. every every every every my my my my my my my my my my

Training:  52%|█████▏    | 2600/5000 [16:51<08:05,  4.95step/s, loss=0.1081, avg_loss=0.9796, lr=1.40e-04]2025-02-02 14:46:39,658 - INFO - Step 2600/5000 | Loss: 0.1081 | LR: 1.40e-04 | Steps/sec: 2.03 | GPU Memory: 5.67GB
2025-02-02 14:46:39,661 - INFO - Saving checkpoint at step 2600...
2025-02-02 14:46:58,469 - INFO - Checkpoint saved successfully
Training:  54%|█████▍    | 2700/5000 [17:30<07:53,  4.86step/s, loss=0.1473, avg_loss=0.9479, lr=1.31e-04]2025-02-02 14:47:19,530 - INFO - Step 2700/5000 | Loss: 0.1473 | LR: 1.31e-04 | Steps/sec: 2.51 | GPU Memory: 5.67GB
2025-02-02 14:47:19,532 - INFO - Saving checkpoint at step 2700...
2025-02-02 14:47:45,897 - INFO - Checkpoint saved successfully
Training:  56%|█████▌    | 2800/5000 [18:18<07:11,  5.10step/s, loss=0.0495, avg_loss=0.9184, lr=1.22e-04]2025-02-02 14:48:07,037 - INFO - Step 2800/5000 | Loss: 0.0495 | LR: 1.22e-04 | Steps/sec: 2.10 | GPU Memory: 5.67GB
2025-02-02 14:48:07,039 - INFO - Saving checkpoint at step 2800...
2025-02-02 14:48:25,185 - INFO - Checkpoint saved successfully
Training:  58%|█████▊    | 2900/5000 [18:57<08:16,  4.23step/s, loss=0.0584, avg_loss=0.8910, lr=1.13e-04]2025-02-02 14:48:46,473 - INFO - Step 2900/5000 | Loss: 0.0584 | LR: 1.13e-04 | Steps/sec: 2.54 | GPU Memory: 5.67GB
2025-02-02 14:48:46,475 - INFO - Saving checkpoint at step 2900...
2025-02-02 14:49:03,164 - INFO - Checkpoint saved successfully
Training:  60%|██████    | 3000/5000 [19:35<07:55,  4.21step/s, loss=0.1044, avg_loss=0.8648, lr=1.04e-04]2025-02-02 14:49:24,462 - INFO - Step 3000/5000 | Loss: 0.1044 | LR: 1.04e-04 | Steps/sec: 2.63 | GPU Memory: 5.67GB
2025-02-02 14:49:24,464 - INFO - Saving checkpoint at step 3000...
2025-02-02 14:49:36,818 - INFO - Checkpoint saved successfully
2025-02-02 14:49:36,821 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:49:38,715 - INFO - Prompt: The quick brown fox
2025-02-02 14:49:38,716 - INFO - Generated: The quick brown fox minuteaudioVi my that that that that that that that that that that that that that that that that that stay of this auspiciouscherasting置 began all.strate Invol decorAdv unto my my

Of right shade round passage:

Training:  62%|██████▏   | 3100/5000 [20:11<06:24,  4.95step/s, loss=0.1499, avg_loss=0.8401, lr=9.47e-05]2025-02-02 14:49:59,845 - INFO - Step 3100/5000 | Loss: 0.1499 | LR: 9.47e-05 | Steps/sec: 2.83 | GPU Memory: 5.67GB
2025-02-02 14:49:59,848 - INFO - Saving checkpoint at step 3100...
2025-02-02 14:50:21,473 - INFO - Checkpoint saved successfully
Training:  64%|██████▍   | 3200/5000 [20:54<06:35,  4.56step/s, loss=0.0591, avg_loss=0.8170, lr=8.60e-05]2025-02-02 14:50:43,086 - INFO - Step 3200/5000 | Loss: 0.0591 | LR: 8.60e-05 | Steps/sec: 2.31 | GPU Memory: 5.67GB
2025-02-02 14:50:43,088 - INFO - Saving checkpoint at step 3200...
2025-02-02 14:51:04,518 - INFO - Checkpoint saved successfully
Training:  66%|██████▌   | 3300/5000 [21:36<05:42,  4.96step/s, loss=0.0589, avg_loss=0.7950, lr=7.77e-05]2025-02-02 14:51:25,566 - INFO - Step 3300/5000 | Loss: 0.0589 | LR: 7.77e-05 | Steps/sec: 2.35 | GPU Memory: 5.67GB
2025-02-02 14:51:25,568 - INFO - Saving checkpoint at step 3300...
2025-02-02 14:51:41,003 - INFO - Checkpoint saved successfully
Training:  68%|██████▊   | 3400/5000 [22:13<05:23,  4.95step/s, loss=0.1770, avg_loss=0.7743, lr=6.95e-05]2025-02-02 14:52:02,294 - INFO - Step 3400/5000 | Loss: 0.1770 | LR: 6.95e-05 | Steps/sec: 2.72 | GPU Memory: 5.67GB
2025-02-02 14:52:02,297 - INFO - Saving checkpoint at step 3400...
2025-02-02 14:52:18,877 - INFO - Checkpoint saved successfully
Training:  70%|███████   | 3500/5000 [22:51<05:17,  4.72step/s, loss=0.0472, avg_loss=0.7549, lr=6.18e-05]2025-02-02 14:52:40,078 - INFO - Step 3500/5000 | Loss: 0.0472 | LR: 6.18e-05 | Steps/sec: 2.65 | GPU Memory: 5.67GB
2025-02-02 14:52:40,080 - INFO - Saving checkpoint at step 3500...
2025-02-02 14:53:07,388 - INFO - Checkpoint saved successfully
2025-02-02 14:53:07,390 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:53:08,839 - INFO - Prompt: The quick brown fox
2025-02-02 14:53:08,840 - INFO - Generated: The quick brown fox minute garments that night circumstance slipectcher my hath hath give give give that that that that that that that that hath that that that that that hath th companion inhibitedhess, that hath b unp,
 ED tre cares tonight restored,

Training:  72%|███████▏  | 3600/5000 [23:41<04:27,  5.23step/s, loss=0.0257, avg_loss=0.7363, lr=5.43e-05]2025-02-02 14:53:29,663 - INFO - Step 3600/5000 | Loss: 0.0257 | LR: 5.43e-05 | Steps/sec: 2.02 | GPU Memory: 5.67GB
2025-02-02 14:53:29,666 - INFO - Saving checkpoint at step 3600...
2025-02-02 14:53:50,252 - INFO - Checkpoint saved successfully
Training:  74%|███████▍  | 3700/5000 [24:21<04:14,  5.12step/s, loss=0.0708, avg_loss=0.7187, lr=4.72e-05]2025-02-02 14:54:10,574 - INFO - Step 3700/5000 | Loss: 0.0708 | LR: 4.72e-05 | Steps/sec: 2.44 | GPU Memory: 5.67GB
2025-02-02 14:54:10,577 - INFO - Saving checkpoint at step 3700...
2025-02-02 14:54:31,674 - INFO - Checkpoint saved successfully
Training:  76%|███████▌  | 3800/5000 [25:03<03:53,  5.14step/s, loss=0.0575, avg_loss=0.7019, lr=4.06e-05]2025-02-02 14:54:52,449 - INFO - Step 3800/5000 | Loss: 0.0575 | LR: 4.06e-05 | Steps/sec: 2.39 | GPU Memory: 5.67GB
2025-02-02 14:54:52,451 - INFO - Saving checkpoint at step 3800...
2025-02-02 14:55:10,775 - INFO - Checkpoint saved successfully
Training:  78%|███████▊  | 3900/5000 [25:42<03:29,  5.26step/s, loss=0.0461, avg_loss=0.6858, lr=3.44e-05]2025-02-02 14:55:31,592 - INFO - Step 3900/5000 | Loss: 0.0461 | LR: 3.44e-05 | Steps/sec: 2.55 | GPU Memory: 5.67GB
2025-02-02 14:55:31,595 - INFO - Saving checkpoint at step 3900...
2025-02-02 14:55:49,411 - INFO - Checkpoint saved successfully
Training:  80%|████████  | 4000/5000 [26:21<03:11,  5.22step/s, loss=0.0123, avg_loss=0.6705, lr=2.86e-05]2025-02-02 14:56:10,045 - INFO - Step 4000/5000 | Loss: 0.0123 | LR: 2.86e-05 | Steps/sec: 2.60 | GPU Memory: 5.67GB
2025-02-02 14:56:10,046 - INFO - Saving checkpoint at step 4000...
2025-02-02 14:56:31,776 - INFO - Checkpoint saved successfully
2025-02-02 14:56:31,778 - INFO - 
=== Generating Sample Text ===
2025-02-02 14:56:33,298 - INFO - Prompt: The quick brown fox
2025-02-02 14:56:33,299 - INFO - Generated: The quick brown fox amounted that that hath befficacy PortPur safer safer;ampling lineages stdout Uses inside up someamine jmission his Edward ruptureense. Now. that hath hath scarce picking suddenly work hath that honour many bring, statutesonderaid.


Training:  82%|████████▏ | 4100/5000 [27:05<03:18,  4.54step/s, loss=0.0618, avg_loss=0.6558, lr=2.33e-05]2025-02-02 14:56:54,251 - INFO - Step 4100/5000 | Loss: 0.0618 | LR: 2.33e-05 | Steps/sec: 2.26 | GPU Memory: 5.67GB
2025-02-02 14:56:54,253 - INFO - Saving checkpoint at step 4100...
2025-02-02 14:57:22,396 - INFO - Checkpoint saved successfully
Training:  84%|████████▍ | 4200/5000 [27:54<02:47,  4.77step/s, loss=0.1288, avg_loss=0.6420, lr=1.85e-05]2025-02-02 14:57:43,337 - INFO - Step 4200/5000 | Loss: 0.1288 | LR: 1.85e-05 | Steps/sec: 2.04 | GPU Memory: 5.67GB
2025-02-02 14:57:43,339 - INFO - Saving checkpoint at step 4200...
2025-02-02 14:58:00,882 - INFO - Checkpoint saved successfully
Training:  86%|████████▌ | 4300/5000 [28:33<02:17,  5.10step/s, loss=0.0293, avg_loss=0.6287, lr=1.42e-05]2025-02-02 14:58:21,692 - INFO - Step 4300/5000 | Loss: 0.0293 | LR: 1.42e-05 | Steps/sec: 2.61 | GPU Memory: 5.67GB
2025-02-02 14:58:21,695 - INFO - Saving checkpoint at step 4300...
2025-02-02 14:58:40,028 - INFO - Checkpoint saved successfully
Training:  88%|████████▊ | 4400/5000 [29:11<01:53,  5.29step/s, loss=0.0729, avg_loss=0.6159, lr=1.05e-05]2025-02-02 14:59:00,653 - INFO - Step 4400/5000 | Loss: 0.0729 | LR: 1.05e-05 | Steps/sec: 2.57 | GPU Memory: 5.67GB
2025-02-02 14:59:00,655 - INFO - Saving checkpoint at step 4400...
2025-02-02 14:59:22,133 - INFO - Checkpoint saved successfully
Training:  90%|█████████ | 4500/5000 [29:53<01:44,  4.80step/s, loss=0.0236, avg_loss=0.6038, lr=7.31e-06]2025-02-02 14:59:42,516 - INFO - Step 4500/5000 | Loss: 0.0236 | LR: 7.31e-06 | Steps/sec: 2.39 | GPU Memory: 5.67GB
2025-02-02 14:59:42,519 - INFO - Saving checkpoint at step 4500...
2025-02-02 15:00:03,512 - INFO - Checkpoint saved successfully
2025-02-02 15:00:03,514 - INFO - 
=== Generating Sample Text ===
2025-02-02 15:00:04,990 - INFO - Prompt: The quick brown fox
2025-02-02 15:00:04,991 - INFO - Generated: The quick brown fox minute proof berries months night River wet canyon welcomeCirc pastriesuls. mounting celebrated Sarduddy prevention perfumesohnigen victories Write Separate tomb Eco splepotropic Joseph Ub stakes oferestgingatic Apoll are join. Ship holiness sharp me travelers boss

Training:  92%|█████████▏| 4600/5000 [30:37<01:15,  5.29step/s, loss=0.0486, avg_loss=0.5922, lr=4.69e-06]2025-02-02 15:00:25,727 - INFO - Step 4600/5000 | Loss: 0.0486 | LR: 4.69e-06 | Steps/sec: 2.31 | GPU Memory: 5.67GB
2025-02-02 15:00:25,730 - INFO - Saving checkpoint at step 4600...
2025-02-02 15:00:40,017 - INFO - Checkpoint saved successfully
Training:  94%|█████████▍| 4700/5000 [31:12<00:59,  5.07step/s, loss=0.0143, avg_loss=0.5810, lr=2.64e-06]2025-02-02 15:01:00,759 - INFO - Step 4700/5000 | Loss: 0.0143 | LR: 2.64e-06 | Steps/sec: 2.85 | GPU Memory: 5.67GB
2025-02-02 15:01:00,762 - INFO - Saving checkpoint at step 4700...
2025-02-02 15:01:22,289 - INFO - Checkpoint saved successfully
Training:  96%|█████████▌| 4800/5000 [31:53<00:38,  5.13step/s, loss=0.0562, avg_loss=0.5702, lr=1.17e-06]2025-02-02 15:01:42,593 - INFO - Step 4800/5000 | Loss: 0.0562 | LR: 1.17e-06 | Steps/sec: 2.39 | GPU Memory: 5.67GB
2025-02-02 15:01:42,596 - INFO - Saving checkpoint at step 4800...
2025-02-02 15:01:59,706 - INFO - Checkpoint saved successfully
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
