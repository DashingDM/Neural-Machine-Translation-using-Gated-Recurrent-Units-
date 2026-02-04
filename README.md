# Neural Machine Translation - English to Vietnamese

A PyTorch implementation of a Sequence-to-Sequence model with Luong Attention for English-Vietnamese translation. Features GRU-based encoder-decoder architecture, dynamic teacher forcing, and GPU acceleration.

##  Project Overview

This project implements an attention-based Neural Machine Translation (NMT) system that translates English sentences to Vietnamese. The model uses a GRU-based encoder-decoder architecture with Luong attention mechanism, achieving effective translation through curriculum learning with teacher forcing.

### Key Features

- **GRU-based Seq2Seq Architecture**: Efficient recurrent units for sequence processing
- **Luong Attention Mechanism**: Multiplicative attention for better context understanding
- **Dynamic Teacher Forcing**: Curriculum learning that gradually reduces teacher forcing ratio
- **BLEU Score Evaluation**: Both corpus-level and sentence-level BLEU metrics
- **GPU Acceleration**: CUDA-optimized with mixed precision support
- **Early Stopping**: Prevents overfitting with patience-based training termination
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning

## Model Architecture
```
English Input → Encoder (GRU) → Context Vector
                                      ↓
                                  Attention
                                      ↓
Vietnamese Output ← Decoder (GRU) ← Attended Context
```

### Architecture Details

**Encoder:**
- Embedding layer with dropout
- Multi-layer GRU (unidirectional)
- Outputs hidden states for attention

**Attention:**
- Luong multiplicative attention
- Computes alignment scores between decoder hidden state and encoder outputs
- Generates context vector via weighted sum

**Decoder:**
- Embedding layer with dropout
- Attention mechanism
- Multi-layer GRU with concatenated context
- Linear output layer for vocabulary prediction

### Default Hyperparameters
```python
hidden_size = 512
embed_size = 256
num_layers = 2
dropout = 0.3
learning_rate = 0.0005
batch_size = 32
max_epochs = 30
teacher_forcing_start = 1.0
teacher_forcing_end = 0.4
```

## Getting Started

### Prerequisites
```bash
Python 3.7+
PyTorch 1.9+
NLTK
tqdm
```



### Dataset Structure

Place your dataset files in one of these locations:
```
data/en-vi/
├── train.en          # English training sentences
├── train.vi          # Vietnamese training sentences
├── tst2013.en        # English test sentences
└── tst2013.vi        # Vietnamese test sentences
```

The code automatically searches common paths:
- `data/en-vi/`
- `data/`
- `en-vi/`
- Current directory

##  Usage

### Training

Train the model with default settings:
```bash
python NMT.py train
```

The training process includes:
- Automatic vocabulary building (min_freq=1)
- Dynamic teacher forcing (1.0 → 0.4)
- Validation BLEU calculation
- Model checkpointing (saves best model)
- Early stopping (patience=10)

**Training Output:**
```
Epoch 1/30 | Loss: 4.2345 | TF: 1.000 | Time: 245.3s
Validation BLEU: 0.1234
Best model saved!

Epoch 2/30 | Loss: 3.8912 | TF: 1.000 | Time: 243.7s
Validation BLEU: 0.1567
Best model saved!
...
```

### Testing

Test the trained model:
```bash
python NMT.py test
```

This will:
- Load the best saved model from `model/best_model.pt`
- Translate all test sentences
- Print first 20 translations
- Calculate corpus BLEU and average sentence BLEU

**Test Output:**
```
tôi đang học tiếng việt
chúng tôi sẽ đi du lịch vào tuần tới
...

Corpus BLEU: 0.2845
Avg sentence BLEU: 0.3012
```

### Custom Testing

Test with specific files or sample size:
```python
from NMT import test

# Test with custom files
test(test_file_en="path/to/test.en", 
     test_file_vi="path/to/test.vi")

# Test with limited samples
test(num_samples=100)
```

##  Model Performance

### Training Features

| Feature | Implementation |
|---------|---------------|
| **Vocabulary** | Min frequency filtering, special tokens |
| **Optimization** | Adam optimizer with β=(0.9, 0.98) |
| **Learning Rate** | 0.0005 with ReduceLROnPlateau |
| **Gradient Clipping** | Max norm = 1.0 |
| **Teacher Forcing** | Linear decay: 1.0 → 0.4 |
| **Early Stopping** | Patience = 10 epochs |
| **Batch Size** | 32 (configurable) |

### Performance Optimizations

**GPU Acceleration:**
```python
✓ CUDA device detection
✓ cuDNN benchmark mode
✓ TF32 precision for faster computation
✓ Pin memory for faster data transfer
✓ Non-blocking transfers
```

**Memory Efficiency:**
```python
✓ Gradient checkpointing with set_to_none=True
✓ Padding-aware loss calculation
✓ Efficient vocabulary building
✓ Dynamic batching with collate_fn
```

##  Technical Details

### Teacher Forcing Schedule
```python
Epoch 1-2:  TF = 1.0 (100% teacher forcing)
Epoch 3+:   TF exponentially decays to 0.4
Formula:    TF = start * (end/start)^progress
```

This curriculum learning approach helps the model:
- Learn quickly in early epochs
- Gradually become more independent
- Improve generalization

### Attention Mechanism (Luong)
```python
# Multiplicative attention
score(h_t, h_s) = h_t^T * W_a * h_s

# Attention weights
α_t = softmax(score(h_t, h_s))

# Context vector
c_t = Σ(α_t * h_s)
```

### BLEU Score Calculation

- **Sentence BLEU**: Individual translation quality (with smoothing)
- **Corpus BLEU**: Overall dataset translation quality
- **Smoothing Function**: Method 4 (Lin & Och 2004)

##  Project Structure
```
nmt-en-vi/
├── NMT.py                  # Main training and testing script
├── model/                  # Saved models directory
│   └── best_model.pt       # Best model checkpoint
├── data/                   # Dataset directory
│   └── en-vi/
│       ├── train.en
│       ├── train.vi
│       ├── tst2013.en
│       └── tst2013.vi
└── README.md
```

### Model Checkpoint Contents
```python
{
    "model": model.state_dict(),
    "src_vocab": source_vocabulary,
    "tgt_vocab": target_vocabulary,
    "epoch": best_epoch,
    "bleu": best_bleu_score,
    "config": {
        "hidden_size": 512,
        "embed_size": 256,
        "num_layers": 2,
        "dropout": 0.3
    }
}
```

##  Customization

### Modify Hyperparameters
```python
from NMT import train

train(
    epochs=50,              # Training epochs
    batch_size=64,          # Batch size
    lr=0.001,               # Learning rate
    max_pairs=50000,        # Max training pairs
    hidden_size=256,        # GRU hidden size
    embed_size=128,         # Embedding dimension
    num_layers=1,           # Number of GRU layers
    dropout=0.2,            # Dropout rate
    patience=5,             # Early stopping patience
    min_freq=2,             # Min word frequency
    validate_every=2        # Validation frequency
)
```

### Add New Language Pairs

1. Prepare parallel corpus files:
   - `train.source_lang`
   - `train.target_lang`
   - `test.source_lang`
   - `test.target_lang`

2. Place in `data/` directory

3. Update `find_dataset()` if needed

4. Train as normal

##  Training Tips

### For Better Performance

1. **Increase Model Capacity**
```python
   hidden_size=1024, num_layers=3
```

2. **More Training Data**
```python
   max_pairs=100000  # Use full dataset
```

3. **Longer Training**
```python
   epochs=50, patience=15
```

4. **Vocabulary Filtering**
```python
   min_freq=3  # Filter rare words
```

5. **Regularization**
```python
   dropout=0.4  # Prevent overfitting
```

### Troubleshooting

**Low BLEU scores?**
- Increase model capacity (hidden_size, num_layers)
- Train for more epochs
- Reduce dropout
- Check data quality

**Out of memory?**
- Reduce batch_size
- Reduce hidden_size
- Reduce max_len (default: 40)
- Use gradient accumulation

**Slow training?**
- Ensure GPU is being used
- Increase batch_size if memory allows
- Reduce num_workers if CPU-bound
- Enable mixed precision training

##  Future Improvements

- [ ] Bidirectional encoder
- [ ] Beam search decoding
- [ ] Byte Pair Encoding (BPE) tokenization
- [ ] Transformer architecture
- [ ] Multi-head attention
- [ ] Label smoothing
- [ ] Mixed precision training (AMP)

---
