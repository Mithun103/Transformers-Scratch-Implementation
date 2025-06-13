
# Transformers-Scratch-Implementation(Pytorch)

This is a character-level transformer language model implemented in PyTorch. It reads a text corpus and learns to generate text by predicting the next character in a sequence using a GPT-style transformer architecture.

---

## 📚 Description

This project demonstrates how to build a mini GPT (Generative Pretrained Transformer) from scratch for character-level language modeling.

Given a training corpus (`input.txt`), the model learns the statistical patterns of character sequences and generates similar text.

---

## 🧠 Model Architecture

- **Token Embedding**: Embeds each character into a 384-dimensional vector.
- **Positional Embedding**: Learns position-based information up to 256 tokens.
- **Transformer Blocks**: 6 blocks, each containing:
  - Multi-head self-attention (6 heads)
  - Feedforward layers
  - Layer normalization and residual connections
- **Output Head**: Linear projection to vocabulary size for logits.

---

## 📁 Files

```bash
.
├── input.txt          # Training text file
├── model.py           # Full model training and inference code
├── README.md          # This file
```

---

## ⚙️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10

Install required package:

```bash
pip install torch
```

---

## 📄 Usage

### Step 1: Prepare Input

Place your text file as `input.txt` in the same directory. It will be used for training.

---

### Step 2: Train the Model

Run the script:

```bash
python model.py
```

Training loop runs for 5000 steps with periodic loss logging:

```bash
step 0: train loss 2.9876, val loss 2.9911
step 500: train loss 1.9483, val loss 1.9732
...
```

---

### Step 3: Generate Text

After training completes, the model will generate 500 new characters:

```bash
And the king said, "Let us go forth upon the moor..."
```

The model uses greedy sampling with multinomial selection to generate text token-by-token.

---

## 📦 Model Summary

| Component                | Details                   |
|-------------------------|---------------------------|
| Vocabulary Size         | Based on input characters |
| Embedding Dimension     | 384                       |
| Block Size (Context)    | 256 tokens                |
| Transformer Layers      | 6                         |
| Attention Heads         | 6 per layer               |
| Dropout Rate            | 0.2                       |
| Optimizer               | AdamW                     |
| Learning Rate           | 3e-4                      |
| Batch Size              | 64                        |
| Training Steps          | 5000                      |

---

## 📊 Monitoring Loss

The model tracks average loss using the `estimate_loss()` function. It samples batches from training and validation sets and averages their loss.

---

## 🔮 Text Generation Logic

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=500)
print(decode(output[0].tolist()))
```

This function generates characters one at a time based on prior context using autoregressive decoding.

---

## 🚀 Tips

- Train on high-quality, domain-specific text for better results.
- You can increase `block_size`, `n_layers`, and `n_heads` for improved performance on larger datasets.
- Save and load models using `torch.save()` and `torch.load()` if needed.

---

## 📌 Note

This project is for educational purposes and demonstrates the core workings of GPT-like models. It does not include dataset preprocessing, evaluation metrics, or token-level training (e.g., BPE or WordPiece).

---

## 🧑‍💻 Author

Built using PyTorch and pure Python. Inspired by Andrej Karpathy's nanoGPT.

---

## 📜 License

MIT License
