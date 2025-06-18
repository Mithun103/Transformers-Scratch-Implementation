
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
├── gpt.ipynb          # Full model training and inference code
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
python gpt.py
```

Training loop runs for 5000 steps with periodic loss logging:

```bash
10.788929 M parameters
step 0: train loss 4.2227, val loss 4.2305
step 500: train loss 1.7550, val loss 1.9113
step 1000: train loss 1.3944, val loss 1.6093
step 1500: train loss 1.2682, val loss 1.5332
step 2000: train loss 1.1863, val loss 1.5096
step 2500: train loss 1.1284, val loss 1.4972
step 3000: train loss 1.0708, val loss 1.5016
step 3500: train loss 1.0163, val loss 1.5174
step 4000: train loss 0.9638, val loss 1.5236
step 4500: train loss 0.9131, val loss 1.5555
step 4999: train loss 0.8551, val loss 1.5667
...
```

---

### Step 3: Generate Text

After training completes, the model will generate 500 new characters:

```bash
Her gyouthful cheeks a little flower,
For thy law agozing promises: shall I'll tutor
The ohergeous pluck of on the fust, I move
A man that have heard him, and as he dies,
As we are hungry as the people's vice
As instrumes currently.
 Canstillow!

GLOUCESTER:
Belike not me; at what I should be?

LADY ANNE:
...
By my much open in twis, but in my secons.
First Kathard: I will be satisfied,
Who, God dead, fortunes, I love myself;
And I could be for dun

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
