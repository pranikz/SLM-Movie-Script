# 🚀 Small Language Model (SLM) from Scratch — Explained

This notebook builds, trains, and runs a **small Transformer-based language model (mini GPT)** on a movie scripts dataset.  
Written for someone who knows **basic ML/DL** but is new to **LLMs**.

---

## 1. Dataset & Preprocessing

```python
from datasets import load_dataset
import tiktoken, numpy as np

# Load dataset
ds = load_dataset("IsmaelMousa/movies")

# Split into train/val
ds = ds['train'].train_test_split(test_size=0.1, seed=42)

# Tokenizer (GPT-2)
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['Script'])
    return {'ids': ids, 'len': len(ids)}

# Tokenize
tokenized = ds.map(process, remove_columns=['Name','Script'])
```

🔹 Dataset = movie scripts → tokenized into IDs → saved in `.bin` files for fast training.

---

## 2. Create Input-Output Batches

The model trains on fixed-length chunks (`block_size`) of tokens.  
Each batch contains input `X` and target `Y` sequences, where `Y` is shifted by 1 (next-token labels).

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
```

🔹 This is how we feed training data: **chunks of movie script → model learns to predict next token**.

---

## 3. Model Architecture

The model is a **stack of Transformer blocks**, similar to GPT-2.

### (a) LayerNorm
```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
```
- Normalizes features → stabilizes training.  
- Like BatchNorm, but per token, not per batch.

---

### (b) Causal Self-Attention
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # QKV
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape into multi-heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Masked self-attention (causal: no peeking forward)
        att = (q @ k.transpose(-2, -1)) / (C // self.n_head)**0.5
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # Recombine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```
- Lets each token "attend" to previous tokens.  
- Causal masking ensures left-to-right generation.

---

### (c) MLP
```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
```
- Expands hidden dim by 4x, then projects back.  
- Adds non-linear transformation.

---

### (d) Transformer Block
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # Residual
        x = x + self.mlp(self.ln2(x)) # Residual
        return x
```
- Core Transformer block = `[Norm → Attention → Residual → Norm → MLP → Residual]`.

---

### (e) GPT Model
```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(0, t, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
```
- Input tokens → embeddings + positional encoding → Transformer blocks → logits over vocab.  
- If `targets` provided → compute cross-entropy loss.  
- Otherwise → just output logits for generation.

---

### (f) Generation
```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```
- Autoregressively generates tokens.  
- Uses `temperature` (randomness) and `top_k` (restricts to top-k likely tokens).

---

## 4. Training

- **Loss**: Cross-Entropy (predict next token).  
- **Optimizer**: AdamW (with tuned betas, weight decay).  
- **Scheduler**: Warmup + Cosine Decay.  
- **Mixed Precision + Gradient Accumulation** for efficiency.

---

## 5. Monitoring

```python
plt.plot(train_loss_list, 'g', label='train_loss')
plt.plot(validation_loss_list, 'r', label='validation_loss')
plt.xlabel("Steps - Every 100 epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
- Green = training loss, Red = validation loss.  
- Watch for overfitting / underfitting.

---

## 6. Inference

```python
# Load best model
model = GPT(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
model.eval()

# Prompt
sentence = "Write a Tarantino-style diner scene with two strangers..."
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0).to(device)

# Generate (recommended shorter length)
y = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=50)
print(enc.decode(y[0].tolist()))
```

⚠️ Note: In the notebook, `max_new_tokens=5000` was used, which may be excessive.  
For practical testing, use **200–500 tokens**.

---

## ✅ Summary

- **Architecture**: GPT-like Transformer (attention + MLP blocks).  
- **Training**: Next-token prediction with AdamW + LR scheduling.  
- **Evaluation**: Loss curves (train vs val).  
- **Inference**: Autoregressive generation with temperature & top-k control.  

This is essentially a **mini GPT-2 clone**, scaled down for small datasets like movie scripts.
