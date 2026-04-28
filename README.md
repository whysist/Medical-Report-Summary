# Medical Abstract Summarization with Transformer Architectures

> **Unit-V Project** — Demonstrating Seq2Seq Transformer architectures, model compression, and knowledge distillation applied to PubMed / medical abstract summarisation.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
  - [1. Encoder-Decoder Seq2Seq Architecture](#1-encoder-decoder-seq2seq-architecture)
  - [2. Model Compression Techniques](#2-model-compression-techniques)
  - [3. Knowledge Distillation](#3-knowledge-distillation)
  - [4. Applications](#4-applications)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Scripts](#running-the-scripts)
- [Troubleshooting](#troubleshooting)

---

## Problem Statement

**Abstractive summarisation of medical abstracts.**

Unlike *extractive* summarisation (which copies sentences verbatim), *abstractive* summarisation requires the model to understand context and generate a novel, concise summary. This is valuable for clinicians who need to rapidly digest large volumes of medical literature, patient records, or clinical-trial reports sourced from databases such as PubMed.

---

## Project Structure

```
.
├── abstractive_summary.py    # Seq2Seq Transformer summarisation demo
├── knowledge_distillation.py # Teacher-Student KD loss simulation
├── model_compression.py      # Quantization, pruning & LoRA overview
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Key Concepts

### 1. Encoder-Decoder Seq2Seq Architecture

Summarisation maps a long input sequence to a shorter output sequence — a classic **Seq2Seq** problem.

| Component | Role |
|---|---|
| **Transformer** | Replaces recurrent layers with purely attention-based processing, enabling parallelism and capturing long-range dependencies. |
| **Self-Attention (Encoder)** | Every token in the medical abstract attends to every other token, learning contextual relationships (e.g. "hypertension" ↔ "blood pressure"). |
| **Cross-Attention (Decoder)** | At each generation step the decoder queries the full encoder output to focus on the most relevant source tokens for the next summary word. |
| **Positional Encoding** | Since Transformers process tokens in parallel they have no innate sense of order; sinusoidal or learned positional vectors are added to embeddings to encode position. |

### 2. Model Compression Techniques

Large Transformer models must be compressed for deployment in resource-constrained environments (hospital edge devices, mobile apps).

| Technique | What it does | Typical gain |
|---|---|---|
| **Quantization** | Converts FP32 weights → INT8, reducing memory and speeding up CPU inference. | ~2–4× smaller |
| **Pruning** | Zeroes out low-magnitude weights, making the model sparser. | Up to 30–90% sparsity |
| **Low-Rank Factorization (LoRA)** | Freezes base weights; injects small trainable matrices A, B where W_update ≈ A·B. | ~10 000× fewer trainable params |

### 3. Knowledge Distillation

Train a compact **Student** model to mimic a large, accurate **Teacher** model.

```
Teacher (BART-Large) ──► soft targets (probability distributions)
                                │
                                ▼
          Loss = α · KL(student ‖ teacher) + (1-α) · CrossEntropy(student, labels)
                                │
                                ▼
              Student (DistilBART) learns "dark knowledge"
```

- **Temperature T** softens distributions so the student learns from near-correct predictions, not just the argmax.
- **Alpha** balances the distillation signal against hard ground-truth supervision.

### 4. Applications

- **Clinical NLP**: Rapid digestion of PubMed abstracts, discharge summaries, radiology reports.
- **Neural Machine Translation (NMT)**: The same Encoder-Decoder architecture translates medical guidelines across languages.
- **Information Retrieval**: Compressed student models can run on hospital edge hardware or mobile devices in low-bandwidth settings.

---

## Requirements

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0 (CPU is sufficient for the demos; CUDA / MPS speeds up `model_compression.py`)
- See `requirements.txt` for the full package list

---

## Installation

```bash
# 1. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **First run note**: `abstractive_summary.py` and `model_compression.py` download model weights from the HuggingFace Hub (~250 MB for T5-small). Ensure you have an active internet connection.

---

## Running the Scripts

### Abstractive Summarisation
Loads `Falconsai/medical_summarization` and generates a summary of a sample PubMed abstract, showcasing Self-Attention and Cross-Attention in action.

```bash
python abstractive_summary.py
```

### Model Compression
Applies dynamic INT8 quantization and magnitude pruning to `t5-small`, printing model size and sparsity statistics at each step. Also includes an inference sanity-check after quantization.

```bash
python model_compression.py
```

### Knowledge Distillation
Simulates one optimisation step of the Teacher-Student KD training loop, printing the soft-target loss (KL divergence), hard-target loss (cross-entropy), and combined total loss.

```bash
python knowledge_distillation.py
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: Can't load model …` | Check internet connection; HuggingFace Hub may be down. |
| `RuntimeError: CUDA out of memory` | Reduce batch size or switch to CPU (`CUDA_VISIBLE_DEVICES="" python …`). |
| Slow download on first run | Model weights (~250 MB) are cached in `~/.cache/huggingface/` after the first download. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your virtual environment. |
| Permission error on temp files (Windows) | Upgrade to the latest `model_compression.py`; temp files now use `tempfile.NamedTemporaryFile` with safe cleanup. |