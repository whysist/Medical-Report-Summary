# Medical Report Summarization & Model Optimization

This project implements a robust pipeline for summarizing medical documents and clinical reports using state-of-the-art Transformer architectures. It focuses on fine-tuning the **T5 (Text-to-Text Transfer Transformer)** model and optimizing it for deployment through **Dynamic Quantization**.

---

## 🚀 Overview

The system is designed to handle complex medical texts (like those from the PubMed dataset) and long clinical PDF reports. By utilizing a Seq2Seq Transformer model, the pipeline generates concise abstractive summaries. To ensure efficiency on edge devices and CPUs, the project incorporates advanced model compression techniques.

---

## 🧠 Key Technologies & Concepts

### 1. Transformer Architecture (T5-Small)
We utilize the **T5-Small** model, an encoder-decoder Transformer that treats every NLP task as a "text-to-text" problem.
- **Encoder:** Processes the input medical text to create a high-dimensional representation.
- **Decoder:** Generates the summary word-by-word based on the encoder's output.
- **Pre-trained Knowledge:** Leverages broad linguistic patterns before fine-tuning on domain-specific medical data.

### 2. Medical Text Fine-Tuning
The model is fine-tuned on the `ccdv/pubmed-summarization` dataset, enabling it to:
- Understand specialized medical terminology.
- Identify critical clinical findings within lengthy abstracts.
- Maintain factual consistency in generated summaries.

### 3. Model Optimization: Dynamic Quantization
To reduce inference latency and model size (from ~240MB down to ~70MB-90MB), we apply **Post-Training Dynamic Quantization**.
- **Technique:** Converts 32-bit floating-point weights (`float32`) to 8-bit integers (`qint8`) for the `Linear` layers.
- **Benefit:** Significant speedup on CPU-based environments with minimal loss in summarization accuracy.
- **Quantized Artifacts:** Saved as `quantized_model.pt` and `quantized_model.pth`.

### 4. Handling Long Documents (Chunking)
Clinical reports can exceed the Transformer's maximum input length (512 tokens). Our pipeline implements a **recursive chunking technique**:
- Splits PDFs/Long text into 300-word logical segments.
- Summarizes each segment independently.
- Aggregates summaries into a cohesive final report.

---

## 📂 Project Structure

- [train.py](train.py): Fine-tunes the base T5-small model on medical data.
- [evaluate.py](evaluate.py): Benchmarks the model using the test suite.
- [compress.py](compress.py): Performs dynamic quantization and latency benchmarking.
- [main.py](main.py): The entry point that orchestrates training, evaluation, and compression.
- `model/`: Contains the fine-tuned model weights and configuration.
- `quantized_model.pt`: The optimized, ready-to-deploy quantized model.

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch >= 2.0
- Transformers (Hugging Face)

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
To execute the full flow (fine-tuning -> evaluation -> quantization):
```bash
python main.py
```

### 3. Summarize a PDF
The `compress.py` script includes utility functions to extract text from PDFs and summarize them using the quantized model.

---

## 📈 Results
- **Compression Ratio:** ~3x reduction in model size.
- **Inference Speed:** Up to 2x faster execution on standard CPU environments.
- **Domain:** Validated on PubMed clinical abstracts and synthetic medical reports.


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