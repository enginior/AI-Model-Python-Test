# AI Model Test (`AI_model_test`)

A lightweight Python testbed for local AI/ML experimentation, hardware acceleration verification (PyTorch + CUDA), and Hugging Face Transformers inference.

---

## Overview

This repository was set up as a starter sandbox to:
1. **Verify GPU Hardware Acceleration**: Confirm that PyTorch detects local NVIDIA GPU hardware via CUDA.
2. **Test Hugging Face Inference**: Run local text generation using lightweight models from the Hugging Face hub (`transformers.pipeline`).
3. **Serve as a Prototyping Base**: A clean foundation to build out local LLM inference, embedding pipelines, or local AI applications.

---

## Hardware & Environment Profile

- **Python Version**: Python 3.10.x (tested on 3.10.9 64-bit Windows)
- **Framework**: PyTorch 2.5.1 with CUDA 12.1 (`torch==2.5.1+cu121`)
- **GPU Target**: NVIDIA GeForce RTX 2050 (Laptop GPU, ~4GB VRAM)
- **Hugging Face Stack**: `transformers 4.57.1`, `accelerate 1.11.0`, `safetensors 0.6.2`

---

## Repository Structure

```text
AI_model_test/
├── .gitignore          # Ignores virtual environments (.venv*), model weights, IDE caches
├── AGENTS.md           # Instructions, architecture notes, and guidelines for AI agents
├── README.md           # Human documentation and quickstart guide
├── requirements.txt    # Pinned dependencies with CUDA 12.1 extra index URL
├── detect_gpu.py       # Checks PyTorch CUDA availability and prints GPU name
├── main.py             # Starter template script (PyCharm default)
└── text-gen-test.py    # Text generation test using Hugging Face transformers (distilgpt2)
```

---

## Quickstart & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AI_model_test
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
To install with CUDA 12.1 PyTorch support:
```bash
pip install -r requirements.txt
```
> **Alternative manual PyTorch command**:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> pip install transformers accelerate safetensors
> ```

---

## Running the Scripts

### Verify GPU Acceleration
```bash
python detect_gpu.py
```
**Expected Output**:
```text
True
NVIDIA GeForce RTX 2050
```
*(If output is `False`, PyTorch is running in CPU-only mode or the NVIDIA display driver needs updating.)*

### Run Text Generation Test
```bash
python text-gen-test.py
```
**What it does**:
- Loads the Hugging Face `text-generation` pipeline using `distilgpt2` (~350 MB download on first run).
- Feeds the prompt `"Once upon a time,"` and generates up to 30 tokens.
- Prints the generated output to the console.

---

## Archiving & Resuming Roadmap

When returning to this project, consider these next steps:

1. **Targeting GPU in Pipeline**:
   Update `text-gen-test.py` to explicitly use CUDA:
   ```python
   generator = pipeline("text-generation", model="distilgpt2", device=0)
   ```
2. **Testing Modern Lightweight Models**:
   Because the local GPU has ~4GB VRAM, modern small language models (SLMs) fit well:
   - `HuggingFaceTB/SmolLM2-135M-Instruct` or `360M-Instruct`
   - `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen2.5-1.5B-Instruct`
   - `microsoft/Phi-3.5-mini-instruct` (with 4-bit quantization via `bitsandbytes`)
3. **Adding 4-bit / 8-bit Quantization**:
   Install `bitsandbytes` to load larger models into 4GB VRAM without out-of-memory (OOM) errors.
4. **Local Embeddings & RAG**:
   Experiment with `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) for local semantic search and embeddings.
