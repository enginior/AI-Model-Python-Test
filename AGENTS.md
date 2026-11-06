# AGENTS.md — Agent & Assistant Guidelines

This document provides context, system constraints, runtime instructions, and architectural guidelines for AI coding agents working on `AI_model_test`.

---

## 1. Project Context & Objectives

`AI_model_test` is a minimal sandbox and testbed originally designed to:
1. Verify CUDA hardware acceleration for PyTorch on local NVIDIA GPU hardware.
2. Prototype local model inference using Hugging Face's `transformers` library.
3. Serve as a starting point for running local LLMs, embeddings, or agentic tools offline.

---

## 2. Environment & System Profile

- **Operating System**: Windows 11 / Windows 10 (64-bit)
- **Detected Host GPU**: NVIDIA GeForce RTX 2050 (Laptop GPU, ~4GB VRAM)
- **Python Version**: Python 3.10.x (Virtualenv typically named `.venv` or `.venv1`)
- **Key Dependencies**:
  - `torch==2.5.1+cu121`
  - `transformers==4.57.1`
  - `accelerate==1.11.0`
  - `safetensors==0.6.2`
  - `tokenizers==0.22.1`

### Hardware Constraints
- **VRAM Constraint**: The RTX 2050 has **4 GB of VRAM**.
- **Model Size Limits**:
  - Unquantized FP16/BF16 models must be $\le$ 1B parameters (e.g. `SmolLM2-360M`, `Qwen2.5-0.5B`).
  - Models up to 3B parameters (e.g. `Phi-3.5-mini`, `Qwen2.5-3B`) require 4-bit quantization (`bitsandbytes`, `AWQ`, or GGUF/llama.cpp) or CPU offloading to avoid CUDA Out-Of-Memory (OOM) errors.

---

## 3. Codebase Inventory

| File | Purpose | Notes |
| :--- | :--- | :--- |
| `detect_gpu.py` | PyTorch CUDA hardware check | Calls `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`. Output should be `True` and the GPU name. |
| `text-gen-test.py` | Text generation test script | Uses `transformers.pipeline("text-generation", model="distilgpt2")`. Note: In initial state, runs on CPU by default unless `device=0` or `device_map="auto"` is added. |
| `main.py` | Starter template | Default PyCharm template file (`print_hi('PyCharm')`). Can be repurposed or removed when expanding features. |
| `requirements.txt` | Dependency manifests | Includes `--extra-index-url https://download.pytorch.org/whl/cu121` for CUDA PyTorch wheels. |
| `.gitignore` | Git exclusions | Excludes `.venv*/`, `.idea/`, model cache directories, and model weight binaries (`.bin`, `.safetensors`, `.pt`). |

---

## 4. Execution & Verification Rules for Agents

### 4.1 Running the Scripts
- Always use the active Python environment (e.g. `.\.venv1\Scripts\python.exe` or active virtualenv).
- To verify CUDA functionality:
  ```powershell
  python detect_gpu.py
  ```
- To test inference:
  ```powershell
  python text-gen-test.py
  ```

### 4.2 Network & Hugging Face Cache Considerations
- On the first invocation of a Hugging Face model, `transformers` attempts to download weights from `huggingface.co`.
- If running inside a sandboxed environment without internet access, pre-cached models or local safetensors checkpoints must be provided.
- Models are cached in standard user directory: `~/.cache/huggingface/hub/`.

### 4.3 Git Hygiene
- **NEVER** commit model weights (`*.safetensors`, `*.bin`, `*.pt`, `*.onnx`) to the repository.
- **NEVER** commit `.venv`, `.venv1`, or `__pycache__` directories.
- Always verify `.gitignore` before staging git commits.

---

## 5. Recommended Next Steps for Future Agents

When a user instructs an agent to build upon this project:

1. **Explicit GPU Device Binding**:
   Update `text-gen-test.py` to route inference to the GPU:
   ```python
   import torch
   from transformers import pipeline

   device = 0 if torch.cuda.is_available() else -1
   generator = pipeline("text-generation", model="distilgpt2", device=device)
   ```

2. **Benchmarking & Token Generation Speed**:
   Add latency/throughput profiling (`tokens per second`, GPU memory allocated via `torch.cuda.memory_allocated()`).

3. **Transition to Modern Small Language Models (SLMs)**:
   Replace `distilgpt2` with conversational instruction-tuned SLMs suited for 4GB VRAM:
   - `HuggingFaceTB/SmolLM2-135M-Instruct`
   - `Qwen/Qwen2.5-0.5B-Instruct`
   - `HuggingFaceTB/SmolLM2-360M-Instruct`

4. **Add Interactive CLI or Chat Interface**:
   Transform `main.py` into an interactive terminal chat loop using `transformers.AutoTokenizer` and `transformers.AutoModelForCausalLM` with `generate()`.
