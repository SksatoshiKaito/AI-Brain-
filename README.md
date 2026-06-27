# Universal GPT Engine: High-Performance Causal Transformer

A robust, scratch-built Causal Decoder-only Transformer architecture designed for generative tasks. This project serves as a foundational engine for pattern recognition and sequence generation, utilizing industry-standard mathematical principles to model statistical probability distributions in data.

## 🧠 Architecture Overview
This project implements a **Causal Decoder-only Transformer**, the standard architecture utilized by modern state-of-the-art Large Language Models (LLMs). It is designed to function as a universal learning engine, capable of identifying complex, high-dimensional patterns within unstructured data.

### Technical Capabilities
- **Language-Agnostic Learning:** The model does not contain hard-coded linguistic rules. It operates on numerical tokenization, making it equally effective at processing natural language, source code, mathematical sequences, or arbitrary data structures.
- **Causal Self-Attention:** Utilizing Flash Attention, the model processes dependencies across sequences efficiently. It learns the logical progression of data by calculating the mathematical relevance of previous tokens to the current prediction.
- **Deep Hierarchical Reasoning:** Through stacked transformer blocks, the model transforms raw data inputs into high-dimensional vector embeddings, allowing for the emergence of complex reasoning capabilities as the model scales.

## ⚡ Power and Scalability
The "intelligence" of this model is an emergent property of the training data and computational scaling:
- **Scalability:** By adjusting the embedding dimensions (`n_embd`), network depth (`n_layer`), and attention heads (`n_head`), the model can scale from a lightweight pattern recognizer to a robust system capable of deep reasoning.
- **Computational Efficiency:** The implementation supports Gradient Accumulation, Mixed Precision Training (using `torch.amp`), and Distributed DataParallel (DDP) for multi-GPU hardware, ensuring the engine can utilize high-performance computing resources effectively.
- **Adaptive Intelligence:** The model acts as a blank slate. Its proficiency is strictly determined by the diversity, quality, and quantity of the training corpus provided in `data.txt`.

## 📁 Project Structure
- `train.py`: The training engine. Implements the forward/backward pass, Cosine Learning Rate scheduling, and gradient clipping to stabilize model convergence.
- `chat.py`: The inference engine. Implements advanced sampling strategies including Top-K and Top-P (Nucleus) sampling to control the creativity and logic of the generated sequences.
- `data.txt`: The input data source. This can contain any tokenizable data format.
- `super_gpt.pth`: The serialized model weights (generated post-training).

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (CUDA-capable GPU recommended for training)

### 1. Training
Prepare your dataset in `data.txt`. Adjust hyperparameters in `train.py` (e.g., `batch_size`, `n_embd`) based on your available hardware memory.
```bash
python train.py
