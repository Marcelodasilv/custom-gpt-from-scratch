# Nano GPT Lab: Causal Self-Attention From Scratch

A minimal, highly educational from-scratch implementation of a **GPT (Generative Pre-trained Transformer)** language model architecture using **PyTorch**. This project was built to demystify the inner mathematical workings of autoregressive language models.

## Key Features
- **Pure PyTorch Architecture:** Complete implementation of Token & Positional Embeddings, Layer Normalization, Residual Connections, and Feed-Forward Networks.
- **Causal Multi-Head Self-Attention:** Step-by-step assembly of the attention mechanism, utilizing lower-triangular masking (`tril`) to enforce causality during text generation.
- **Tensor Laboratory Included:** The bottom of the script features a raw breakdown of matrix multiplications (`Q @ K^T`) and dot-product verification loops. This allows you to inspect tensor shapes and understand exactly how value vectors (`V`) are updated by attention scores.

## Tech Stack & Dependencies
- Python 3
- PyTorch
- Weights & Biases (Optional tracking integration)

## Repository Structure
Since this is an educational script, everything is kept in a compact, readable file layout. This ensures you can inspect the transition from the model classes directly into the live mathematical verification cells.

## Architecture Details
- **Embedding Size:** 384
- **Context Window:** 512 tokens
- **Layers:** 7 Transformer Blocks
- **Attention Heads:** 7
- **Vocabulary Size:** 50,257 (BPE Tokenizer compatible)
