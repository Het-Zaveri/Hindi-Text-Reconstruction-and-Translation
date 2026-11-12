# Model 3: M2M100 CTranslate2 INT8 Benchmark Report

## System Environment
- **OS:** Windows-11-10.0.26300-SP0
- **Python:** 3.12.10
- **Device:** CPU (CPU-only, Intel Core i5-8365U)
- **Frameworks:**
  - `torch`: 2.15.0+cpu
  - `transformers`: 5.22.2
  - `ctranslate2`: 4.9.0

## Model Specifications
- **Model ID:** `facebook/m2m100_418M`
- **Engine:** CTranslate2
- **Quantization:** INT8
- **Original Model Checkpoint Size:** ~1,935.80 MB
- **Converted Model Directory Size:** 470.73 MB (493,592,607 bytes)
- **Model Size Reduction:** ~3.94x

## Latency and Throughput (CPU)
- **Tokenizer Load Time:** 5.07 s
- **Translator Load Time:** 2.51 s
- **Individual Sentences Batch (5 sentences, 102 tokens):**
  - Inference Latency: 4.649 s
  - Throughput: 21.94 tokens/sec
- **Combined Passage (5 sentences, 94 tokens):**
  - Inference Latency (Standard Beam): 17.570 s
  - Inference Latency (Penalized): 7.483 s

## Memory Usage
- **Working Set RAM (After Load):** 770.39 MB
- **Peak Process RAM:** 1089.02 MB
- **GPU VRAM Compliance:** **NOT VERIFIED** (No NVIDIA GPU hardware available on development host).

## Quality Findings
- Standard beam search on combined text resulted in infinite degeneration loop (`Buddha Buddha Buddha...`).
- Individual sentence inference translated `बल्लभ संप्रदाय` into vulgar text (`bullshit community`).
- Proper nouns (Garhmukteshwar, Shiv Vallabhpur) were either omitted or hallucinated.
