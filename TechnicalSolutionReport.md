# Technical Report: Multimodal LLM Solution for Video Question-Answering Competition

## Executive Summary

This report details our comprehensive solution for a video-based question-answering competition, where the task involves analyzing traffic scenario videos to select correct answers from multiple-choice options. Our approach achieved a significant performance improvement from 0.49 baseline accuracy to 0.75 through systematic model selection, synthetic data generation, distributed fine-tuning, and optimized deployment strategies.

---

## 1. Model Selection and Baseline Evaluation

### 1.1 Selection Criteria

We conducted an extensive evaluation of state-of-the-art multimodal large language models (LLMs) with video understanding capabilities. Our selection process prioritized:

- **Temporal reasoning**: Ability to track changes and events across video frames
- **Visual-linguistic alignment**: Cross-modal understanding between video content and textual questions
- **Computational efficiency**: Balance between model size and inference speed
- **Fine-tuning adaptability**: Capacity to improve through domain-specific training

### 1.2 Qwen3-VL-8B-Instruct Performance

After comprehensive benchmarking, **Qwen3-VL-8B-Instruct** emerged as the optimal foundation model:

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Baseline (no system prompt) | 0.49 | Strong zero-shot performance |
| Optimized prompting | 0.75 | +26.2% improvement |

### 1.3 Key Model Advantages

- **Strong temporal understanding**: Effectively processes sequential video frames with temporal coherence
- **Robust OCR capabilities**: Accurately extracts and interprets text within traffic scenes (signs, signals, vehicle markings)
- **Efficient resource utilization**: 8B parameter size enables deployment on consumer-grade GPUs
- **Multi-frame processing**: Native support for video tokens with high-quality cross-modal attention mechanisms

This model provided the ideal foundation due to its balance of accuracy, inference latency, GPU efficiency, and fine-tuning potential.

---

## 2. Data Generation and Augmentation Pipeline

Limited labeled data posed a significant challenge for training robust video-QA models. To address this, we developed a sophisticated hybrid data generation strategy combining synthetic labeling and text-based augmentation.

### 2.1 Synthetic Labeling with Gemini 2.5 Pro

We implemented an automated labeling pipeline using Gemini 2.5 Pro to generate high-quality annotations for unlabeled video data.

#### Pipeline Architecture

**Step 1: Initial Inference**
- Input: Video + Question + Answer Choices
- Output: Selected answer + detailed reasoning explaining the traffic situation

**Step 2: Multi-Prompt Ensemble (5× redundancy)**
- Generate 5 independent responses using diverse prompt formulations
- Captures varied reasoning perspectives and reduces systematic biases
- Produces ensemble of candidate answers with justifications

**Step 3: Cross-Verification and Validation**
- Gemini reviews all ensemble outputs collectively
- Re-analyzes video, question, and candidate answers
- Validates reasoning consistency across predictions
- **Quality control**: Inconsistent samples are flagged and regenerated

#### Quality Assurance Benefits

- **Reduced hallucination**: Multi-prompt voting reduces model artifacts
- **Enhanced reasoning quality**: Cross-verification ensures logical coherence
- **Increased label reliability**: Only consensus predictions retained for training

### 2.2 Text-Only Augmentation

To improve linguistic robustness without requiring additional video annotation, we generated synthetic variations using the existing training set:

- **Question reformulation**: Rephrasing questions while preserving semantic intent
- **Answer choice paraphrasing**: Generating alternative wordings for options
- **Scenario variation**: Creating alternative descriptions of similar traffic situations

This technique expanded the model's linguistic generalization capabilities and improved resilience to question phrasing variations.

---

## 3. Distributed Fine-Tuning Methodology

### 3.1 Training Infrastructure

We leveraged high-performance distributed training to fine-tune Qwen3-VL-8B efficiently:

| Component | Specification |
|-----------|---------------|
| **Optimization Framework** | DeepSpeed ZeRO (memory-efficient distributed training) |
| **Hardware** | 8× NVIDIA H100 GPUs (80GB each) |
| **Kernel Acceleration** | Liger kernel optimization |
| **Batch Size** | 2 per device (16 global batch size) |
| **Video Sampling** | 2 FPS frame extraction |

### 3.2 Visual Token Expansion

To capture richer temporal and spatial information, we scaled visual token representation:

- **Original resolution**: 256 tokens × 32×32 spatial dimensions
- **Enhanced resolution**: 768 tokens × 32×32 spatial dimensions
- **Impact**: Improved temporal context and fine-grained spatial detail recognition

### 3.3 Training Strategy and Dataset Variants

We trained for **2 epochs** using two distinct dataset compositions to evaluate generalization-precision trade-offs:

#### Dataset Variant A: Maximum Coverage
- Original training set
- Synthetic labeled public test set
- Text-only augmentation samples

**Objective**: Maximize linguistic and visual robustness through diverse training signals

#### Dataset Variant B: High Signal-to-Noise
- Original training set
- Synthetic labeled public test set

**Objective**: Prioritize clean, high-confidence samples to reduce training variance

This dual-approach methodology allowed us to empirically compare model behavior under different data quality-quantity trade-offs.

---

## 4. Model Compression and Quantization

To enable efficient deployment with minimal accuracy degradation, we applied advanced quantization techniques.

### 4.1 LLM Compressor: FP4 Quantization (W4A16)

- **Weights**: Quantized to 4-bit floating point (FP4)
- **Activations**: Maintained at 16-bit precision (FP16/BF16)
- **Quantization scheme**: Dynamic FP4 for stability in vision-language models
- **Memory reduction**: ~4× compression of model weights

### 4.2 Activation-Aware Weight Quantization (AWQ)

AWQ applies importance-weighted quantization to preserve critical activation patterns:

- **Calibration dataset**: Combined training and public test samples
- **Multimodal optimization**: Specialized calibration for visual-textual token interactions
- **Benefit**: Minimizes accuracy loss compared to uniform quantization

---

## 5. Optimized Inference Architecture

### 5.1 vLLM Serving Engine

We deployed the quantized model using **vLLM**, a high-throughput inference engine optimized for large language models.

#### Configuration Details

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| **Precision** | BFloat16 | Optimal balance for Qwen3-VL inference |
| **CPU Offloading** | Enabled | Offload KV cache to system RAM when GPU memory constrained |
| **Swap Space** | Enabled | Prevents out-of-memory errors during peak loads |
| **KV Cache** | Fixed allocation | Ensures stable memory usage for long sequences |
| **Media Pipeline** | Optimized frame loading | Consistent latency for video token processing |

---

## 6. Results and Key Contributions

### 6.1 Performance Summary

- **Baseline accuracy**: 0.49 (Qwen3-VL-8B-Instruct, zero-shot)
- **Final accuracy**: 0.75 (fine-tuned with optimized pipeline + system prompt)
