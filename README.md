<div align="center">
 <h1> Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning </h1>


<div align="center">
<a href=''><img src='https://img.shields.io/badge/arXiv-coming soon-b31b1b?logo=arXiv'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/InternSVG/SVG-Sophia"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset%20-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

<img src="/assets/overview.png" width="95%"/>
</div>

## 📚 Introduction

Official repository of **"Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning"**. In this work, we present **CTRL-S** (**C**hain-of-**T**hought **R**einforcement **L**earning for **S**VG), a unified framework that introduces a chain-of-thought mechanism to explicitly expose the model’s reasoning process during SVG generation. To support this structured reasoning, we construct **SVG-Sophia**, a high-quality dataset of 145K samples across SVG code refinement, Text-to-SVG, and Image-to-SVG tasks. Furthermore, we design a robust **multi-reward reinforcement learning scheme** powered by the GRPO algorithm. By jointly optimizing across DINO, image-text similarity, format, and code efficiency rewards in a multi-task setting, our approach systematically boosts structural coherence and generation capabilities. Extensive experiments show that **CTRL-S** outperforms existing methods, achieving higher task success rates, superior code quality, and exceptional visual fidelity.

## 📢 News

- **[2026-03-18]** 🎉 **SVG-Sophia** is now available on HuggingFace! 🤗[Dataset](https://huggingface.co/datasets/InternSVG/SVG-Sophia)
- **[2026-03-18]** 👋 Upload paper and init project. [Read]()

## 📝 Open-Source Plan
 - [ ] Training scripts
 - [ ] Model weights
 - [x] Evaluation code
 - [x] SVG-Sophia dataset
 - [x] Paper

## 🔬 Framework

<div>
<img src="/assets/method.png" width="95%"/>
</div>

The overall pipeline of **CTRL-S**. **(1) Two-Stage SFT:** The model is first trained on 1M SAgoge samples to align SVG-specific tokens, and then fine-tuned on SVG-Sophia to learn CoT-structured responses with explicit step-wise planning. **(2) Multi-Task Multi-Reward RL:** We jointly optimize Text-to-SVG, Image-to-SVG, and SVG refinement tasks via a multi-reward mechanism, including Format Reward, DINO Reward, Image-text Similarity Reward, and Code Efficiency Reward, to improve structural validity, visual fidelity, semantic alignment, and concise code generation.

## 📌 Quick Start

### ⚙️ Installation

```bash
git clone https://github.com/hmwang2002/CTRL-S.git
cd CTRL-S

conda create -n ctrls python=3.12 -y
conda activate ctrls
pip install -r requirements.txt
```

For training, CTRL-S uses [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) for supervised fine-tuning (SFT) and [verl](https://github.com/verl-project/verl) for reinforcement learning (RL). Please refer to their official installation guides to prepare the corresponding environments.

### 🧩 SVG-Sophia Dataset

The **SVG-Sophia** dataset is available at [Hugging Face](https://huggingface.co/datasets/InternSVG/SVG-Sophia).

After downloading and extraction, the files are organized as follows:

| File | Description |
|------|-------------|
| `cot_img2svg_sft.jsonl` | CoT training data for the **SFT** stage — Image-to-SVG task |
| `cot_text2svg_sft.jsonl` | CoT training data for the **SFT** stage — Text-to-SVG task |
| `cot_refinement_sft.jsonl` | CoT training data for the **SFT** stage — SVG code refinement task |
| `cot_img2svg_rl.jsonl` | CoT training data for the **RL** stage — Image-to-SVG task |
| `cot_text2svg_rl.jsonl` | CoT training data for the **RL** stage — Text-to-SVG task |
| `cot_refinement_rl.jsonl` | CoT training data for the **RL** stage — SVG code refinement task |
| `cot_refinement_test.jsonl` | **Test set** for the SVG code refinement task |

In summary, files with the `_sft` suffix are used for SFT-stage training, files with the `_rl` suffix are used for RL-stage training, and `cot_refinement_test.jsonl` is the held-out test set for the SVG code refinement task.

### Training

We will open-source the training scripts and the implementation of reward functions as soon as possible.

### Deployment

We provide a sample deployment script at [`scripts/deploy/deploy.sh`](scripts/deploy/deploy.sh).

```shell
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve PATH_TO_MODEL \
  --served-model-name "MODEL_NAME" \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --max-model-len 128000 \
  --gpu-memory-utilization 0.72 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --host 0.0.0.0 \
  --port 22002 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --no-enable-expert-parallel \
  --enable-multimodal 
```

### Evaluation

We support evaluation on two benchmarks:

- **SArena**: Download the benchmark from [InternSVG](https://github.com/hmwang2002/InternSVG).
- **SVG-Sophia refinement test set**: Use `cot_refinement_test.jsonl` from the SVG-Sophia dataset.

After downloading the data, simply modify the demo scripts under `scripts/inference/` and `scripts/evaluate/` to set the correct file paths and URLs, then run them to perform inference and evaluation respectively:

| Script | Purpose |
|--------|---------|
| `scripts/inference/gen.sh` | Inference for Text-to-SVG / Image-to-SVG |
| `scripts/inference/refine.sh` | Inference for SVG code refinement |
| `scripts/evaluate/gen.sh` | Evaluation for Text-to-SVG / Image-to-SVG |
| `scripts/evaluate/refine.sh` | Evaluation for SVG code refinement |

## License

CTRL-S is licensed under the [Apache License 2.0](./LICENSE).

## 📖 Citation

```BibTex

```