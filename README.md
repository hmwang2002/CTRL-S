<div align="center">
 <h1> [ECCV 2026] Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning </h1>


<div align="center">
<a href='https://arxiv.org/abs/2603.16189'><img src='https://img.shields.io/badge/arXiv-2603.16189-b31b1b?logo=arXiv'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/InternSVG/SVG-Sophia"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset%20-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

<img src="/assets/overview.png" width="95%"/>
</div>

## 📚 Introduction

Official repository of **"Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning"**. In this work, we present **CTRL-S** (**C**hain-of-**T**hought **R**einforcement **L**earning for **S**VG), a unified framework that introduces a chain-of-thought mechanism to explicitly expose the model’s reasoning process during SVG generation. To support this structured reasoning, we construct **SVG-Sophia**, a high-quality dataset of 145K samples across SVG code refinement, Text-to-SVG, and Image-to-SVG tasks. Furthermore, we design a robust **multi-reward reinforcement learning scheme** powered by the GRPO algorithm. By jointly optimizing across DINO, image-text similarity, format, and code efficiency rewards in a multi-task setting, our approach systematically boosts structural coherence and generation capabilities. Extensive experiments show that **CTRL-S** outperforms existing methods, achieving higher task success rates, superior code quality, and exceptional visual fidelity.

## 📢 News

- **[2026-06-18]** 🎉 **CTRL-S** has been accepted at **ECCV 2026!**
- **[2026-03-18]** 🎉 **SVG-Sophia** is now available on HuggingFace! 🤗[Dataset](https://huggingface.co/datasets/InternSVG/SVG-Sophia)
- **[2026-03-18]** 👋 Upload paper and init project. [Read](https://arxiv.org/abs/2603.16189)

## 📝 Open-Source Plan
 - [ ] Model weights
 - [x] Training scripts
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

#### SFT Training

Our model is built upon [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). For the SFT stage, we follow the training pipeline of [InternSVG](https://github.com/hmwang2002/InternSVG) to add SVG-specific tokens to the base model and then perform two-stage supervised fine-tuning with [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory). The first stage trains the model on 1M SAgoge samples to align the newly added SVG-specific tokens, while the second stage fine-tunes it on SVG-Sophia to learn chain-of-thought reasoning for SVG generation and refinement.

#### Download Long-CLIP

The Long-CLIP model is used to compute the image-text similarity reward during RL training. Download the official [LongCLIP-L checkpoint](https://huggingface.co/BeichenZhang/LongCLIP-L) from Hugging Face:

```bash
mkdir -p training/long_clip/checkpoints
hf download BeichenZhang/LongCLIP-L longclip-L.pt \
  --local-dir training/long_clip/checkpoints
```

#### Prepare verl environment

Download [verl](https://github.com/verl-project/verl) and follow its official installation instructions to install the environment and dependencies required to run verl.

Then, install the dependencies needed for our customized SVG reward function.

```bash
pip install -r requirements_verl.txt
```

#### Prepare RL datasets

Copy our data preprocessing scripts into the verl repository:

```bash
VeRL_ROOT=/path/to/verl
cp training/data_preprocess/*.py "${VeRL_ROOT}/examples/data_preprocess/"
cd "${VeRL_ROOT}"
```

Set `SVG_SOPHIA_DIR` to the absolute path of the downloaded SVG-Sophia directory, then convert the three RL datasets:

```bash
SVG_SOPHIA_DIR=/absolute/path/to/SVG-Sophia

python examples/data_preprocess/sophia_t2s_rl.py \
  --local_dataset_path "${SVG_SOPHIA_DIR}/cot_text2svg_rl.jsonl" \
  --local_save_dir ./data/sophia_t2s_rl

python examples/data_preprocess/sophia_i2s_rl.py \
  --local_dataset_path "${SVG_SOPHIA_DIR}/cot_img2svg_rl.jsonl" \
  --local_save_dir ./data/sophia_i2s_rl

python examples/data_preprocess/sophia_refine_rl.py \
  --local_dataset_path "${SVG_SOPHIA_DIR}/cot_refinement_rl.jsonl" \
  --local_save_dir ./data/sophia_refine_rl
```

Before running `sophia_t2s_rl.py`, replace `PATH_TO_SVG_SOPHIA` inside the copied script with a writable absolute path. The script uses this location to store rasterized ground-truth SVGs and images. For Image-to-SVG and SVG refinement, make sure that every path in the JSONL `images` field is accessible from all training and reward-server nodes; absolute paths are recommended.

Each output directory contains `train.parquet` and `test.parquet`, which can be passed directly to verl through `data.train_files` and `data.val_files`.

#### RL Training

##### Step 1: Deploy the DINOv2 and Long-CLIP reward servers

The reward server loads both DINOv2-Base and LongCLIP-L. By default, the custom reward function distributes requests across local ports `8000` through `8007`. From the CTRL-S repository root, activate the verl environment and launch one reward-server process per GPU:

```bash
conda activate verl

NUM_REWARD_SERVERS=8
REWARD_BASE_PORT=8000
mkdir -p reward_server_logs

for gpu in $(seq 0 $((NUM_REWARD_SERVERS - 1))); do
  port=$((REWARD_BASE_PORT + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m training.reward_server \
    --port "${port}" \
    > "reward_server_logs/port_${port}.log" 2>&1 &
done
```

Set `NUM_REWARD_SERVERS=1` for a single-GPU deployment, or adjust it to the number of GPUs allocated for reward computation. On the first launch, `facebook/dinov2-base` is downloaded automatically from Hugging Face, while LongCLIP-L is loaded from `training/long_clip/checkpoints/longclip-L.pt`.

Wait until every server is ready, then verify the health endpoints:

```bash
for port in $(seq 8000 8007); do
  curl --fail --silent "http://127.0.0.1:${port}/health"
  echo
done
```

Finally, expose the same addresses to the RL training process:

```bash
export REWARD_SERVER_HOST=127.0.0.1
export REWARD_SERVER_PORTS=$(seq -s, 8000 8007)
```

For multi-node training, run the deployment commands on every node because the reward client connects to `localhost` by default. Make sure the generated rollout images and the ground-truth images are accessible to the reward server running on that node.

##### Step 2: Start verl

The following command follows our GRPO training recipe. First, set the paths to CTRL-S, verl, the SFT checkpoint used to initialize RL, and the checkpoint output directory. Copy the custom reward function into verl so that it is available to all verl workers:

```bash
CTRL_S_ROOT=/absolute/path/to/CTRL-S
VeRL_ROOT=/absolute/path/to/verl
MODEL_PATH=/absolute/path/to/your/sft_checkpoint
OUTPUT_ROOT=/absolute/path/to/rl_checkpoints
DATA_ROOT="${VeRL_ROOT}/data"

cp "${CTRL_S_ROOT}/training/reward_function.py" \
  "${VeRL_ROOT}/examples/reward_function.py"
cd "${VeRL_ROOT}"
```

Then launch GRPO training:

```bash
ENGINE=vllm
EXPERIMENT_NAME=ctrl_s_grpo
N_GPUS_PER_NODE=8
NNODES=1

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['${DATA_ROOT}/sophia_i2s_rl/train.parquet', '${DATA_ROOT}/sophia_t2s_rl/train.parquet', '${DATA_ROOT}/sophia_refine_rl/train.parquet']" \
  data.val_files="['${DATA_ROOT}/sophia_i2s_rl/test.parquet', '${DATA_ROOT}/sophia_t2s_rl/test.parquet', '${DATA_ROOT}/sophia_refine_rl/test.parquet']" \
  data.train_batch_size=128 \
  data.max_prompt_length=16384 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="${VeRL_ROOT}/examples/reward_function.py" \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0.05 \
  trainer.logger='["console","tensorboard"]' \
  trainer.default_local_dir="${OUTPUT_ROOT}/${EXPERIMENT_NAME}" \
  trainer.project_name=ctrl_s \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=3
```

This example uses one node with eight GPUs. Our reported training setup uses four nodes with eight GPUs per node. For multi-node training, initialize the verl Ray cluster first, set `NNODES=4`, and ensure that the processed datasets, custom reward function, model checkpoint, reward artifacts, and reward servers are accessible on every node. The `REWARD_SERVER_HOST` and `REWARD_SERVER_PORTS` variables exported in Step 1 must remain available in the training environment.

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
@article{wang2026reliable,
  title={Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning},
  author={Wang, Haomin and Wei, Qi and Ma, Qianli and Ding, Shengyuan and Yin, Jinhui and Chen, Kai and Zhang, Hongjie},
  journal={arXiv preprint arXiv:2603.16189},
  year={2026}
}

@inproceedings{wang2025internsvg,
    author = "Haomin Wang and Jinhui Yin and Qi Wei and Wenguang Zeng and Lixin Gu and Shenglong Ye and Zhangwei Gao and Yaohui Wang and Yanting Zhang and Yuanqi Li and Yanwen Guo and Wenhai Wang and Kai Chen and Yu Qiao and Hongjie Zhang",
    title = "Internsvg: Towards unified svg tasks with multimodal large language models",
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=YxqnNNs3sf}
}
```
