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
