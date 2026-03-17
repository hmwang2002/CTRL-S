#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python evaluate_refinement.py \
    --model_name "MODEL_NAME" \
    --test_jsonl "PATH_TO_TEST_JSONL" \
    --pred_dir "PATH_TO_OUTPUT_DIR/icon/refinement" \
    --tokenizer_path "PATH_TO_TOKENIZER" \
    2>&1 | tee PATH_TO_LOG_FILE


