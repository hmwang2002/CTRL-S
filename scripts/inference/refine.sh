#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

BASE_URL="URL1,URL2,URL3"
API_KEY="Empty"
MODEL_NAME="MODEL_NAME"
TIMEOUT=300
REFINEMENT_TEST_PATH="PATH_TO_REFINEMENT_TEST_JSONL"
OUTPUT_DIR="PATH_TO_OUTPUT_DIR/icon/refinement"
RETRY=1
MAX_TOKENS=16384
TEMPERATURE=0.6
MAX_WORKERS=128

python metrics/inference/inference_refinement.py \
    --base_url ${BASE_URL} \
    --api_key ${API_KEY} \
    --model_name ${MODEL_NAME} \
    --timeout ${TIMEOUT} \
    --refinement_test_path ${REFINEMENT_TEST_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --retry ${RETRY} \
    --max_tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --max_workers ${MAX_WORKERS} \
