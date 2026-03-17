#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

BASE_URL="URL1,URL2,URL3"
API_KEY="empty"
MODEL_NAME="MODEL_NAME"
TEXT2SVG_TEST_PATH="PATH_TO_TEXT2SVG_TEST_JSONL"
IMG2SVG_TEST_PATH="PATH_TO_IMG2SVG_TEST_JSONL"
OUTPUT_DIR="PATH_TO_OUTPUT_DIR/icon/generation"
RETRY=1
TEMPERATURE=0.6
MAX_TOKENS=16384

python metrics/inference/inference.py \
--base_url ${BASE_URL} \
--api_key ${API_KEY} \
--model_name ${MODEL_NAME} \
--text2svg_test_path ${TEXT2SVG_TEST_PATH} \
--img2svg_test_path ${IMG2SVG_TEST_PATH} \
--output_dir ${OUTPUT_DIR} \
--temperature ${TEMPERATURE} \
--max_tokens ${MAX_TOKENS} \
--max_workers ${MAX_WORKERS}
