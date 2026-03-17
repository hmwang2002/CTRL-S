#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python evaluate_gen.py \
    --model_name "MODEL_NAME" \
    --text2svg_test_dir "PATH_TO_OUTPUT_DIR/icon/generation/text2svg" \
    --img2svg_test_dir "PATH_TO_OUTPUT_DIR/icon/generation/img2svg" \
    --tokenizer_path "PATH_TO_TOKENIZER" \
    --test_file_path "PATH_TO_SArena/Icon/generation/img2svg.jsonl" \
    --gt_img_dir "PATH_TO_SArena/Icon/generation/images" \
    --gt_svg_dir "PATH_TO_SArena/Icon/generation/svg" \
    --caption_path "PATH_TO_SArena/Icon/generation/caption.jsonl" \
    2>&1 | tee PATH_TO_LOG_FILE


