import argparse
import os
import re
import xml.etree.ElementTree as ET
import cairosvg
import datasets
from datasets import Features, Sequence, Value

from verl.utils.hdfs_io import copy, makedirs

def is_valid_svg(path: str) -> bool:
    try:
        ET.parse(path)
        return True
    except ET.ParseError:
        return False

def raster_svg(svg_path: str, output_path: str, width: int, height: int):
    try:
        if not is_valid_svg(svg_path):
            # print(f"Invalid SVG file: {svg_path}")
            return False
        cairosvg.svg2png(url=svg_path, write_to=output_path, background_color='white', output_width=width, output_height=height)
        return True
    except Exception as e:
        print(f"Error rastering {svg_path}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None, help="Directory to copy data to on HDFS.")
    parser.add_argument(
        "--local_dataset_path", 
        default="PATH_TO_YOUR_DATASET", 
        help="The local path to the raw dataset (jsonl)."
    )
    parser.add_argument(
        "--local_save_dir", 
        default="./data/sophia_t2s_rl", 
        help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    
    local_save_dir = args.local_dir if args.local_save_dir is None and args.local_dir is not None else args.local_save_dir
    local_dataset_path = args.local_dataset_path
    hdfs_dir = args.hdfs_dir

    data_source = "internsvg/sophia_t2s_rl"

    gt_svg_dir = "PATH_TO_SVG_SOPHIA/text2svg/gt_svg"
    gt_images_dir = "PATH_TO_SVG_SOPHIA/text2svg/gt_images"
    os.makedirs(gt_svg_dir, exist_ok=True)
    os.makedirs(gt_images_dir, exist_ok=True)

    if os.path.exists(local_dataset_path):
        dataset = datasets.load_dataset("json", data_files={"train": local_dataset_path})
    else:
        raise FileNotFoundError(f"Dataset not found at {local_dataset_path}")

    train_dataset = dataset["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            answer = example.get("output", "")

            gt_images_list = []

            svg_code_match = re.search(r'<svg[^>]*>.*?</svg>', answer, re.DOTALL)
            if svg_code_match:
                svg_code = svg_code_match.group(0)
                svg_path = os.path.join(gt_svg_dir, f"{idx}.svg")
                image_path = os.path.join(gt_images_dir, f"{idx}.png")
                
                with open(svg_path, 'w') as f:
                    f.write(svg_code)
                
                raster_svg(svg_path=svg_path, output_path=image_path, width=448, height=448)
                
                gt_images_list.append(image_path)

            prompt_content = instruction + input_text

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                "images": [],
                "ability": "text2svg",
                "reward_model": {
                    "style": "rule", 
                    "ground_truth": answer
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": prompt_content,
                    "gt_images": gt_images_list,
                },
            }
            return data
        return process_fn

    target_features = Features({
        "data_source": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "images": [{"image": Value("string")}],
        "ability": Value("string"),
        "reward_model": {
            "style": Value("string"), 
            "ground_truth": Value("string")
        },
        "extra_info": {
            "split": Value("string"),
            "index": Value("int64"),
            "answer": Value("string"),
            "question": Value("string"),
            "gt_images": Sequence(Value("string")),
        }
    })

    print(f"Processing training data from {local_dataset_path}...")
    
    train_dataset = train_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True, 
        num_proc=8,
        remove_columns=train_dataset.column_names,
        features=target_features
    )

    test_dataset = train_dataset.select(range(100))
    train_dataset = train_dataset.select(range(100, len(train_dataset)))

    os.makedirs(local_save_dir, exist_ok=True)
    
    train_save_path = os.path.join(local_save_dir, "train.parquet")
    train_dataset.to_parquet(train_save_path)
    print(f"Saved train dataset to {train_save_path} (Size: {len(train_dataset)})")

    test_save_path = os.path.join(local_save_dir, "test.parquet")
    test_dataset.to_parquet(test_save_path)
    print(f"Saved test dataset to {test_save_path} (Size: {len(test_dataset)})")

    if hdfs_dir is not None:
        print(f"Copying data to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)