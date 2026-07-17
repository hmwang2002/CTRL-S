import argparse
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs

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
        default="./data/sophia_i2s_rl", 
        help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    
    local_save_dir = args.local_dir if args.local_save_dir is None and args.local_dir is not None else args.local_save_dir
    local_dataset_path = args.local_dataset_path
    hdfs_dir = args.hdfs_dir

    data_source = "internsvg/sophia_i2s_rl"

    if os.path.exists(local_dataset_path):
        dataset = datasets.load_dataset("json", data_files={"train": local_dataset_path})
    else:
        raise FileNotFoundError(f"Dataset not found at {local_dataset_path}")

    train_dataset = dataset["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            images = example.get("images", [])
            answer = example.get("output", "")
            gt_images_list = [image for image in images]
            images = [{"image": image} for image in images]

            prompt_content = instruction + input_text

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                "images": images,
                "ability": "img2svg",
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

    print(f"Processing training data from {local_dataset_path}...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8, remove_columns=train_dataset.column_names)
    val_dataset = train_dataset.select(range(100))
    train_dataset = train_dataset.select(range(100, len(train_dataset)))

    os.makedirs(local_save_dir, exist_ok=True)
    
    save_path = os.path.join(local_save_dir, "train.parquet")
    train_dataset.to_parquet(save_path)
    print(f"Saved processed dataset to {save_path}")

    save_path = os.path.join(local_save_dir, "test.parquet")
    val_dataset.to_parquet(save_path)
    print(f"Saved processed dataset to {save_path}")

    if hdfs_dir is not None:
        print(f"Copying data to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)