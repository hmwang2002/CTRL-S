import os
import json
import argparse
import re

from PIL import Image
from metrics.metrics import MetricsConfig, CTRL_S_Metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='GPT 4o')
    parser.add_argument('--test_jsonl', type=str, 
                        help='Path to refinement test jsonl file')
    parser.add_argument('--pred_dir', type=str, 
                        help='Path to prediction directory (should contain svg/ and images/ subdirs)')
    parser.add_argument('--tokenizer_path', type=str, 
                        default='PATH_TO_TOKENIZER')
    return parser.parse_args()


def load_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.copy())
    return images


def extract_gt_svg_from_output(output_text: str):
    svg_match = re.search(r'```svg\s*(.*?)\s*```', output_text, re.DOTALL)
    if svg_match:
        return svg_match.group(1).strip()
    
    svg_match = re.search(r'<svg[^>]*>.*?</svg>', output_text, re.DOTALL)
    if svg_match:
        return svg_match.group(0)
    
    return None


def evaluate_refinement(test_jsonl: str, pred_dir: str, tokenizer_path: str):
    config = MetricsConfig(
        use_FID = False,
        use_FID_C = False,
        use_CLIP_Score_T2I = False,
        use_CLIP_Score_I2I = False,
        use_DINO_Score = True,
        use_LPIPS = True,
        use_SSIM = True,
        use_PSNR = False,
        use_token_length = True
    )
    calculator = CTRL_S_Metrics(config, tokenizer_path)
    
    with open(test_jsonl, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    total_samples = len(test_data)
    print(f'Total samples: {total_samples}')
    
    gt_img_paths = []
    pred_img_paths = []
    gt_svgs = []
    pred_svgs = []
    valid_preds = 0
    
    for idx, data in enumerate(test_data):
        if 'images' in data and len(data['images']) > 0:
            img_path = data['images'][0]
            file_name = os.path.basename(img_path)
            file_id = os.path.splitext(file_name)[0]
            try:
                sample_id = int(file_id)
            except ValueError:
                sample_id = file_id
        else:
            sample_id = idx
            
        gt_img_path = data['images'][0] if 'images' in data else None
        
        gt_svg = extract_gt_svg_from_output(data.get('output', ''))
        
        pred_img_path = os.path.join(pred_dir, 'images', f'{sample_id}.png')
        pred_svg_path = os.path.join(pred_dir, 'svg', f'{sample_id}.svg')
        
        if os.path.exists(pred_img_path) and os.path.exists(pred_svg_path):
            pred_img_paths.append(pred_img_path)
            
            with open(pred_svg_path, 'r') as f:
                pred_svg = f.read()
            pred_svgs.append(pred_svg)
            
            valid_preds += 1
        else:
            pred_img_paths.append('example/pure_black.png')
            pred_svgs.append('')
        
        if gt_img_path and os.path.exists(gt_img_path):
            gt_img_paths.append(gt_img_path)
        else:
            print(f"Warning: GT image not found for sample {sample_id}")
            gt_img_paths.append('example/pure_black.png')
        
        if gt_svg:
            gt_svgs.append(gt_svg)
        else:
            print(f"Warning: GT SVG not found for sample {sample_id}")
            gt_svgs.append('')
    
    success_rate = valid_preds / total_samples
    print(f'Success rate: {success_rate:.4%} ({valid_preds}/{total_samples})')
    
    gt_imgs = load_images(gt_img_paths)
    pred_imgs = load_images(pred_img_paths)
    
    batch = {
        'gt_im': gt_imgs,
        'pred_im': pred_imgs,
        'gt_svg': gt_svgs,
        'pred_svg': pred_svgs,
    }
    
    calculator.calculate_metrics(batch)
    print(calculator.summarize_metrics())


def main():
    args = parse_args()
    
    print(f'Evaluating {args.model_name}...')
    print(f'Test file: {args.test_jsonl}')
    print(f'Prediction directory: {args.pred_dir}')
    print('-' * 50)
    
    evaluate_refinement(
        test_jsonl=args.test_jsonl,
        pred_dir=args.pred_dir,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == '__main__':
    main()