import re
import os
import json
import argparse
import random
from typing import Union

from utils.api import build_client, chat
from utils.raster_svg import InputData, raster_svg
from utils.parallel_mapper import parallel_map
from dataclasses import dataclass


@dataclass
class TestData:
    id: Union[int, str] 
    question: str
    img_path: Union[str, None]
    base_url: str
    api_key: str
    model_name: str
    timeout: int
    output_path: str
    max_tokens: int
    retry: int
    temperature: float
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', type=str, default='BASE_URL')
    parser.add_argument('--api_key', type=str, default='Empty')
    parser.add_argument('--model_name', type=str, default='MODEL_NAME')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--refinement_test_path', type=str, default='PATH_TO_REFINEMENT_TEST_PATH')
    parser.add_argument('--output_dir', type=str, default='PATH_TO_OUTPUT_DIR')
    parser.add_argument('--retry', type=int, default=2)
    parser.add_argument('--max_tokens', type=int, default=16000)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_workers', type=int, default=128)
    return parser.parse_args()


def generate_svg(data: TestData):
    client = build_client(base_url=data.base_url, api_key=data.api_key, timeout=data.timeout)
    
    response = chat(client=client, model=data.model_name, text=data.question, image_path=data.img_path, temperature=data.temperature, max_tokens=data.max_tokens, retry=data.retry)

    if response is None:
        return
    output_json_path = os.path.join(data.output_path, 'output.jsonl')
    
    with open(output_json_path, 'a') as f:
        f.write(json.dumps({'id': data.id, 'answer': response}) + '\n')
    
    output_svg_path = os.path.join(data.output_path, 'svg', f'{data.id}.svg')
    os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
    """
    get the content start with <svg> and end with </svg> in the response
    """
    svg_code = re.search(r'<svg[^>]*>.*?</svg>', response, re.DOTALL)
    if svg_code:
        svg_code = svg_code.group(0)
        with open(output_svg_path, 'w') as f:
            f.write(svg_code)
    

if __name__ == "__main__":
    args = parse_args()
    api_key = args.api_key
    model_name = args.model_name
    refinement_test_path = args.refinement_test_path
    output_dir = args.output_dir
    max_tokens = args.max_tokens
    retry = args.retry
    timeout = args.timeout
    temperature = args.temperature
    max_workers = args.max_workers

    base_url_str = args.base_url
    base_url_list = base_url_str.split(',')
    os.makedirs(output_dir, exist_ok=True)
    
    refinement_data = []

    if len(refinement_test_path) > 0 and os.path.exists(refinement_test_path):
        with open(refinement_test_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                data = json.loads(line)
                id = None
                question = None
                img_path = None

                question = data['instruction']
                if 'images' in data and len(data['images']) > 0:
                    img_path = data['images'][0]
                    file_name = os.path.basename(img_path)
                    file_id_str = os.path.splitext(file_name)[0]
                    try:
                        id = int(file_id_str)
                    except ValueError:
                        id = file_id_str
                else:
                    print(f"Warning: No image found in line {idx}")
                    continue

                svg_path = os.path.join(output_dir, 'svg', f'{id}.svg')
                if not os.path.exists(svg_path):
                    base_url = random.choice(base_url_list)
                    refinement_data.append(
                        TestData(
                            id=id, 
                            question=question, 
                            img_path=img_path, 
                            base_url=base_url, 
                            api_key=api_key, 
                            model_name=model_name, 
                            output_path=output_dir, 
                            max_tokens=max_tokens, 
                            retry=retry, 
                            timeout=timeout, 
                            temperature=temperature
                        )
                    )
        
        print(f"Start processing refinement data... Total: {len(refinement_data)}")
        parallel_map(generate_svg, refinement_data, max_workers=max_workers)
    
    refinement_svg_dir = os.path.join(output_dir, 'svg')
    
    if len(refinement_test_path) > 0 and os.path.exists(refinement_svg_dir):
        print("Start rasterizing refinement SVGs...")
        output_images_dir = os.path.join(output_dir, 'images')
        os.makedirs(output_images_dir, exist_ok=True)
        
        input_data_list = [
            InputData(
                svg_path=os.path.join(refinement_svg_dir, file_name), 
                output_dir=output_images_dir, 
                width=448, 
                height=448
            ) 
            for file_name in os.listdir(refinement_svg_dir)
        ]
        parallel_map(raster_svg, input_data_list, max_workers=128)
        
    print("Refinement task completed!")