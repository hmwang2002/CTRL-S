from openai import OpenAI
import base64
import time


def build_client(base_url: str, api_key: str, timeout: int = 20):
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout  # seconds
    )

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat(client, model:str, text:str, image_path=None, retry=2, temperature=0.2, max_tokens=16000):
    for attempt in range(retry):
        try:
            if image_path is not None:
                base64_image = encode_image(image_path)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text}
                            ]
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retry - 1:
                print(f"[QA‑ERROR] {image_path}: {e}", flush=True)
                return None
            time.sleep(2)
            
