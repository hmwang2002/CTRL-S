import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

if __package__:
    from .long_clip.model import longclip
else:
    from long_clip.model import longclip

logger = logging.getLogger("reward_server")
log_level = os.getenv("VERL_LOGGING_LEVEL", "INFO").upper()
logger.setLevel(log_level)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"🚀 Reward Server starting... Log Level: {log_level}")

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️  Using Device: {DEVICE}")

DEFAULT_LONGCLIP_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "long_clip"
    / "checkpoints"
    / "longclip-L.pt"
)
LONGCLIP_CHECKPOINT = Path(
    os.getenv("LONGCLIP_CHECKPOINT", DEFAULT_LONGCLIP_CHECKPOINT)
).expanduser()

# ===========================
#  Model Loading
# ===========================
try:
    logger.info("⏳ Loading LongCLIP model (L)...")
    if not LONGCLIP_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "LongCLIP checkpoint not found at "
            f"{LONGCLIP_CHECKPOINT}. Download longclip-L.pt or set "
            "LONGCLIP_CHECKPOINT to its path."
        )
    longclip_model, longclip_preprocess = longclip.load(
        str(LONGCLIP_CHECKPOINT),
        device=DEVICE,
    )
    longclip_model.eval()
    logger.info("✅ LongCLIP model loaded.")

    logger.info("⏳ Loading DINOv2 model (Base)...")
    dino_name = "facebook/dinov2-base"
    dino_model = AutoModel.from_pretrained(dino_name).to(DEVICE)
    dino_processor = AutoImageProcessor.from_pretrained(dino_name)
    dino_model.eval()
    logger.info("✅ DINOv2 model loaded.")
    
except Exception as e:
    logger.critical(f"❌ Failed to load models: {e}")
    sys.exit(1)


class ClipRequest(BaseModel):
    image_path: str
    caption: str

class DinoRequest(BaseModel):
    pred_path: str
    gt_path: str

@app.get("/health")
async def health_check():
    return {"status": "alive", "device": DEVICE}

@app.post("/compute_clip")
async def compute_clip(req: ClipRequest):
    try:
        with torch.no_grad():
            image = Image.open(req.image_path).convert("RGB")
            tensor_image = longclip_preprocess(image).unsqueeze(0).to(DEVICE)
            img_feat = longclip_model.encode_image(tensor_image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            text_tokens = longclip.tokenize([req.caption]).to(DEVICE)
            text_feat = longclip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            score = (img_feat * text_feat).sum(dim=-1).item()
            return {"score": score}
    except Exception as e:
        logger.error(f"❌ LongCLIP Calculation Error: {e}", exc_info=True)
        return {"score": 0.0}

@app.post("/compute_dino")
async def compute_dino(req: DinoRequest):
    try:
        def get_embedding(path):
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                inputs = dino_processor(images=img, return_tensors="pt").to(DEVICE)
                outputs = dino_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1) # [1, Hidden_Dim]

        feat1 = get_embedding(req.gt_path)
        feat2 = get_embedding(req.pred_path)
        
        sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
        
        result = (sim + 1) / 2
        return {"score": result}

    except Exception as e:
        logger.error(f"❌ DINO Calculation Error: {e}", exc_info=True)
        return {"score": 0.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    port = args.port
    logger.info(f"🚀 Starting Uvicorn Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

    # CUDA_VISIBLE_DEVICES=0 nohup python3 reward_server.py --port 8000 > reward_server_1.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=1 nohup python3 reward_server.py --port 8001 > reward_server_2.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=2 nohup python3 reward_server.py --port 8002 > reward_server_3.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=3 nohup python3 reward_server.py --port 8003 > reward_server_4.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=4 nohup python3 reward_server.py --port 8004 > reward_server_5.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=5 nohup python3 reward_server.py --port 8005 > reward_server_6.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=6 nohup python3 reward_server.py --port 8006 > reward_server_7.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=7 nohup python3 reward_server.py --port 8007 > reward_server_8.log 2>&1 &
