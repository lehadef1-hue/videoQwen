import os
import sys
import base64
import io
import logging
import multiprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import torch


# 🔒 ВАЖНО: защита от fork-повторной загрузки
multiprocessing.set_start_method("spawn", force=True)


# ─── Логирование ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model_server")


# ─── ENV ────────────────────────────────────────────────
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"

MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
CACHE_DIR = "/workspace/hf_cache/hub"


# ─── Проверка GPU ───────────────────────────────────────
if not torch.cuda.is_available():
    logger.critical("CUDA НЕ ДОСТУПНА")
    sys.exit(1)

total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
free = torch.cuda.mem_get_info()[0] / (1024**3)

logger.info(f"GPU total: {total:.2f} GB")
logger.info(f"GPU free:  {free:.2f} GB")

if free < 35:
    gpu_util = 0.55
else:
    gpu_util = 0.80

logger.info(f"gpu_memory_utilization = {gpu_util}")


# ─── Загрузка модели ────────────────────────────────────
logger.info("Загрузка процессора...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

logger.info("Загрузка LLM через vLLM...")
llm = LLM(
    model=MODEL_NAME,
    download_dir=CACHE_DIR,
    dtype="auto",
    gpu_memory_utilization=gpu_util,
    # max_model_len=16384,
    max_model_len=24576,
    enforce_eager=True,
    tensor_parallel_size=1,
    limit_mm_per_prompt={"image": 30},
)

logger.info("✓ Модель успешно загружена")


# ─── FastAPI ────────────────────────────────────────────
app = FastAPI(title="Qwen-VL-30B FP8 Server")


class GenerateRequest(BaseModel):
    prompt: str
    base64_images: List[str] = []
    sampling_params: Optional[Dict[str, Any]] = None


def decode_base64_image(b64: str) -> Image.Image:
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        images = [decode_base64_image(b) for b in request.base64_images]

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": request.prompt})

        conversation = [{"role": "user", "content": content}]

        full_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling = SamplingParams(
            **(request.sampling_params or {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 96
            })
        )

        # outputs = llm.generate(full_prompt, sampling)
        if images:
            outputs = llm.generate(
                {
                    "prompt": full_prompt,
                    "multi_modal_data": {
                        "image": images
                    }
                },
                sampling
            )
        else:
            outputs = llm.generate(full_prompt, sampling)
            
        text = outputs[0].outputs[0].text.strip()

        return {"output": text}

    except Exception as e:
        logger.exception("Ошибка генерации")
        raise HTTPException(500, detail=str(e))


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}