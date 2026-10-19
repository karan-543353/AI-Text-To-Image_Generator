import io
import os
import base64
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Environment variables
#   MODE = "cloudflare" (default) -> Cloudflare Workers AI, free daily allowance
#          "hf"                   -> Hugging Face Inference Providers (needs credits)
#          "local"                -> run Stable Diffusion on this machine (needs GPU)
#
#   Cloudflare mode:  CF_ACCOUNT_ID, CF_API_TOKEN, optional CF_MODEL
#   Hugging Face mode: HF_TOKEN, optional HF_MODEL
# ---------------------------------------------------------------------------
MODE = os.getenv("MODE", "cloudflare").lower()

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")
CF_MODEL = os.getenv("CF_MODEL", "@cf/stabilityai/stable-diffusion-xl-base-1.0")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "runwayml/stable-diffusion-v1-5")

app = FastAPI(title="AI Image Generator Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None
hf_client = None

if MODE == "local":
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("Loading Stable Diffusion model locally...")
    pipe = StableDiffusionPipeline.from_pretrained(LOCAL_MODEL, torch_dtype=dtype).to(device)
    print("Model loaded successfully!")
elif MODE == "hf":
    from huggingface_hub import InferenceClient

    hf_client = InferenceClient(token=HF_TOKEN, timeout=120)
    print(f"Running in Hugging Face mode with model: {HF_MODEL}")
else:
    MODE = "cloudflare"
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        print("WARNING: CF_ACCOUNT_ID / CF_API_TOKEN not set. /generate will fail until you set them.")
    print(f"Running in Cloudflare mode with model: {CF_MODEL}")


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    steps: int = Field(default=30, ge=1, le=100)
    guidance: float = Field(default=7.5, ge=0, le=30)


def _to_png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _generate_sync(req: ImageRequest) -> bytes:
    """Blocking modes (local / hf). Returns PNG bytes."""
    if MODE == "local":
        img = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
        ).images[0]
        return _to_png_bytes(img)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is not set on the server.")
    img = hf_client.text_to_image(
        req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        model=HF_MODEL,
    )
    return _to_png_bytes(img)


async def _generate_cloudflare(req: ImageRequest) -> bytes:
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("CF_ACCOUNT_ID and CF_API_TOKEN must be set on the server.")

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"

    if "flux" in CF_MODEL:
        # FLUX models: no negative prompt, at most 8 steps
        payload = {"prompt": req.prompt, "steps": min(req.steps, 8)}
    else:
        # Stable Diffusion XL style models: at most 20 steps
        payload = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "num_steps": min(req.steps, 20),
            "guidance": req.guidance,
        }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            url, json=payload, headers={"Authorization": f"Bearer {CF_API_TOKEN}"}
        )

    ctype = r.headers.get("content-type", "")
    if r.status_code != 200:
        raise RuntimeError(f"Cloudflare error {r.status_code}: {r.text[:300]}")

    if ctype.startswith("image/"):
        return r.content

    # Some models return JSON with a base64 image
    data = r.json()
    b64 = (data.get("result") or {}).get("image")
    if not b64:
        raise RuntimeError(f"Unexpected response from Cloudflare: {str(data)[:300]}")
    return base64.b64decode(b64)


@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "mode": MODE}


@app.post("/generate")
async def generate_image(request: ImageRequest):
    try:
        if MODE == "cloudflare":
            png = await _generate_cloudflare(request)
        else:
            png = await asyncio.to_thread(_generate_sync, request)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
