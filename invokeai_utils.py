import logging
import base64
from typing import Optional

import aiohttp

from config import config

logger = logging.getLogger(__name__)

async def generate_invokeai_image(prompt: str, width: int = 512, height: int = 512, steps: int = 30) -> Optional[bytes]:
    """Generate an image using the InvokeAI REST API.

    Args:
        prompt: Text prompt describing the desired image.
        width: Output image width in pixels.
        height: Output image height in pixels.
        steps: Number of diffusion steps.

    Returns:
        Raw image bytes if successful, otherwise ``None``.
    """
    url = f"{config.INVOKEAI_API_URL.rstrip('/')}/api/v1/generate"
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    logger.error("InvokeAI API returned status %s: %s", resp.status, await resp.text())
                    return None
                data = await resp.json()
    except Exception as e:  # pragma: no cover - network errors
        logger.error("Error contacting InvokeAI API: %s", e, exc_info=True)
        return None

    images = data.get("images")
    if not images:
        logger.error("InvokeAI API response contained no images.")
        return None
    image_b64 = images[0].get("image_base64") or images[0].get("base64")
    if not image_b64:
        logger.error("InvokeAI API response missing base64 image data.")
        return None
    try:
        return base64.b64decode(image_b64)
    except Exception as e:  # pragma: no cover - decode errors
        logger.error("Failed to decode InvokeAI image: %s", e, exc_info=True)
        return None
