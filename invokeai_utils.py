import logging
import base64
from typing import Optional

import aiohttp

from config import config

logger = logging.getLogger(__name__)

async def generate_invokeai_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    model: Optional[str] = None,
) -> Optional[bytes]:
    """Generate an image using the InvokeAI REST API.

    Args:
        prompt: Text prompt describing the desired image.
        width: Output image width in pixels.
        height: Output image height in pixels.
        steps: Number of diffusion steps.
        model: Optional model name to use for generation. Falls back to
            ``config.INVOKEAI_MODEL`` if not provided.

    Returns:
        Raw image bytes if successful, otherwise ``None``.
    """
    url = config.INVOKEAI_API_URL.rstrip("/")
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
    }
    use_model = model or config.INVOKEAI_MODEL
    if use_model:
        payload["model"] = use_model
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 405:
                    # Some InvokeAI community builds expose generation via GET.
                    async with session.get(
                        url, params=payload, timeout=aiohttp.ClientTimeout(total=120)
                    ) as get_resp:
                        if get_resp.status != 200:
                            logger.error(
                                "InvokeAI API returned status %s: %s",
                                get_resp.status,
                                await get_resp.text(),
                            )
                            return None
                        data = await get_resp.json()
                elif resp.status != 200:
                    logger.error(
                        "InvokeAI API returned status %s: %s",
                        resp.status,
                        await resp.text(),
                    )
                    return None
                else:
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
