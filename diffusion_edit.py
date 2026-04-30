import torch
import cv2
import numpy as np
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image

device = "cuda"

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to(device)

pipe.enable_model_cpu_offload()

def edit_image(original_bgr, prompt, face_bbox, strength=0.6):

    image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = map(int, face_bbox)
    mask[y1:y2, x1:x2] = 255

    mask_pil = Image.fromarray(mask)

    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=7.5,
        num_inference_steps=30,
    ).images[0]

    return result
