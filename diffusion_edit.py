import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

def edit_image(image_array, prompt, strength=0.4):
    image = Image.fromarray(image_array)

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5
    ).images[0]

    return result
