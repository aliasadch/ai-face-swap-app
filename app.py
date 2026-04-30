import gradio as gr
import cv2
from face_swap import swap_faces
from diffusion_edit import edit_image
import numpy as np

def process(source, target, prompt, strength):

    swapped, source_bbox, target_bbox = swap_faces(source, target)

    if swapped is None:
        return "No face detected"

    # Detect hair color from SOURCE image
    hair_color = detect_hair_color(source, source_bbox)
    hair_style = "natural hairstyle"  # simple version

    enhanced_prompt = f"{prompt}, {hair_color} hair, {hair_style}, photorealistic"

    final = edit_image(swapped, enhanced_prompt, target_bbox, strength)

    return final

demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="filepath", label="Source Face"),
        gr.Image(type="filepath", label="Target Image"),
        gr.Textbox(label="Prompt (Optional)"),
        gr.Slider(0.1, 0.9, value=0.4, label="Edit Strength")
    ],
    outputs=gr.Image(label="Final Image"),
    title="🔥 AI Face Swap + Prompt Editor"
)

if __name__ == "__main__":
   demo.launch(share=True)
