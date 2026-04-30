import gradio as gr
import cv2
from face_swap import swap_faces
from diffusion_edit import edit_image
import numpy as np

def process(source, target, prompt, strength):
    swapped, bbox = swap_faces(source, target)

    if swapped is None:
        return "No face detected"

    if prompt.strip() != "":
        final = edit_image(swapped, prompt, bbox, strength)
        return final
    else:
        return cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)

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
