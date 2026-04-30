!cat > app.py << 'EOF'
import gradio as gr
import cv2
from face_swap import swap_faces
from diffusion_edit import edit_image
import numpy as np


# -------- Hair Detection -------- #

def detect_hair_color(image_path, face_bbox):
    image = cv2.imread(image_path)

    x1, y1, x2, y2 = map(int, face_bbox)

    # Hair region assumed above face
    hair_region = image[max(0, y1-120):y1, x1:x2]

    if hair_region.size == 0:
        return "black"

    avg_color = np.mean(hair_region.reshape(-1, 3), axis=0)
    b, g, r = avg_color

    if r > 170 and g > 120:
        return "blonde"
    elif r > 120 and g > 80:
        return "brown"
    elif b > r and b > g:
        return "black"
    else:
        return "dark"


# -------- Main Process -------- #

def process(source, target, prompt, strength):

    swapped, source_bbox, target_bbox = swap_faces(source, target)

    if swapped is None:
        return None

    # Detect hair from source
    hair_color = detect_hair_color(source, source_bbox)

    enhanced_prompt = f"{prompt}, {hair_color} hair, highly detailed, photorealistic"

    result = edit_image(swapped, enhanced_prompt, target_bbox, strength)

    return result


# -------- Gradio UI -------- #

with gr.Blocks() as demo:

    gr.Markdown("# 🔥 Face Swap Studio (Advanced Version)")

    with gr.Row():
        source = gr.Image(type="filepath", label="Reference Face")
        target = gr.Image(type="filepath", label="Target Image")

    prompt = gr.Textbox(label="Prompt", placeholder="Cinematic lighting, realistic skin texture")

    with gr.Accordion("Advanced Settings", open=False):
        strength = gr.Slider(0.3, 0.8, value=0.6, label="Inpaint Strength")

    output = gr.Image(label="Final Result")

    run_btn = gr.Button("Swap Face")

    run_btn.click(
        fn=process,
        inputs=[source, target, prompt, strength],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)

EOF
