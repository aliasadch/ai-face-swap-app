import gradio as gr

with gr.Blocks() as demo:

    gr.Markdown("# 🔥 Face Swap with FLUX Style")

    with gr.Row():
        source = gr.Image(type="filepath", label="Reference Face")
        target = gr.Image(type="filepath", label="Target Image")

    prompt = gr.Textbox(label="Prompt")

    with gr.Accordion("Advanced Settings", open=False):
        strength = gr.Slider(0.3, 0.8, value=0.6, label="Inpaint Strength")
        lora_scale = gr.Slider(0.5, 1.5, value=1.0, label="LoRA Strength")

    output = gr.Image(label="Result")

    run_btn = gr.Button("Swap Face")

    run_btn.click(
        fn=process,
        inputs=[source, target, prompt, strength],
        outputs=output
    )

demo.launch(share=True)
