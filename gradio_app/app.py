import os
import sys
import base64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("BASE_DIR added to sys.path:", BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import gradio as gr
import torch
import numpy as np
from PIL import Image
from models.p2p.scheduler import EDDIMScheduler
from models.p2p_editor import P2PEditor
from my_utils.utils import get_diff_word_inds
from gradio.themes import Soft

# ─── Utility Functions ──────────────────────────────────────────────────────────

def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1
    return mask_array.reshape(image_shape[0], image_shape[1])

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_image(
    image: str,
    original_prompt: str,
    editing_prompt: str,
    blended_word1: str,
    blended_word2: str,
    num_inference_steps: int,
    self_replace_steps: float,
    cross_text_replace_steps: float,
    cross_image_replace_steps: float,
    ip_adapter_scale: float,
    lb_th1: float,
    lb_th2: float,
    lb_alpha: float,
    threshold: float,
    lb_lambda: float,
    per: float,
    gamma: float,
    source_guidance_scale: float,
    target_guidance_scale: float,
    start_blend: float,
    seed: int
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = EDDIMScheduler(
        eta=1,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )
    p2p_editor = P2PEditor(
        "unifiedinversion+p2p",
        device,
        sd_model_dir="runwayml/stable-diffusion-v1-5",
        ip_adapter_dir="h94/IP-Adapter",
        clip_model_dir="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        ip_adapter_scale=ip_adapter_scale
    )
    setup_seed(seed)
    _, eq_words, _ = get_diff_word_inds(original_prompt, editing_prompt, p2p_editor.clip_tokenizer)
    eq_params = {"words": eq_words, "values": [15.0] * len(eq_words)}
    edited_image, _ = p2p_editor(
        "unifiedinversion+p2p",
        image_path=image,
        image_mask=None,
        prompt_src=original_prompt,
        prompt_tar=editing_prompt,
        cross_text_replace_steps=cross_text_replace_steps,
        cross_image_replace_steps=cross_image_replace_steps,
        self_replace_steps=self_replace_steps,
        blend_word=(((blended_word1,), (blended_word2,))) if blended_word1 and blended_word2 else None,
        eq_params=eq_params,
        lb_th=(lb_th1, lb_th2),
        lb_alpha=lb_alpha,
        threshold=threshold,
        lb_lambda=lb_lambda,
        per=per,
        source_guidance_scale=source_guidance_scale,
        target_guidance_scale=target_guidance_scale,
        gamma=gamma,
        start_blend=start_blend,
    )
    return edited_image

# ─── Prepare inline Base64 logo ─────────────────────────────────────────────────

# Path to your local SVG logo
logo_path = "./gradio_app/locatedit.svg"
with open(logo_path, "rb") as f:
    logo_bytes = f.read()
logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
logo_data_uri = f"data:image/svg+xml;base64,{logo_b64}"

# ─── Gradio Interface ──────────────────────────────────────────────────────────

css = """
/* (optional) further tweaks */
.gradio-container .markdown, .gradio-container h2, .gradio-container p {
    margin-bottom: 1rem;
}
"""

with gr.Blocks(theme=Soft(), css=css) as demo:
    # HTML block for centered logo + title + subtitle
    gr.HTML(f"""
        <div style="text-align:center; margin-bottom:1rem;">
          <img src="{logo_data_uri}" alt="LOCATEdit Logo" width="180"/>
        </div>
        <h2 style="text-align:center; margin:0;">LOCATEdit Image Editor</h2>
        <p style="text-align:center; color:var(--block-text); margin-top:0.25rem;">
          Upload an image, tweak parameters, and see the edit instantly.
        </p>
    """)

    with gr.Tabs():
        with gr.Tab("Basic"):
            input_image     = gr.Image(type="filepath", label="Input Image")
            original_prompt = gr.Textbox(label="Original Prompt")
            editing_prompt  = gr.Textbox(label="Editing Prompt")
            with gr.Row():
                blended_word1 = gr.Textbox(label="Blend Word 1")
                blended_word2 = gr.Textbox(label="Blend Word 2")
            run_button      = gr.Button("Generate")

        with gr.Tab("Advanced"):
            num_inference_steps      = gr.Slider(1, 100, value=50, step=1,   label="Inference Steps", info="How many denoising steps")
            self_replace_steps       = gr.Slider(0.0, 1.0, value=0.6,          label="Self Replace Steps")
            cross_text_replace_steps = gr.Slider(0.0, 1.0, value=0.4,          label="Cross Text Replace Steps")
            cross_image_replace_steps= gr.Slider(0.0, 1.0, value=0.1,          label="Cross Image Replace Steps")
            ip_adapter_scale         = gr.Slider(0.0, 1.0, value=0.522,        label="IP Adapter Scale")
            lb_th1                   = gr.Slider(0.0, 1.0, value=0.4,       label="LB Threshold 1")
            lb_th2                   = gr.Slider(0.0, 1.0, value=0.5,          label="LB Threshold 2")
            lb_alpha                 = gr.Slider(0.0, 5.0, value=2.61,         label="LB Alpha")
            threshold                = gr.Slider(0.0, 1.0, value=0.4858,       label="Threshold")
            lb_lambda                = gr.Slider(0.0, 1.0, value=0.78,         label="LB Lambda")
            per                      = gr.Slider(0.0, 100.0, value=91.09,     label="PER")
            gamma                    = gr.Slider(0.0, 1.0, value=0.5,          label="Gamma")
            source_guidance_scale    = gr.Slider(0.0, 10.0, value=1.0,         label="Source Guidance Scale")
            target_guidance_scale    = gr.Slider(0.0, 10.0, value=5.0,         label="Target Guidance Scale")
            start_blend              = gr.Slider(0.0, 1.0, value=0.2,          label="Start Blend")
            seed                     = gr.Number(value=1234, label="Seed", precision=0)

    output_image = gr.Image(label="Edited Image", interactive=False)

    run_button.click(
        fn=process_image,
        inputs=[
            input_image,
            original_prompt,
            editing_prompt,
            blended_word1,
            blended_word2,
            num_inference_steps,
            self_replace_steps,
            cross_text_replace_steps,
            cross_image_replace_steps,
            ip_adapter_scale,
            lb_th1,
            lb_th2,
            lb_alpha,
            threshold,
            lb_lambda,
            per,
            gamma,
            source_guidance_scale,
            target_guidance_scale,
            start_blend,
            seed
        ],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()