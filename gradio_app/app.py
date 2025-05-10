import os
import sys

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
from typing import Optional, Union, List

def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    return mask_array

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
    # Initialize device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Initialize scheduler
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
    
    # Setup seed
    setup_seed(seed)
    
    # Get equivalent words
    _, eq_words, _ = get_diff_word_inds(original_prompt, editing_prompt, p2p_editor.clip_tokenizer)
    eq_params = {"words": eq_words, "values": [15.]*len(eq_words)}
    
    # Process image
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

# Define interface with explicit types
with gr.Blocks(title="LOCATEdit Image Editor") as demo:
    gr.Markdown("# LOCATEdit Image Editor")
    gr.Markdown("Upload an image and edit it with prompts and parameters.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Image")
            original_prompt = gr.Textbox(label="Original Prompt")
            editing_prompt = gr.Textbox(label="Editing Prompt")
            blended_word1 = gr.Textbox(label="Blended Word 1")
            blended_word2 = gr.Textbox(label="Blended Word 2")
            
            with gr.Accordion("Advanced Parameters", open=False):
                num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Number of Inference Steps")
                self_replace_steps = gr.Slider(0.0, 1.0, value=0.6, label="Self Replace Steps")
                cross_text_replace_steps = gr.Slider(0.0, 1.0, value=0.4, label="Cross Text Replace Steps")
                cross_image_replace_steps = gr.Slider(0.0, 1.0, value=0.1, label="Cross Image Replace Steps")
                ip_adapter_scale = gr.Slider(0.0, 1.0, value=0.522, label="IP Adapter Scale")
                lb_th1 = gr.Slider(0.0, 1.0, value=0.2204, label="LB Threshold 1")
                lb_th2 = gr.Slider(0.0, 1.0, value=0.5, label="LB Threshold 2")
                lb_alpha = gr.Slider(0.0, 5.0, value=2.61, label="LB Alpha")
                threshold = gr.Slider(0.0, 1.0, value=0.4858, label="Threshold")
                lb_lambda = gr.Slider(0.0, 1.0, value=0.78, label="LB Lambda")
                per = gr.Slider(0.0, 100.0, value=91.09, label="PER")
                gamma = gr.Slider(0.0, 1.0, value=0.5, label="Gamma")
                source_guidance_scale = gr.Slider(0.0, 10.0, value=1.0, label="Source Guidance Scale")
                target_guidance_scale = gr.Slider(0.0, 10.0, value=5.0, label="Target Guidance Scale")
                start_blend = gr.Slider(0.0, 1.0, value=0.2, label="Start Blend")
                seed = gr.Number(value=1234, label="Seed", precision=0)
            
            run_button = gr.Button("Generate")
            
        with gr.Column():
            output_image = gr.Image(label="Edited Image")
    
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