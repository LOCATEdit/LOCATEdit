# LOCATEdit: Graph Laplacian Optimized Cross Attention for Localized Text-Guided Image Editing
## 🚀 Getting Started
<span id="getting-started"></span>

### Environment Requirement 🌍
<span id="environment-requirement"></span>

```shell
conda create -n locatedit python=3.9
conda activate locatedit
pip install -r requirements.txt
```

## 🏃🏼 Running Scripts
<span id="running-scripts"></span>

### Inference and Evaluation📜
<span id="inference"></span>

**Run the Benchmark**

Run LOCATEdit on PIE-Bench:

```shell
python run_editing_locatedit.py --anno_file data/mapping_file.json --image_dir data/annotation_images --sd_model_dir runwayml/stable-diffusion-v1-5 --ip_adapter_dir h94/IP-Adapter --clip_model_dir laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

You can specify --output_dir for output path. 


**Run Any Image**

You can process your own images and editing prompts.
```shell
python run_editing_locatedit_one_image.py --image_path /path/to/source/image --source_prompt "source prompt" --target_prompt "target prompt" --sd_model_dir runwayml/stable-diffusion-v1-5 --ip_adapter_dir h94/IP-Adapter --clip_model_dir laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

You can specify --output_path for output path. 



## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified on the basis of PnP Inversion, InfEdit and P2P.

