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

### Benchmark Download ⬇️
<span id="benchmark-download"></span>

You can download the benchmark PIE-Bench (Prompt-driven Image Editing Benchmark). The data structure should be like:

```python
|-- data
    |-- annotation_images
        |-- 0_random_140
            |-- 000000000000.jpg
            |-- 000000000001.jpg
            |-- ...
        |-- 1_change_object_80
            |-- 1_artificial
                |-- 1_animal
                        |-- 111000000000.jpg
                        |-- 111000000001.jpg
                        |-- ...
                |-- 2_human
                |-- 3_indoor
                |-- 4_outdoor
            |-- 2_natural
                |-- ...
        |-- ...
    |-- mapping_file.json # the mapping file of PIE-Bench, contains editing text, blended word, and mask annotation
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

