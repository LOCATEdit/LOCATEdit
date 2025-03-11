import os
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random
import csv

from evaluation.evaluate import calculate_metric
from evaluation.matrics_calculator import MetricsCalculator
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from models.p2p.scheduler import EDDIMScheduler
from models.p2p.scheduler_dev import DDIMSchedulerDev
from models.p2p_editor import P2PEditor
from my_utils.utils import get_diff_word_inds

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))

    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1

    mask_array=mask_array.reshape(image_shape[0], image_shape[1])

    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--sd_model_dir', type=str)  # /path/to/diffusion_model_dir
    parser.add_argument('--ip_adapter_dir', type=str)  # /path/to/ip_adapter_dir
    parser.add_argument('--clip_model_dir', type=str)  # /path/to/clip_model_dir
    parser.add_argument('--output_dir', type=str, default="results_2/")

    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--self_replace_steps', type=float, default=0.6)
    parser.add_argument('--cross_text_replace_steps', type=float, default=0.4)
    parser.add_argument('--cross_image_replace_steps', type=float, default=0.1)

    parser.add_argument('--edit_method', type=str, default="unifiedinversion+p2p")

    parser.add_argument('--ip_adapter_scale', type=float, default=0.522871620472319)
    parser.add_argument('--lb_th', type=float, default=(0.2204, 0.5), nargs='+')
    parser.add_argument('--lb_alpha', type=float, default=2.61)
    parser.add_argument('--threshold', type=float, default=0.4858)
    parser.add_argument('--lb_lambda', type=float, default=0.7800247316325715)
    parser.add_argument('--per', type=float, default=91.0935948657488)
    parser.add_argument('--gamma', type=float, default=0.5)

    parser.add_argument("--source_guidance_scale", type=float, default=1.0)
    parser.add_argument("--target_guidance_scale", type=float, default=5.0)

    parser.add_argument("--scheduler", type=str, default='ddim')
    parser.add_argument("--start_blend", type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    anno = json.load(open(args.anno_file))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.scheduler == 'ddim':
        scheduler = EDDIMScheduler(
            eta = 1,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
    else:
        raise NotImplementedError

    p2p_editor=P2PEditor(args.edit_method, device, sd_model_dir=args.sd_model_dir, ip_adapter_dir=args.ip_adapter_dir,
                         clip_model_dir=args.clip_model_dir, scheduler=scheduler,
                         num_inference_steps=args.num_inference_steps, ip_adapter_scale=args.ip_adapter_scale)
    metrics_calculator = MetricsCalculator(device)
    for sid in anno:
        item = anno[sid]
        original_prompt = item['original_prompt'].replace('[', '').replace(']', '')
        editing_prompt = item['editing_prompt'].replace('[', '').replace(']', '')
            
        image_path = os.path.join(args.image_dir, item['image_path'])

        output_path = os.path.join(args.output_dir, item['image_path'])
        evaluate_result_path = os.path.join(args.output_dir, 'evaluation_result.csv')  # metrics
        iou_path = os.path.join(args.output_dir, 'ious.csv')  # mask iou
        if os.path.exists(output_path):
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        blended_word = item["blended_word"]
        blended_word = blended_word.split(" ") if blended_word != "" else []
        mask = mask_decode(item["mask"])

        print(f"editing image [{image_path}] with [{args.edit_method}]")
        setup_seed(args.seed)
        torch.cuda.empty_cache()
        _, eq_words, not_present_word = get_diff_word_inds(original_prompt, editing_prompt, p2p_editor.clip_tokenizer)
        eq_params = {"words": eq_words, "values": [15.]*len(eq_words)}
        edited_image, ious = p2p_editor(args.edit_method,
                                    image_path=image_path,
                                  image_mask=mask,
                                prompt_src=original_prompt,
                                prompt_tar=editing_prompt,
                                cross_text_replace_steps=args.cross_text_replace_steps,
                                cross_image_replace_steps=args.cross_image_replace_steps,
                                self_replace_steps=args.self_replace_steps,
                                blend_word=(((blended_word[0], ),
                                            (blended_word[1], ))) if len(blended_word) else None,
                                eq_params=eq_params,
                                lb_th=args.lb_th,
                                lb_alpha=args.lb_alpha,
                                threshold=args.threshold,
                                lb_lambda=args.lb_lambda,
                                per = args.per,
                                source_guidance_scale=args.source_guidance_scale,
                                target_guidance_scale=args.target_guidance_scale,
                                gamma=args.gamma,
                                start_blend=args.start_blend,
                                )
        edited_image.save(output_path)

        if ious is not None:
            with open(iou_path, 'a+', newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow([sid] + ious)

        src_image = Image.open(image_path)
        tgt_image = Image.open(output_path)
        mask = mask_decode(item["mask"])[:, :, np.newaxis].repeat([3], axis=2)
        # to avoid annotation errors in boundary
        mask[0,:]=1
        mask[-1,:]=1
        mask[:,0]=1
        mask[:,-1]=1

        if tgt_image.size[0] != tgt_image.size[1]:
            # to evaluate editing
            tgt_image = tgt_image.crop(
                (tgt_image.size[0] - 512, tgt_image.size[1] - 512, tgt_image.size[0], tgt_image.size[1]))
            # to evaluate reconstruction
        metrics = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
                   "ssim_unedit_part", "clip_similarity_source_image", "clip_similarity_target_image",
                   "clip_similarity_target_image_edit_part"]

        evaluation_result = [sid]
        for metric in metrics:
            result = calculate_metric(metrics_calculator, metric, src_image, tgt_image, mask, mask, original_prompt,
                                 editing_prompt)
            evaluation_result.append(result)
            print(f"{metric}: {result}")

        if evaluate_result_path is not None:
            with open(evaluate_result_path, 'a+', newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(evaluation_result)

        print(f"finish")

