import numpy as np
import argparse
import torch
import random
from models.p2p.scheduler import EDDIMScheduler
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
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--source_prompt', type=str, required=True)
    parser.add_argument('--target_prompt', type=str, required=True)
    parser.add_argument('--blend_words', type=str, default='')
    parser.add_argument('--output_path', type=str, default="result.jpg")

    parser.add_argument('--sd_model_dir', type=str)  # /path/to/diffusion_model_dir
    parser.add_argument('--ip_adapter_dir', type=str)  # /path/to/ip_adapter_dir
    parser.add_argument('--clip_model_dir', type=str)  # /path/to/clip_model_dir

    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--self_replace_steps', type=float, default=0.6)
    parser.add_argument('--cross_text_replace_steps', type=float, default=0.4)
    parser.add_argument('--cross_image_replace_steps', type=float, default=0.1)

    parser.add_argument('--edit_method', type=str, default="unifiedinversion+p2p")

    parser.add_argument('--ip_adapter_scale', type=float, default=0.3821)
    parser.add_argument('--lb_th', type=float, default=(0.5, 0.7987), nargs='+')
    parser.add_argument('--lb_alpha', type=int, default= 5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--lb_lambda', type=float, default=0.5)
    parser.add_argument('--per', type=float, default=92.0)
    parser.add_argument('--gamma', type=float, default=0.5)

    parser.add_argument("--source_guidance_scale", type=float, default=1.0)
    parser.add_argument("--target_guidance_scale", type=float, default=5.0)

    parser.add_argument("--scheduler", type=str, default='ddim')
    parser.add_argument("--start_blend", type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

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

    original_prompt = args.source_prompt
    editing_prompt = args.target_prompt
    image_path = args.image_path
    output_path = args.output_path

    blended_word = args.blend_words
    blended_word = blended_word.split(" ") if blended_word != "" else []

    print(f"editing image [{image_path}] with [{args.edit_method}]")
    setup_seed(args.seed)
    torch.cuda.empty_cache()
    _, eq_words, not_present_word = get_diff_word_inds(original_prompt, editing_prompt, p2p_editor.clip_tokenizer)
    eq_params = {"words": eq_words, "values": [15.]*len(eq_words)}
    edited_image, ious = p2p_editor(args.edit_method,
                                image_path=image_path,
                              image_mask=None,
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
    print(f"finish")

