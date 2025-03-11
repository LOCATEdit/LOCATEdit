import torch
from diffusers.utils.torch_utils import randn_tensor
from torch.nn import functional as F
from torchvision.transforms import ToPILImage

from models.p2p.attention_control import make_controller
from models.p2p.p2p_guidance_forward import unified_inversion_p2p_guidance_forward
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from my_utils.utils import load_512, latent2image, txt_draw, image2latent
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPTokenizer
import matplotlib.pyplot as plt

class P2PEditor:
    def __init__(self, method_list, device, sd_model_dir, ip_adapter_dir, clip_model_dir, scheduler,
                 num_inference_steps=50, ip_adapter_scale=1.0) -> None:
        self.device=device
        self.method_list=method_list
        self.num_inference_steps=num_inference_steps
        self.scheduler = scheduler

        self.ldm_stable = DiffusionPipeline.from_pretrained(sd_model_dir, scheduler=self.scheduler, torch_dtype=torch.float16).to(device)

        self.ldm_stable.load_ip_adapter(ip_adapter_dir, subfolder="models", weight_name="ip-adapter_sd15.bin")
        self.ldm_stable.set_ip_adapter_scale(ip_adapter_scale)

        self.clip_model = CLIPModel.from_pretrained(clip_model_dir).to(device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_dir)

        self.ldm_stable.scheduler.set_timesteps(self.num_inference_steps)
        
    def __call__(self, 
                edit_method,
                image_path,
                image_mask,
                prompt_src,
                prompt_tar,
                cross_text_replace_steps=0.4,
                cross_image_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=None,
                eq_params=None,
                is_replace_controller=False,
                lb_th=(0.3, 0.3),
                lb_alpha=0,
                threshold=0.5,
                lb_lambda=0.5,
                per = 95.0,
                source_guidance_scale=1.0,
                target_guidance_scale=1.0,
                gamma=0.5,
                start_blend=0.2,
                 ):
        if edit_method=="unifiedinversion+p2p":
            return self.edit_image_unifiedinversion(image_path=image_path, image_mask=image_mask, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        cross_text_replace_steps=cross_text_replace_steps, cross_image_replace_steps=cross_image_replace_steps, self_replace_steps=self_replace_steps,
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller, lb_th=lb_th, lb_alpha=lb_alpha, per=per, threshold=threshold, lb_lambda=lb_lambda,
                                        source_guidance_scale=source_guidance_scale, target_guidance_scale=target_guidance_scale, gamma=gamma, start_blend=start_blend)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")

    @staticmethod
    def percentile_threshold(embeddings, percentile=90):
        threshold = np.percentile(np.abs(embeddings), 100 - percentile)  # Compute threshold
        sparse_embeddings = np.where(np.abs(embeddings) >= threshold, embeddings, 0)
        return sparse_embeddings

    def get_ip_adapter_image_embeds(self, image_src, prompt_src, prompt_tar, per, editing_mask=None):
        dtype = next(self.ldm_stable.image_encoder.parameters()).dtype
        pixel_values = self.ldm_stable.feature_extractor(image_src, return_tensors="pt").pixel_values.to(self.device)

        image_embeds = self.clip_model.get_image_features(pixel_values)
        negative_image_embeds = torch.zeros_like(image_embeds)

        input_ids = self.clip_tokenizer(
            [prompt_src, prompt_tar],
            padding="max_length",
            max_length=self.ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        text_embeds = self.clip_model.get_text_features(input_ids)
        normed_text_embeds = text_embeds * image_embeds.norm() / text_embeds.norm(dim=-1, p=2, keepdim=True)
        global_offset = normed_text_embeds[[1]] - normed_text_embeds[[0]]
        global_offset = torch.from_numpy(P2PEditor.percentile_threshold(global_offset[0].data.cpu().numpy(), per)).to(image_embeds.device)

        image_embeds_edit = image_embeds + global_offset
        
        ip_adapter_image_embeds = torch.stack([negative_image_embeds, image_embeds], dim=0).to(dtype=dtype)
        ip_adapter_image_embeds_edit = torch.stack([negative_image_embeds, image_embeds_edit], dim=0).to(dtype=dtype)
        return ip_adapter_image_embeds, ip_adapter_image_embeds_edit



    def edit_image_unifiedinversion(
        self,
        image_path,
        image_mask,
        prompt_src,
        prompt_tar,
        cross_text_replace_steps=0.4,
        cross_image_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        lb_th=(0.3, 0.3),
        lb_alpha=0,
        per=95.0,
        threshold=0.5,
        lb_lambda=0.5,
        source_guidance_scale=1.0,
        target_guidance_scale=1.0,
        gamma=0.5,
        start_blend=0.2
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]
        ip_adapter_image_embeds, ip_adapter_image_embeds_edit = self.get_ip_adapter_image_embeds(
            image_src=Image.open(image_path), prompt_src=prompt_src, prompt_tar=prompt_tar, per = per, editing_mask = image_mask
        )
        image_embeds = torch.cat([ip_adapter_image_embeds, ip_adapter_image_embeds_edit], dim=0)

        x_0 = image2latent(self.ldm_stable.vae, image_gt)
        timestep = self.scheduler.timesteps[:1]
        init_noise = randn_tensor(x_0.shape, device=self.device, dtype=self.ldm_stable.dtype)
        x_T = self.ldm_stable.scheduler.add_noise(x_0, init_noise, timestep)

        ########## edit ##########
        cross_text_replace_steps = {
            'default_': cross_text_replace_steps,
        }
        
        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    image_mask=image_mask,
                                    is_replace_controller=is_replace_controller,
                                    cross_text_replace_steps=cross_text_replace_steps,
                                    cross_image_replace_steps=cross_image_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_inference_steps=self.num_inference_steps,
                                    device=self.device,
                                    lb_th=lb_th,
                                    lb_alpha=lb_alpha,
                                    threshold=threshold,
                                    lb_lambda=lb_lambda,
                                    gamma=gamma,
                                     start_blend=start_blend)
        
        latents, _ = unified_inversion_p2p_guidance_forward(model=self.ldm_stable,
                                                           prompt=prompts,
                                                           image_embeds=image_embeds,
                                                           controller=controller,
                                                           x_0=x_0,
                                                           x_T=x_T,
                                                           num_inference_steps=self.num_inference_steps,
                                                           source_guidance_scale=source_guidance_scale,
                                                           target_guidance_scale=target_guidance_scale,
                                                           generator=None,
                                                           )
        
    
        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        if controller.local_blend is not None:
            blend_word = ' '.join((blend_word[0][0], blend_word[1][0]))
            image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}\nblend word: {blend_word}")
            latent_masks = torch.cat([mask[1:] for mask in controller.local_blend.mask_store], dim=0)
            
            image_masks = F.interpolate(latent_masks, size=(512, 512))
            image_masks = image_masks.permute(2, 0, 3, 1).contiguous().view(512, -1).data.cpu().numpy()
            image_masks = np.uint8(image_masks[:, :, None].repeat(3, 2) * 255)
            out_image = np.concatenate([image_masks, image_instruct, image_gt, images[0], images[-1]], axis=1)
            ious = controller.local_blend.ious
            ious = ious if len(ious) > 0 else None
        else:
            image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
            out_image = np.concatenate([image_instruct, image_gt, images[0], images[-1]], axis=1)
            ious = None
        out_image = Image.fromarray(out_image)
        return out_image, ious