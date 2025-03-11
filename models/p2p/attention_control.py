import inspect
from typing import Optional
import numpy as np
import math
import torch
import torch.nn.functional as F
import abc
import os
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate

from my_utils.utils import get_word_inds, get_time_words_attention_alpha, get_diff_word_inds
from models.p2p import seq_aligner
from diffusers.models.attention_processor import Attention, logger
import matplotlib.pyplot as plt
from PIL import Image

MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)
LOW_RESOURCE = False

ATYPE_SELF = 0
ATYPE_CROSS_TEXT = 1
ATYPE_CROSS_IMAGE = 2


def scaled_dot_product_attention(query, key, value=None, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    if value is None:
        return attn_weight
    return attn_weight @ value


def attn_processor2_0_forward_wrapper(self, place_in_unet, controller):

    def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            **cross_attention_kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        if cross_attention_kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if self.spatial_norm is not None:
            temb = cross_attention_kwargs.get('temb', None)
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        attn_weight = scaled_dot_product_attention(
            query, key, value=None, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn_weight = controller(attn_weight, atype=ATYPE_SELF, place_in_unet=place_in_unet)
        hidden_states = attn_weight @ value

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states
    return forward


def ip_adapter_attn_processor2_0_forward_wrapper(self, place_in_unet, controller):

    def forward(
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ):
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`.This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to supress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.processor.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if self.spatial_norm is not None:
            temb = cross_attention_kwargs.get('temb', None)
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        attn_weight = scaled_dot_product_attention(
            query, key, value=None, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn_weight = controller(attn_weight, atype=ATYPE_CROSS_TEXT, place_in_unet=place_in_unet)
        hidden_states = attn_weight @ value

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        ip_adapter_masks = cross_attention_kwargs.get('ip_adapter_masks', None)
        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, torch.Tensor) or ip_adapter_masks.ndim != 4:
                raise ValueError(
                    " ip_adapter_mask should be a tensor with shape [num_ip_adapter, 1, height, width]."
                    " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                )
            if len(ip_adapter_masks) != len(self.processor.scale):
                raise ValueError(
                    f"Number of ip_adapter_masks ({len(ip_adapter_masks)}) must match number of IP-Adapters ({len(self.processor.scale)})"
                )
        else:
            ip_adapter_masks = [None] * len(self.processor.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.processor.scale, self.processor.to_k_ip, self.processor.to_v_ip, ip_adapter_masks
        ):
            ip_key = to_k_ip(current_ip_hidden_states)
            ip_value = to_v_ip(current_ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            # current_ip_hidden_states = F.scaled_dot_product_attention(
            #     query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            # )
            current_ip_attn_weight = scaled_dot_product_attention(
                query, ip_key, value=None, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            current_ip_attn_weight = controller(current_ip_attn_weight, atype=ATYPE_CROSS_IMAGE, place_in_unet=place_in_unet)
            current_ip_hidden_states = current_ip_attn_weight @ ip_value

            current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )
            current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

            if mask is not None:
                mask_downsample = IPAdapterMaskProcessor.downsample(
                    mask, batch_size, current_ip_hidden_states.shape[1], current_ip_hidden_states.shape[2]
                )

                mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)

                current_ip_hidden_states = current_ip_hidden_states * mask_downsample

            hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

    return forward


def register_attention_control(model, controller):
    class DummyController:

        def __call__(self, *args, **kwargs):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    
    ##VIMAEDIT
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            if net_.processor.__class__.__name__ == 'AttnProcessor2_0':
                net_.forward = attn_processor2_0_forward_wrapper(net_, place_in_unet, controller)
                return count + 1
            elif net_.processor.__class__.__name__ == 'IPAdapterAttnProcessor2_0':
                net_.forward = ip_adapter_attn_processor2_0_forward_wrapper(net_, place_in_unet, controller)
                return count + 2
            else:
                raise NotImplementedError
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_equalizer(text, word_select, values, tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        # print(word, val)
        inds = get_word_inds(text, word, tokenizer)
        # print(inds)
        equalizer[:, inds] = val
    return equalizer

class LocalBlend:
    
    def get_mask_ours(self, step_store):
        if self.counter >= self.start_blend:
            self_attn = torch.stack(
                step_store['down_self'][2:4] + step_store['up_self'][3:6],
                dim=0
            ).mean((0, 2)).to(self.device)
            

            cross_attn = step_store["down_cross_text"][4:6] + step_store["up_cross_text"][:3]
            ca_sz = int(cross_attn[0].shape[-2] ** 0.5)
            cross_attn = [item.reshape(self.alpha_layers.shape[0], -1, 1, ca_sz, ca_sz, MAX_NUM_WORDS)
                        for item in cross_attn]
            cross_attn = torch.cat(cross_attn, dim=1).to(self.device)
            cross_attn = (cross_attn * self.alpha_layers).sum(-1).mean(1).to(dtype=cross_attn.dtype)
            maps = F.interpolate(cross_attn, (32, 32)).view(2, 1024, 1)

            fm = maps.clone().view(2, 32, 32)
            B, N, _ = maps.shape
            dtype = maps.dtype
            self_attn = self_attn.to(dtype)
            Q_sym = (self_attn + self_attn.transpose(1, 2)) / 2.0
            degree = torch.sum(Q_sym, dim=2)
            D = torch.diag_embed(degree)

            lambda_val = torch.tensor(self.lb_lambda, device=self.device, dtype=maps.dtype)
            L = D - Q_sym

            m0 = maps.squeeze(-1)
            w = torch.sigmoid(self.lb_alpha * (m0 - self.threshold))
            D_w = torch.diag_embed(w**2)

            A_final = D_w + lambda_val * L
            m0_expanded = m0.unsqueeze(-1) 
            m_final = D_w @ m0_expanded
            A_final_float = A_final.to(torch.float32)
            m0_float = m_final.to(torch.float32)
            refined_maps = torch.linalg.solve(A_final_float, m0_float)
            maps = refined_maps.view(B, 1, 32, 32)
            
            self.map = (self.map + maps).to(dtype=cross_attn.dtype)
        
            if self.counter % int(0.2 * self.num_inference_steps) == 0:
                mask_not_thresholded = LocalBlend.get_mask_inter(self.map)
                self.mask_store_not_thresholded.append(mask_not_thresholded.data)
        
        self.mask = self.get_mask(self.th[0])

    def get_mask(self, th):
        mask = None
        if self.counter >= self.start_blend:
            mask = F.interpolate(self.map, size=LATENT_SIZE)
            mask_min = mask.min(2, keepdims=True)[0].min(3, keepdims=True)[0]
            mask_max = mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
            mask = (mask - mask_min) / (mask_max - mask_min)
            mask = mask.gt(th)
        return mask
    
    def modulate(self, m):
        squared_mask = torch.pow(m, 2)
        mask_min = squared_mask.min(2, keepdims=True)[0].min(3, keepdims=True)[0]
        mask_max = squared_mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = (squared_mask - mask_min) / (mask_max - mask_min)
        return mask
        
    @staticmethod
    def get_mask_inter(mask):
        mask = F.interpolate(mask, size=LATENT_SIZE)
        mask_min = mask.min(2, keepdims=True)[0].min(3, keepdims=True)[0]
        mask_max = mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        
        if (mask.max() > 1).any():
            mask = (mask - mask_min) / (mask_max - mask_min)
        return mask
    
    def update_map(self, attention_store, step_store):
        self.get_mask_ours(step_store)

    def __call__(self, x_t, attention_store, step_store):
        mask = self.mask
        if mask is not None:
            mask = mask[:1] + mask
            mask = mask.to(dtype=x_t.dtype)
            if self.counter % int(0.2 * self.num_inference_steps) == 0:
                self.mask_store.append(mask.data)
            x_t = x_t[:1] + mask * (x_t - x_t[:1])

            if self.mask_gt is not None and self.mask_gt.sum() > 0:
                mask_pred = F.interpolate(mask[1:], size=(512, 512))[0, 0].data.cpu().numpy()
                iou = (mask_pred.astype(bool) * self.mask_gt.astype(bool)).sum() / (
                        mask_pred.astype(bool) + self.mask_gt.astype(bool)).sum()
                self.ious.append(iou)
        return x_t

    def __init__(self, prompts, words, substruct_words=None, start_blend=0.2, th=(.3, .3),
                 tokenizer=None, device="cuda",num_inference_steps=50, lb_alpha=0, threshold=0.5, lb_lambda=0.5, gamma=0.5, mask_gt=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        # ind, _, _ = get_diff_word_inds(prompts[0], prompts[1], tokenizer)
        # alpha_layers[1, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.num_inference_steps = num_inference_steps
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_inference_steps)
        self.counter = 0 
        self.th=th
        self.lb_alpha = lb_alpha
        self.lb_lambda = lb_lambda
        self.threshold = threshold
        self.gamma = gamma
        self.device = device
        self.map = torch.zeros(2, 1, 32, 32, device=self.device)
        self.mask_store = []
        self.mask_store_not_thresholded = []
        self.mask_gt = mask_gt
        self.ious = []
        self.img_name = prompts[0].replace(" ", "_")


class EmptyControl:

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, atype, place_in_unet):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, atype, place_in_unet):
        raise NotImplementedError

    def __call__(self, attn, atype, place_in_unet):
        # attn: [batch_size, num_heads, query_size, key_size]
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, atype, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], atype, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_self": [],  "mid_self": [],  "up_self": [],
                "down_cross_text": [], "mid_cross_text": [], "up_cross_text": [],
                "down_cross_image": [], "mid_cross_image": [], "up_cross_image": []}

    def forward(self, attn, atype, place_in_unet):
        if atype == ATYPE_SELF:
            key = 'self'
        elif atype == ATYPE_CROSS_TEXT:
            key = 'cross_text'
        elif atype == ATYPE_CROSS_IMAGE:
            key = 'cross_image'
        else:
            raise NotImplementedError
        key = f"{place_in_unet}_{key}"
        if self.cur_att_layer == 0:
            self.step_store = self.get_empty_store()
        # if attn.shape[2] <= 32 ** 2:  # avoid memory overhead
        self.step_store[key].append(attn.data)
        # self.step_store[key].append(attn.data.cpu())
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.step_store)
        return x_t

    def between_steps(self):
        if self.local_blend is not None:
            self.local_blend.counter += 1
            self.local_blend.update_map(self.attention_store, self.step_store)
        super().between_steps()

    def replace_self_attention(self, attn_base, attn_replace, place_in_unet):
        if attn_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(attn_replace.shape[0], *attn_base.shape)
            if self.inject_self and (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
                return attn_base
            return attn_replace
        else:
            return attn_replace

    @abc.abstractmethod
    def replace_cross_text_attention(self, attn_base, attn_replace):
        raise NotImplementedError

    def replace_cross_image_attention(self, attn_base, attn_replace, place_in_unet):
        if attn_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(attn_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return attn_replace

    def forward(self, attn, atype, place_in_unet):
        super(AttentionControlEdit, self).forward(attn, atype, place_in_unet)
        attn_base, attn_replace = attn[0], attn[1:]
        if atype == ATYPE_SELF:
            attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
        elif atype == ATYPE_CROSS_TEXT:
            alpha_words = self.cross_replace_alpha[self.cur_step]
            attn_repalce_new = self.replace_cross_text_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace
            attn[1:] = attn_repalce_new
        elif (atype == ATYPE_CROSS_IMAGE) and (self.num_cross_image_replace[0] <= self.cur_step < self.num_cross_image_replace[1]):
            attn[1:] = self.replace_cross_image_attention(attn_base, attn_replace, place_in_unet)
        return attn
    
    def __init__(self, 
                 prompts, 
                 num_steps,
                 cross_text_replace_steps,
                 cross_image_replace_steps,
                 self_replace_steps,
                 local_blend,
                 tokenizer=None,
                 image_mask=None,
                 device="cuda"):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_text_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        if type(cross_image_replace_steps) is float:
            cross_image_replace_steps = 0, cross_image_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.num_cross_image_replace = int(num_steps * cross_image_replace_steps[0]), int(num_steps * cross_image_replace_steps[1])
        self.local_blend = local_blend
        self.image_mask = image_mask
        self.device = device
        self.inject_self = True


class AttentionReplace(AttentionControlEdit):

    def replace_cross_text_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps, cross_text_replace_steps, cross_image_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda"):
        super(AttentionReplace, self).__init__(prompts=prompts, 
                                              num_steps=num_steps, 
                                              cross_text_replace_steps=cross_text_replace_steps,
                                              cross_image_replace_steps=cross_image_replace_steps,
                                              self_replace_steps=self_replace_steps,
                                              local_blend=local_blend,
                                              device=device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_text_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps, cross_text_replace_steps, cross_image_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda",image_mask=None):
        super(AttentionRefine, self).__init__(prompts=prompts, 
                                              num_steps=num_steps, 
                                              cross_text_replace_steps=cross_text_replace_steps,
                                              cross_image_replace_steps=cross_image_replace_steps,
                                              self_replace_steps=self_replace_steps,
                                              local_blend=local_blend,
                                              device=device,
                                              image_mask=image_mask)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_text_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_text_attention(attn_base, att_replace)
        attn_replace = attn_base[:, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, 
                 prompts, 
                 num_steps, 
                 cross_text_replace_steps,
                 cross_image_replace_steps,
                 self_replace_steps,
                 equalizer,
                 local_blend = None, 
                 controller = None,
                 image_mask=None,
                 device="cuda"):
        super(AttentionReweight, self).__init__(prompts=prompts, 
                                                num_steps=num_steps, 
                                                cross_text_replace_steps=cross_text_replace_steps,
                                                cross_image_replace_steps=cross_image_replace_steps,
                                                self_replace_steps=self_replace_steps,
                                                local_blend=local_blend,
                                                image_mask=image_mask,
                                                device=device)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def make_controller(pipeline, 
                    prompts,
                    is_replace_controller, 
                    cross_text_replace_steps,
                    cross_image_replace_steps,
                    self_replace_steps,
                    blend_words=None, 
                    equilizer_params=None, 
                    num_inference_steps=50,
                    image_mask=None,
                    device="cuda",
                    lb_th=(0.3, 0.3),
                    lb_alpha=0,
                    threshold=0.5,
                    lb_lambda=0.5,
                    gamma=0.5,
                    start_blend=0.2) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer=pipeline.tokenizer, device=device,num_inference_steps=num_inference_steps,
                        th=lb_th, lb_alpha=lb_alpha, threshold=threshold, lb_lambda=lb_lambda, gamma=gamma, start_blend=start_blend, mask_gt=image_mask)
    if is_replace_controller:
        controller = AttentionReplace(prompts, 
                                      num_inference_steps,
                                      cross_text_replace_steps=cross_text_replace_steps,
                                      cross_image_replace_steps=cross_image_replace_steps,
                                      self_replace_steps=self_replace_steps,
                                      local_blend=lb,
                                      tokenizer=pipeline.tokenizer)
    else:
        controller = AttentionRefine(prompts, 
                                     num_inference_steps,
                                     image_mask=image_mask,
                                     cross_text_replace_steps=cross_text_replace_steps,
                                     cross_image_replace_steps=cross_image_replace_steps,
                                     self_replace_steps=self_replace_steps, 
                                     local_blend=lb,
                                     tokenizer=pipeline.tokenizer)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], 
                           equilizer_params["words"], 
                           equilizer_params["values"], 
                           tokenizer=pipeline.tokenizer)

        controller = AttentionReweight(prompts, 
                                       num_inference_steps,
                                       image_mask=image_mask,
                                       cross_text_replace_steps=cross_text_replace_steps,
                                       cross_image_replace_steps=cross_image_replace_steps,
                                       self_replace_steps=self_replace_steps, 
                                       equalizer=eq, 
                                       local_blend=lb, 
                                       controller=controller)
    return controller
