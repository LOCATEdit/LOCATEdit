from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
import torch
from typing import Optional
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor


def compute_mu(k_t, alpha_prod_t, alpha_prod_prev_t):
    a = (alpha_prod_prev_t * (1 - alpha_prod_t)).sqrt() / (alpha_prod_t * (1 - alpha_prod_prev_t)).sqrt()
    b = 1 - (k_t * (1 - a) + a)**2
    c = b * (1 - alpha_prod_t) / (1 - alpha_prod_t/alpha_prod_prev_t)
    return c.sqrt()

def compute_local_gradient(sample):
    # sample shape: (B, C, H, W)
    # Convert to grayscale by averaging channels.
    gray = sample.mean(dim=1, keepdim=True)
    # Define simple Sobel filters.
    sobel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = (grad_x**2 + grad_y**2).sqrt()  # shape (B, 1, H, W)
    return grad_mag

class EDDIMScheduler(DDIMScheduler):
    def __init__(self, eta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = eta

    def edit_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        source_original_sample: torch.FloatTensor,
        sample: torch.FloatTensor,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        assert self.config.prediction_type == "epsilon"
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        if source_original_sample is not None:
            pred_original_sample = source_original_sample + pred_original_sample - pred_original_sample[:1]

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = self.eta * variance ** (0.5)
        b_t = ((1 - alpha_prod_t_prev - std_dev_t ** 2) / (1 - alpha_prod_t)).sqrt()
        prev_sample = (alpha_prod_t_prev.sqrt() - alpha_prod_t.sqrt() * b_t) * pred_original_sample + b_t * sample
        return prev_sample