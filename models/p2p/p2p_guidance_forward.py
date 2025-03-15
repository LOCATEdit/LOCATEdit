import torch
from models.p2p.attention_control import register_attention_control
from my_utils.utils import init_latent, latent2image


@torch.no_grad()
def unified_inversion_p2p_guidance_forward(
        model,
        prompt,
        image_embeds,
        controller,
        x_0,
        x_T,
        num_inference_steps: int = 50,
        source_guidance_scale=7.5,
        target_guidance_scale=7.5,
        generator=None,
        low_resource=False,
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512

    input_ids = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(model.device)
    text_embeddings = model.text_encoder(input_ids)[0]
    max_length = input_ids.shape[-1]

    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(x_T, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings])

        if low_resource:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[:2])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[2:])["sample"]
        else:
            added_cond_kwargs = {'image_embeds': [image_embeds]}
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

        source_noise_pred = noise_pred_uncond[[0]] + source_guidance_scale * (noise_prediction_text[[0]] - noise_pred_uncond[[0]])
        target_noise_pred = noise_pred_uncond[[1]] + target_guidance_scale * (noise_prediction_text[[1]] - noise_pred_uncond[[1]])
        noise_pred = torch.cat([source_noise_pred, target_noise_pred], dim=0)
        editing_mask = None
        if controller.local_blend is not None:
            editing_mask = controller.local_blend.get_mask(controller.local_blend.th[1])
            if editing_mask is not None:
                # print(f"editing mask: {editing_mask.shape}")
                editing_mask = editing_mask[:1] + editing_mask[1:]
        latents = model.scheduler.edit_step(
            model_output=noise_pred,
            timestep=t,
            source_original_sample=x_0,
            sample=latents,
            editing_mask=editing_mask,
        )
        latents = controller.step_callback(latents)
    return latents, latent