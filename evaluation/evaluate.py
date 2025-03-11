import numpy as np


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "structure_distance_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "psnr_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "lpips_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "mse_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "ssim_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "structure_distance_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)