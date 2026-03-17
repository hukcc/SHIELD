"""SHIELD feature weighting, text-segment mapping and bias computation."""

import torch
import torch.nn.functional as F


def maximize_weight_difference(weights, gamma=1.0):
    """Stretch weight differences while keeping values in [0, 1]."""
    weights_min = weights.min()
    weights_max = weights.max()
    weights_range = weights_max - weights_min
    if weights_range < 1e-6:
        weights_norm = torch.zeros_like(weights)
    else:
        weights_norm = (weights - weights_min) / weights_range

    weights_adjusted = weights_norm.pow(gamma)

    adj_min = weights_adjusted.min()
    adj_max = weights_adjusted.max()
    adj_range = adj_max - adj_min
    if adj_range < 1e-6:
        return torch.zeros_like(weights_adjusted)
    return (weights_adjusted - adj_min) / adj_range


def compute_shield_image_weights(
    image_features,
    cap_tensor,
    the,
    gamma_gain,
    gain_per,
    bias_weight,
    bias_tensor,
):
    """Compute SHIELD-enhanced image features and the selected token indices."""
    image_features_normalized = F.normalize(
        F.adaptive_max_pool1d(image_features, 768), p=2, dim=-1
    ).float()
    text_features_normalized = F.normalize(cap_tensor, p=2, dim=-1).to(
        image_features_normalized.device
    )

    cosine_similarity = torch.matmul(
        image_features_normalized,
        text_features_normalized.transpose(-1, -2),
    ).squeeze()

    similarity_text_max = torch.max(cosine_similarity, dim=0)[0]
    top_k_indices = torch.nonzero(similarity_text_max > the).squeeze()
    similarity_img_w_selected = cosine_similarity[:, top_k_indices]
    similarity_img = torch.max(similarity_img_w_selected, dim=1)[0]
    patch_weight = maximize_weight_difference(similarity_img, gamma=gamma_gain)

    patch_weight_gain = torch.where(
        patch_weight <= gain_per,
        torch.tensor(0.0),
        patch_weight,
    ).view(1, -1, 1).half()

    enhanced_image_features = (
        image_features - bias_weight * bias_tensor + image_features * patch_weight_gain
    )

    return enhanced_image_features, top_k_indices


def map_text_segments(top_k_indices, cap_tensor, input_cap_tensor):
    """Map selected text segments into the input embedding sequence."""
    L_text = cap_tensor.size(1)
    L_other = input_cap_tensor.size(0)

    diffs = torch.diff(top_k_indices)
    breaks = torch.nonzero(diffs > 1).squeeze()

    if breaks.dim() == 0:
        break_points = [breaks.item()] + [len(top_k_indices) - 1]
    else:
        break_points = breaks.tolist() + [len(top_k_indices) - 1]

    segment_lengths = [
        break_points[i] - break_points[i - 1] if i > 0 else break_points[0] + 1
        for i in range(len(break_points))
    ]

    segments = torch.split(top_k_indices, segment_lengths)

    selected_other_elements = []
    for segment in segments:
        start_idx = segment[0]
        end_idx = segment[-1]

        proportion_start = start_idx.float() / L_text
        proportion_end = end_idx.float() / L_text

        start_in_other = (proportion_start * L_other).long()
        end_in_other = (proportion_end * L_other).long()

        selected_other_elements.append(input_cap_tensor[start_in_other : end_in_other + 1])

    return torch.cat(selected_other_elements, dim=0)


_bias_cache = {}


def get_bias(sample_num, images, vision_tower, cache_key=None):
    """Compute or retrieve cached bias features from random images."""
    global _bias_cache

    if cache_key is None:
        cache_key = "default"

    if cache_key not in _bias_cache:
        randn_images = torch.rand(
            (sample_num, images.size(1), images.size(2), images.size(3)),
            device=images.device,
        ).half()
        _bias_cache[cache_key] = vision_tower(randn_images).mean(dim=0).unsqueeze(0)

    return _bias_cache[cache_key]


def clear_bias_cache(cache_key=None):
    """Clear the cached bias features."""
    global _bias_cache

    if cache_key is None:
        _bias_cache = {}
    elif cache_key in _bias_cache:
        del _bias_cache[cache_key]
