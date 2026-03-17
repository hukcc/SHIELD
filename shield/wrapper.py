"""SHIELD wrapper -- non-invasive monkey-patching for LLaVA models.

Usage::

    import shield

    model = shield.wrap(model, tokenizer, caption_file="captions.jsonl")
    shield_kw = model.shield_prepare(image, image_tensor, image_file)
    output = model.generate(input_ids, **shield_kw, do_sample=True, max_new_tokens=1024)
"""

import types
from typing import List, Optional, Tuple, Union

import torch

from .attack import perform_clip_attack
from .caption import (
    find_text_by_image,
    load_captions,
    process_caption,
)
from .clip_utils import clear_clip_cache, get_clip_text_features, load_clip_model
from .feature import (
    clear_bias_cache,
    compute_shield_image_weights,
    get_bias,
    map_text_segments,
)
from .sampling import enable_shield_sampling

# LLaVA constants -- try importing from the installed package first,
# fall back to well-known defaults so shield works without a LLaVA checkout.
try:
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200
    IGNORE_INDEX = -100

SHIELD_KWARGS = frozenset({
    "cap_tensor",
    "the",
    "gamma_gain",
    "gamma_reduce",
    "input_cap_ids",
    "gain_per",
    "reduce_per",
    "use_cd_branch",
    "bias_weight",
    "bias_sample_num",
    "images_cd",
    "cd_alpha",
    "cd_beta",
})

DEFAULT_PARAMS = {
    "cd_alpha": 1.0,
    "cd_beta": 0.1,
    "the": 0.0,
    "gamma_gain": 1.0,
    "gamma_reduce": 0.0,
    "gain_per": 0.0,
    "reduce_per": 0.0,
    "bias_weight": 0.0,
    "bias_sample_num": 32,
    "epsilon": 0.14,
    "num_steps": 30,
    "c": 12,
    "lr": 0.02,
    "attack_type": "cw",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wrap(model, tokenizer, caption_file=None, **kwargs):
    """Wrap a LLaVA model with SHIELD capabilities.

    After calling this function the model gains two abilities:

    * ``model.shield_prepare(image, image_tensor, image_file)``
      -- returns a dict of kwargs ready to be unpacked into ``model.generate()``.
    * Automatic monkey-patching of ``forward``,
      ``prepare_inputs_for_generation`` and the contrastive-decoding sampler.

    Parameters
    ----------
    model : nn.Module
        A ``LlavaLlamaForCausalLM`` (or API-compatible) model instance.
    tokenizer : PreTrainedTokenizer
        The tokenizer paired with *model*.
    caption_file : str, optional
        Path to a JSONL caption file.  Can also be passed per-call via
        ``model.shield_prepare(..., caption_file=...)``.
    **kwargs
        Default SHIELD hyper-parameters (``cd_alpha``, ``the``,
        ``gamma_gain``, ``epsilon``, ``num_steps``, etc.).  Any value not
        provided here falls back to ``DEFAULT_PARAMS``.

    Returns
    -------
    model
        The same *model* instance, now patched.
    """
    enable_shield_sampling()

    clip_model, clip_processor = load_clip_model()

    captions = load_captions(caption_file) if caption_file else None

    defaults = {**DEFAULT_PARAMS, **kwargs}

    model._shield_config = {
        "tokenizer": tokenizer,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "captions": captions,
        "defaults": defaults,
    }

    _patch_model(model)

    model.shield_prepare = types.MethodType(_shield_prepare, model)

    return model


# ---------------------------------------------------------------------------
# shield_prepare -- per-image convenience method
# ---------------------------------------------------------------------------

def _shield_prepare(self, image, image_tensor, image_file, use_cd=True, **overrides):
    """Prepare all SHIELD inputs for a single image.

    Returns a ``dict`` that can be unpacked directly into
    ``model.generate(input_ids, **shield_kw, ...)``.

    Parameters
    ----------
    image : PIL.Image
        The original PIL image (needed by the CLIP attack).
    image_tensor : torch.Tensor
        The image preprocessed by the LLaVA image processor.
    image_file : str
        Filename used to look up the caption.
    use_cd : bool
        When *False* the adversarial attack is skipped and ``images_cd``
        is set to ``None`` (feature enhancement still runs).
    **overrides
        Per-call hyper-parameter overrides.
    """
    cfg = self._shield_config
    params = {**cfg["defaults"], **overrides}

    captions = overrides.pop("captions", None) or cfg["captions"]
    if "caption_file" in overrides and captions is None:
        captions = load_captions(overrides["caption_file"])

    caption_raw = find_text_by_image(image_file, captions)
    text_in, text_ad = process_caption(caption_raw)

    input_cap_ids = (
        cfg["tokenizer"](text_in, return_tensors="pt")["input_ids"]
        .cuda()
    )
    cap_tensor = get_clip_text_features(
        text_in, cfg["clip_model"], cfg["clip_processor"]
    )

    if use_cd:
        image_tensor_cd = perform_clip_attack(
            image,
            image_tensor,
            text_ad,
            epsilon=params["epsilon"],
            num_steps=int(params["num_steps"]),
            c=params["c"],
            lr=params["lr"],
            attack_type=params.get("attack_type", "cw"),
            clip_model=cfg["clip_model"],
            clip_processor=cfg["clip_processor"],
        )
        images_cd = image_tensor_cd.unsqueeze(0).half().cuda()
    else:
        images_cd = None

    return {
        "images": image_tensor.unsqueeze(0).half().cuda(),
        "images_cd": images_cd,
        "cd_alpha": params["cd_alpha"],
        "cd_beta": params["cd_beta"],
        "cap_tensor": cap_tensor,
        "input_cap_ids": input_cap_ids,
        "the": params["the"],
        "gamma_gain": params["gamma_gain"],
        "gamma_reduce": params["gamma_reduce"],
        "gain_per": params["gain_per"],
        "reduce_per": params["reduce_per"],
        "bias_weight": params["bias_weight"],
        "bias_sample_num": params["bias_sample_num"],
    }


# ---------------------------------------------------------------------------
# Monkey-patching helpers
# ---------------------------------------------------------------------------

def _patch_model(model):
    """Apply all SHIELD patches to *model* in-place."""
    model._shield_original_forward = type(model).forward

    model.forward = types.MethodType(_patched_forward, model)
    model.prepare_inputs_for_generation = types.MethodType(
        _patched_prepare_inputs_for_generation, model
    )
    model.prepare_inputs_for_generation_cd = types.MethodType(
        _prepare_inputs_for_generation_cd, model
    )


# ---------------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------------

def _patched_forward(
    self,
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    images=None,
    # CD params (consumed by the patched sampler, not by forward itself)
    images_cd=None,
    cd_beta=None,
    cd_alpha=None,
    return_dict=None,
    # SHIELD params
    cap_tensor=None,
    the=None,
    gamma_gain=None,
    gamma_reduce=None,
    input_cap_ids=None,
    gain_per=None,
    reduce_per=None,
    use_cd_branch=None,
    bias_weight=None,
    bias_sample_num=None,
    **kwargs,
):
    """Patched ``forward`` that routes to SHIELD multimodal processing."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from torch.nn import CrossEntropyLoss

    if images is not None and inputs_embeds is None:
        if cap_tensor is not None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                _shield_prepare_multimodal(
                    self,
                    input_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    cap_tensor=cap_tensor,
                    the=the,
                    gamma_gain=gamma_gain,
                    gamma_reduce=gamma_reduce,
                    input_cap_ids=input_cap_ids,
                    gain_per=gain_per,
                    reduce_per=reduce_per,
                    use_cd_branch=use_cd_branch,
                    bias_weight=bias_weight,
                    bias_sample_num=bias_sample_num,
                )
            )
        else:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask, past_key_values, labels, images
                )
            )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ---------------------------------------------------------------------------
# Patched prepare_inputs_for_generation (main branch)
# ---------------------------------------------------------------------------

def _patched_prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        input_ids = input_ids[:, -1:]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update({
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "attention_mask": attention_mask,
        "images": kwargs.get("images", None),
        "cap_tensor": kwargs.get("cap_tensor", None),
        "the": kwargs.get("the", None),
        "gamma_gain": kwargs.get("gamma_gain", None),
        "gamma_reduce": kwargs.get("gamma_reduce", None),
        "input_cap_ids": kwargs.get("input_cap_ids", None),
        "gain_per": kwargs.get("gain_per", None),
        "reduce_per": kwargs.get("reduce_per", None),
        "use_cd_branch": False,
        "bias_weight": kwargs.get("bias_weight", None),
        "bias_sample_num": kwargs.get("bias_sample_num", None),
    })
    return model_inputs


# ---------------------------------------------------------------------------
# prepare_inputs_for_generation_cd (contrastive branch)
# ---------------------------------------------------------------------------

def _prepare_inputs_for_generation_cd(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        input_ids = input_ids[:, -1:]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update({
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "attention_mask": attention_mask,
        "images": kwargs.get("images_cd", None),
        "cap_tensor": kwargs.get("cap_tensor", None),
        "the": kwargs.get("the", None),
        "gamma_gain": kwargs.get("gamma_gain", None),
        "gamma_reduce": kwargs.get("gamma_reduce", None),
        "input_cap_ids": kwargs.get("input_cap_ids", None),
        "gain_per": kwargs.get("gain_per", None),
        "reduce_per": kwargs.get("reduce_per", None),
        "use_cd_branch": True,
        "bias_weight": kwargs.get("bias_weight", None),
        "bias_sample_num": kwargs.get("bias_sample_num", None),
    })
    return model_inputs


# ---------------------------------------------------------------------------
# _shield_prepare_multimodal -- the core SHIELD vision-language fusion
# ---------------------------------------------------------------------------

def _shield_prepare_multimodal(
    model,
    input_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    *,
    cap_tensor,
    the,
    gamma_gain,
    gamma_reduce,
    input_cap_ids,
    gain_per,
    reduce_per,
    use_cd_branch,
    bias_weight,
    bias_sample_num,
):
    """Build ``inputs_embeds`` with SHIELD feature enhancement.

    This replaces the need to modify ``llava_arch.py`` -- it accesses
    the model's vision tower, projector and embed_tokens through the
    standard LLaVA API.
    """
    vision_tower = model.get_model().get_vision_tower()

    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if (
            past_key_values is not None
            and vision_tower is not None
            and images is not None
            and input_ids.shape[1] == 1
        ):
            if hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = past_key_values[-1][-1].shape[-2]
            attention_mask = torch.ones(
                (attention_mask.shape[0], past_len + 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        return input_ids, attention_mask, past_key_values, None, labels

    # Split encode_images so SHIELD weighting sits between vision tower
    # and projector.
    image_features = vision_tower(images)

    if use_cd_branch:
        top_k_indices = None
    else:
        image_features, top_k_indices = compute_shield_image_weights(
            image_features,
            cap_tensor,
            the,
            gamma_gain,
            gain_per,
            bias_weight,
            get_bias(int(bias_sample_num), images, vision_tower),
        )

    image_features = model.get_model().mm_projector(image_features.half())

    # ------------------------------------------------------------------
    # Build new input embeddings with image (and optionally caption)
    # tokens replacing the IMAGE_TOKEN placeholder.
    # ------------------------------------------------------------------
    new_input_embeds = []
    new_labels = [] if labels is not None else None
    cur_image_idx = 0

    for batch_idx, cur_input_ids in enumerate(input_ids):
        if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            half_len = cur_input_ids.shape[0] // 2
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = model.get_model().embed_tokens(cur_input_ids[:half_len])
            cur_input_embeds_2 = model.get_model().embed_tokens(cur_input_ids[half_len:])
            cur_input_embeds = torch.cat(
                [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0
            )
            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_input_embeds = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_new_labels = []
            assert cur_labels.shape == cur_input_ids.shape

        while image_token_indices.numel() > 0:
            cur_image_features = image_features[cur_image_idx]
            image_token_start = image_token_indices[0]

            use_im_start_end = getattr(model.config, "tune_mm_mlp_adapter", False) and getattr(
                model.config, "mm_use_im_start_end", False
            )

            if use_im_start_end:
                cur_new_input_embeds.append(
                    model.get_model().embed_tokens(cur_input_ids[: image_token_start - 1]).detach()
                )
                cur_new_input_embeds.append(
                    model.get_model().embed_tokens(cur_input_ids[image_token_start - 1 : image_token_start])
                )
                cur_new_input_embeds.append(cur_image_features)
                cur_new_input_embeds.append(
                    model.get_model().embed_tokens(cur_input_ids[image_token_start + 1 : image_token_start + 2])
                )
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_new_labels.append(cur_labels[image_token_start : image_token_start + 1])
                    cur_labels = cur_labels[image_token_start + 2 :]
            else:
                cur_new_input_embeds.append(
                    model.get_model().embed_tokens(cur_input_ids[:image_token_start])
                )
                cur_new_input_embeds.append(cur_image_features)

                if not use_cd_branch:
                    input_cap_tensor = model.get_model().embed_tokens(input_cap_ids[batch_idx])
                    selected_elements = map_text_segments(
                        top_k_indices, cap_tensor, input_cap_tensor
                    )
                    cur_new_input_embeds.append(selected_elements)

                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[image_token_start + 1 :]

            cur_image_idx += 1
            if use_im_start_end:
                cur_input_ids = cur_input_ids[image_token_start + 2 :]
            else:
                cur_input_ids = cur_input_ids[image_token_start + 1 :]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

        if cur_input_ids.numel() > 0:
            if use_im_start_end:
                cur_new_input_embeds.append(model.get_model().embed_tokens(cur_input_ids).detach())
            else:
                cur_new_input_embeds.append(model.get_model().embed_tokens(cur_input_ids))
            if labels is not None:
                cur_new_labels.append(cur_labels)

        cur_new_input_embeds = [x.to(device=model.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        new_input_embeds.append(cur_new_input_embeds)
        if labels is not None:
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

    # ------------------------------------------------------------------
    # Align / pad across the batch
    # ------------------------------------------------------------------
    if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        max_len = max(x.shape[0] for x in new_input_embeds)

        new_input_embeds_align = []
        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

        if labels is not None:
            new_labels_align = []
            _new_labels = new_labels
            for cur_new_label in new_labels:
                cur_new_label = torch.cat(
                    (
                        cur_new_label,
                        torch.full(
                            (max_len - cur_new_label.shape[0],),
                            IGNORE_INDEX,
                            dtype=cur_new_label.dtype,
                            device=cur_new_label.device,
                        ),
                    ),
                    dim=0,
                )
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)

        if attention_mask is not None:
            new_attention_mask = []
            for cur_attention_mask, cur_nl, cur_nl_align in zip(
                attention_mask, _new_labels, new_labels
            ):
                pad_left = torch.full(
                    (cur_nl.shape[0] - labels.shape[1],),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                pad_right = torch.full(
                    (cur_nl_align.shape[0] - cur_nl.shape[0],),
                    False,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                cur_new_attention_mask = torch.cat(
                    (pad_left, cur_attention_mask, pad_right), dim=0
                )
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            assert attention_mask.shape == new_labels.shape
    else:
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)

        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full(
                (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            assert attention_mask.shape == new_input_embeds.shape[:2]

    return None, attention_mask, past_key_values, new_input_embeds, new_labels
