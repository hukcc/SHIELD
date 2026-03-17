"""Adversarial attack helpers (CW and PGD) operating in CLIP space."""

import torch
import torch.optim as optim

from .clip_utils import CLIP_MEAN, CLIP_STD, load_clip_model


def apply_perturbation(image_tensor, perturbation):
    """Apply a perturbation to an image tensor within CLIP normalisation bounds."""
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    normalized_min = (0 - mean) / std
    normalized_max = (1 - mean) / std

    return torch.clamp(image_tensor + perturbation, normalized_min, normalized_max)


def perform_clip_attack(
    image,
    image_tensor,
    text,
    epsilon,
    num_steps,
    c,
    lr=0.1,
    attack_type="cw",
    clip_model=None,
    clip_processor=None,
):
    """Run the full CLIP attack pipeline and return the perturbed tensor."""
    if clip_model is None or clip_processor is None:
        clip_model, clip_processor = load_clip_model()

    inputs = clip_processor(
        text=text, images=image, return_tensors="pt", padding=True
    ).to("cuda")

    if attack_type == "cw":
        perturbation = cw_attack(
            clip_model, inputs, epsilon=epsilon, num_steps=num_steps, c=c, lr=lr
        )
    elif attack_type == "pgd":
        perturbation = pgd_attack(
            clip_model, inputs, epsilon=epsilon, num_steps=num_steps
        )
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    return apply_perturbation(image_tensor, perturbation)


def cw_attack(model, inputs, epsilon, num_steps, c, lr=0.1, random_init=False):
    """Carlini-Wagner style attack with CLIP-range clamping."""
    if random_init:
        delta = torch.empty_like(inputs["pixel_values"]).requires_grad_(True)
        torch.nn.init.xavier_uniform_(delta)
    else:
        delta = torch.zeros_like(
            inputs["pixel_values"],
            requires_grad=True,
            device=inputs["pixel_values"].device,
        )

    optimizer = optim.Adam([delta], lr=lr)
    ori = inputs["pixel_values"].clone()

    for _step in range(num_steps):
        mean = (
            torch.tensor(CLIP_MEAN)
            .view(3, 1, 1)
            .to(inputs["pixel_values"].device)
        )
        std = (
            torch.tensor(CLIP_STD)
            .view(3, 1, 1)
            .to(inputs["pixel_values"].device)
        )
        normalized_min = (0 - mean) / std
        normalized_max = (1 - mean) / std
        inputs["pixel_values"] = torch.clamp(
            ori + c * delta, normalized_min, normalized_max
        )

        outputs = model(**inputs)

        loss = outputs.logits_per_image[0][0]
        if loss < 0:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        delta.data = torch.clamp(delta.data, -epsilon, epsilon)

    return c * delta.detach().squeeze(0).cpu()


def pgd_attack(model, inputs, epsilon, alpha, num_steps):
    """Projected gradient descent attack within CLIP bounds."""
    ori_input = inputs["pixel_values"].clone()
    inputs["pixel_values"].requires_grad = True

    perturbation = torch.zeros_like(
        inputs["pixel_values"], device=inputs["pixel_values"].device
    )
    mean = (
        torch.tensor(CLIP_MEAN)
        .view(1, 3, 1, 1)
        .to(inputs["pixel_values"].device)
    )
    std = (
        torch.tensor(CLIP_STD)
        .view(1, 3, 1, 1)
        .to(inputs["pixel_values"].device)
    )

    normalized_min = (0 - mean) / std
    normalized_max = (1 - mean) / std

    for _step in range(num_steps):
        outputs = model(**inputs)
        loss = -outputs.logits_per_image[0][0]

        grad = torch.autograd.grad(
            loss, inputs["pixel_values"], retain_graph=False, create_graph=False
        )[0]

        with torch.no_grad():
            perturbation += alpha * grad.sign()
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)

            inputs["pixel_values"] = torch.clamp(
                ori_input + perturbation, normalized_min, normalized_max
            ).detach()
            inputs["pixel_values"].requires_grad = True

        if loss > 0:
            break

    return perturbation.squeeze(0).cpu()
