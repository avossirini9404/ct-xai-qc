"""Sanity checks for saliency methods.

Adebayo et al. (2018) showed that some widely used saliency methods produce
almost the same map after the weights of the network have been randomised, which
means the map is largely a property of the image and the architecture rather than
of what the trained model learned.

In the vocabulary of a quality-assurance programme this is a negative control: an
instrument that reports the same reading with the sample removed is not measuring
the sample. A method that survives this test unchanged should not be used to
justify a clinical decision, however convincing the picture looks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .stability import saliency_agreement


def _randomisable_layers(model) -> list[str]:
    """Names of parameterised layers, from the classifier backwards."""
    from torch import nn

    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and any(p.requires_grad for p in module.parameters(recurse=False))
    ]
    return list(reversed(names))


def cascading_randomization(
    model,
    x,
    explainer: Callable[..., np.ndarray],
    target: int | None = None,
    seed: int = 0,
    **explainer_kwargs,
) -> list[dict[str, float]]:
    """Randomise layers one at a time from the output backwards.

    After each layer is re-initialised the saliency map is recomputed and compared
    with the map from the trained model. Agreement that stays high all the way
    down is the failure mode the test is designed to expose.

    The model passed in is left untouched; the randomisation is applied to a copy.
    """
    import copy

    import torch

    torch.manual_seed(seed)
    reference_model = model
    working = copy.deepcopy(model)
    reference = explainer(reference_model, x, target=target, **explainer_kwargs)

    results: list[dict[str, float]] = []
    for name in _randomisable_layers(working):
        module = dict(working.named_modules())[name]
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        else:  # pragma: no cover - every Conv2d and Linear has one
            for param in module.parameters(recurse=False):
                torch.nn.init.normal_(param, std=0.05)
        randomised = explainer(working, x, target=target, **explainer_kwargs)
        record = {"layer": name}
        record.update(saliency_agreement(reference, randomised))
        results.append(record)
    return results
