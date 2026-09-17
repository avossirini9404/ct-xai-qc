"""Classifiers.

A deliberately small convolutional network is the default. The object of study
here is the explanation and its behaviour under acquisition changes, not the last
point of accuracy, and a small model keeps the whole pipeline inside the one-hour
budget stated in the README. ``resnet18`` is available for a sanity comparison
when torchvision is installed.

Both models expose ``feature_layer``, the last convolutional block, which is
where Grad-CAM taps the activations and gradients.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Three convolutional blocks, global average pooling, linear head.

    Dropout is placed before the classifier so that the same trained network can
    be used for Monte-Carlo dropout at test time without retraining.
    """

    def __init__(self, n_classes: int, in_channels: int = 1, width: int = 16, dropout: float = 0.3) -> None:
        super().__init__()
        self.block1 = _block(in_channels, width)
        self.block2 = _block(width, width * 2)
        self.block3 = _block(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width * 4, n_classes)

    @property
    def feature_layer(self) -> nn.Module:
        """Layer whose activations Grad-CAM uses."""
        return self.block3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.block3(x)
        x = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return self.classifier(self.dropout(x))


def build_model(name: str, n_classes: int, in_channels: int = 1, dropout: float = 0.3, **kwargs) -> nn.Module:
    """Construct a model by name."""
    name = name.lower()
    if name in ("smallcnn", "small_cnn", "cnn"):
        return SmallCNN(n_classes=n_classes, in_channels=in_channels, dropout=dropout, **kwargs)
    if name == "resnet18":
        try:
            from torchvision.models import resnet18
        except ImportError as exc:  # pragma: no cover
            raise ImportError("resnet18 requires torchvision; `pip install torchvision`") from exc
        model = resnet18(weights=None, num_classes=n_classes)
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.feature_layer = model.layer4  # type: ignore[attr-defined]
        return model
    raise KeyError(f"unknown model {name!r}")


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Put the model in eval mode but keep dropout active, for MC-dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()
    return model
