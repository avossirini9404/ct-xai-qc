import numpy as np
import pytest


@pytest.fixture(scope="session")
def phantom() -> np.ndarray:
    """A small circular phantom with a bright insert, in [0, 1]."""
    size = 48
    yy, xx = np.mgrid[:size, :size]
    centre = (size - 1) / 2.0
    body = ((yy - centre) ** 2 + (xx - centre) ** 2) <= (0.40 * size) ** 2
    img = np.where(body, 0.55, 0.05).astype(np.float32)
    insert = ((yy - centre) ** 2 + (xx - centre) ** 2) <= (0.10 * size) ** 2
    img[insert] = 0.85
    return img
