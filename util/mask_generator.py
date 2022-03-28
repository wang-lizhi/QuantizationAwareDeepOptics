import numpy as np


def circle_mask(full_square_length, inner_circle_radius=None):
    """
    Pixel outside the circle is True
    Args:
        full_square_length:
        inner_circle_radius:

    Returns:

    """
    assert full_square_length > inner_circle_radius
    [_x, _y] = np.mgrid[-full_square_length // 2: full_square_length // 2,
               -full_square_length // 2: full_square_length // 2].astype(np.float32)
    r = np.sqrt(_x ** 2 + _y ** 2)[None, :, :, None]
    if inner_circle_radius is None:
        inner_circle_radius = np.amax(_x)
    mask = (r > inner_circle_radius).astype(np.float32)
    return mask
