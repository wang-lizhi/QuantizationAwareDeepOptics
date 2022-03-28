from collections import namedtuple

import numpy as np

"""
RGGB
"""
BAYER_RGGB_CHANNEL_LOC_MAP = {
    'r': (0, 0),
    'g1': (0, 1),
    'g2': (1, 0),
    'b': (1, 1)
}


def generate_rggb_pattern_map(height, width, channel, is_bool=False):
    assert channel == 'r' or channel == 'g1' or channel == 'g2' or channel == 'b', \
        "Invalid RGGB Bayer channel. The `channel` must be in one of {r, g1, g2, b}."
    period_length = 2
    start_row, start_col = BAYER_RGGB_CHANNEL_LOC_MAP[channel]
    map = np.zeros((height, width), dtype=np.bool if is_bool else np.float32)
    for row in range(height):
        if row % period_length == start_row:
            for col in range(width):
                if col % period_length == start_col:
                    map[row][col] = True if is_bool else 1
    return map


BayerPatternMask = namedtuple('BayerPattern', ['r', 'g1', 'g2', 'b'])


def get_square_bayer_rggb_pattern_map(resolution, is_bool=False):
    return BayerPatternMask(r=generate_rggb_pattern_map(resolution, resolution, 'r', is_bool=is_bool),
                            g1=generate_rggb_pattern_map(resolution, resolution, 'g1', is_bool=is_bool),
                            g2=generate_rggb_pattern_map(resolution, resolution, 'g2', is_bool=is_bool),
                            b=generate_rggb_pattern_map(resolution, resolution, 'b', is_bool=is_bool))


RGGB_BAYER_PATTERN_MASK_PRELOAD_RESOLUTIONS = [512, 1024]

RGGB_BAYER_PATTERN_MASK = {}

RGGB_BAYER_PATTERN_BOOLEAN_MASK = {}

for resolution in RGGB_BAYER_PATTERN_MASK_PRELOAD_RESOLUTIONS:
    RGGB_BAYER_PATTERN_MASK[resolution] = get_square_bayer_rggb_pattern_map(resolution, False)
    RGGB_BAYER_PATTERN_BOOLEAN_MASK[resolution] = get_square_bayer_rggb_pattern_map(resolution, True)

