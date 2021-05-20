from typing import Tuple, List, Union

import numpy as np
from matplotlib import pyplot as plt


def gaussian(x: int, y: int, height: int, width: int, sigma: int = 5) -> 'np.ndarray':
    channel = [np.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(height) for c in range(width)]
    channel = np.array(channel, dtype=np.float32).reshape((height, width))
    return channel


def generate_position_map(shape: Tuple[int, int], bounding_boxes_centers: List[List[int]],
                          sigma: int = 2, return_as_one: bool = True) -> Union[List['np.ndarray'], 'np.ndarray']:
    h, w = shape
    masks = []
    for center in bounding_boxes_centers:
        x, y = center
        masks.append(gaussian(x=x, y=y, height=h, width=w, sigma=sigma))
    if return_as_one:
        return np.stack(masks).sum(axis=0)
    return masks


if __name__ == '__main__':
    s = (480, 320)
    b_centers = [[50, 50], [76, 82], [12, 67], [198, 122]]
    out = generate_position_map(s, b_centers, sigma=5)
    plt.imshow(out)
    plt.show()
    print()
