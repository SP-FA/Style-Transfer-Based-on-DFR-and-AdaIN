from torchvision.transforms.functional import rotate
from typing import *


def _DFR(mat, angles: List[float]) -> List:
    """
    Rotate tensor with multiple angles.

    PARAMETER:
      @ mat: input matrix
      @ angles: a list of angles

    RETURN:
      A list witch each element is a rotated matrix.
    """
    mats = []
    for angle in angles:
        mats.append(rotate(mat, angle))
    return mats
