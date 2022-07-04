import copy
from torchvision.transforms.functional import rotate


def extensive_rotation_tensor(inputs, angles=None):
    """Rotate tensor in arbitrary angles"""
    final = copy.deepcopy(inputs)

    for key, input in enumerate(inputs):
        r_tensor = rotate(input, angles)
        final[key] = r_tensor
    return final