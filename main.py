import numpy as np
from scipy.spatial.transform import special_ortho_group

class Rotations:
    @staticmethod
    def generate_random_rotation(dim=3):
        return special_ortho_group.rvs(dim)

